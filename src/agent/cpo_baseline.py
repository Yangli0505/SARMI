import numpy as np
import torch
from torch.optim import Adam
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import highway_env
import gym
import time
import core
import wandb
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from torch.nn.functional import softplus

torch.autograd.set_detect_anomaly(True)

class CPOBuffer:
    """Buffer for storing trajectories for CPO with cost constraints"""
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97, cost_gamma=0.99, cost_lam=0.97):
        obs_dim = (obs_dim[0]-1)*obs_dim[1]
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)
        self.re_a_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, re_a, cost, cval):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.re_a_buf[self.ptr] = re_a
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma*self.lam)

        cdeltas = costs[:-1] + self.cost_gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.cost_gamma*self.cost_lam)

        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cval_buf[path_slice] = core.discount_cumsum(costs, self.cost_gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) / cadv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, re_a=self.re_a_buf,
                    cadv=self.cadv_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def cpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, cost_limit=25,
        lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(),
        save_freq=1, config=None):
    """Constrained Policy Optimization"""
    setup_pytorch_for_mpi()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_fn().unwrapped
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac)
    
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = CPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, cost_gamma=gamma, cost_lam=lam)

    pi_optimizer = Adam(ac.pi.parameters(), lr=1e-4)
    vf_optimizer = Adam(ac.v.parameters(), lr=1e-4)

    logger.setup_pytorch_saver(ac)

    def compute_loss_pi_cpo(data):
        obs, act, adv, logp_old, cadv = data['obs'], data['act'], data['adv'], data['logp'], data['cadv']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        surrogate = (ratio * adv).mean()
        cost_surrogate = (ratio * cadv).mean()
        loss_pi = -surrogate
        return loss_pi, cost_surrogate, pi, logp

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    def update():
        data = buf.get()
        # 1. compute policy and cost
        loss_pi, cost_surrogate, _, _ = compute_loss_pi_cpo(data)
        v_l_old = compute_loss_v(data).item()
        # 2. Policy update: simplified (真实CPO需用QP solver)
        pi_optimizer.zero_grad()
        loss_pi.backward()
        mpi_avg_grads(ac.pi)
        pi_optimizer.step()
        # 3. Value update
        train_v_iters = 80
        for _ in range(train_v_iters):
            loss_v = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()
        # 4. Logging
        logger.store(LossPi=loss_pi.item(), LossV=v_l_old, CostSurrogate=cost_surrogate.item())

    def save(args, save_name, model, wandb, ep=None):
        save_dir = './trained_models/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if ep is not None:
            torch.save(model, save_dir + args.proj_name + '_' + args.run_name + save_name + str(ep) + ".pt")
            wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
        else:
            torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
            wandb.save(save_dir + args.run_name + save_name + ".pth")

    start_time = time.time()
    o, ep_ret, ep_cost, ep_len, crash_counter, tra_counter = env.reset(), 0, 0, 0, 0, 0

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):
        wandb.watch(ac, log="gradients", log_freq=10)
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                o = o[1:].flatten()
                a, v, _, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
                re_a = a
                next_o, r, d, info = env.step(int(a))
                c = info['cost']
                r = r - c
                ep_ret += r
                ep_cost += c
                ep_len += 1
                if info['crashed']:
                    crash_counter += 1
                buf.store(o, a, r, v, logp, re_a, c, cval=0)
                o = next_o
                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch-1
                if terminal or epoch_ended:
                    if timeout or epoch_ended:
                        o = o[1:].flatten()
                        _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    buf.finish_path(last_val=v, last_cval=0)
                    if terminal:
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        tra_counter += 1
                    o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0
            update()
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)
                save(config, save_name="_", model=ac, wandb=wandb, ep=config.seed)

            crash_ratio = mpi_sum(crash_counter)/max(1, mpi_sum(tra_counter))
            logger.store(Crash_counter=mpi_sum(crash_counter))
            crash_counter, tra_counter = 0, 0
            wandb.log({"AverageRewards": logger.get_stats('EpRet')[0],
                       "AverageCost": logger.get_stats('EpCost')[0],
                       "Steps": (epoch+1)*steps_per_epoch,
                       "Policy Loss": logger.get_stats('LossPi')[0],
                       "Crash_ratio": crash_ratio})
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Crash_counter')
            logger.log_tabular('Crash_ratio', crash_ratio)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='merge_game_env-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--max_ep_len', type=int, default=4000)
    parser.add_argument('--num_steps', type=int, default=5e5)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--cost_limit', type=float, default=0.01)
    parser.add_argument('--run_name', type=str, default='CPO')
    parser.add_argument('--proj_name', type=str, default='Baseline')
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(data_dir=os.path.join('data',args.proj_name),
                                        exp_name=args.run_name,
                                        seed=args.seed)

    epochs = int(args.num_steps / args.steps_per_epoch)

    cpo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        max_ep_len=args.max_ep_len, seed=args.seed, steps_per_epoch=args.steps_per_epoch,
        epochs=epochs, cost_limit=args.cost_limit, logger_kwargs=logger_kwargs,
        config=args)
