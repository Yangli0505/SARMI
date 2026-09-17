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


class IPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        obs_dim = (obs_dim[0] - 1) * obs_dim[1]
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_ret_buf = np.zeros(size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.re_a_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size


    def store(self, obs, act, rew, cost, val, cost_val, logp, re_a):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_buf[self.ptr] = val
        self.cost_val_buf[self.ptr] = cost_val
        self.logp_buf[self.ptr] = logp
        self.re_a_buf[self.ptr] = re_a

        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        cost_vals = np.append(self.cost_val_buf[path_slice], last_cost_val)
        cost_deltas = costs[:-1] + self.gamma * cost_vals[1:] - cost_vals[:-1]
        self.cost_adv_buf[path_slice] = core.discount_cumsum(cost_deltas, self.gamma * self.lam)
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cost_adv_mean, cost_adv_std = mpi_statistics_scalar(self.cost_adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        self.cost_adv_buf = (self.cost_adv_buf - cost_adv_mean) / (cost_adv_std + 1e-8)
        data = dict(
            obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, re_a=self.re_a_buf,
            cost_ret=self.cost_ret_buf, cost_adv=self.cost_adv_buf
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ipo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=1e-4, vf_lr=1e-4,
        lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=1,
        cost_limit=25, re_action=False, config=None):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    env = env.unwrapped
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = IPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cost_adv, logp_old = data['obs'], data['act'], data['adv'], data['cost_adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        # IPO核心：引入 barrier/cost 项
        barrier = torch.mean(torch.exp(cost_adv))  # 可根据论文调整
        loss_pi = -(ratio * adv).mean() + barrier

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + 0.2) | ratio.lt(1 - 0.2)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up function for computing cost value loss
    def compute_loss_cost_v(data):
        obs, cost_ret = data['obs'], data['cost_ret']
        return ((ac.vc(obs) - cost_ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_lr = 1e-4
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_lr = 1e-4
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cost_vf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(re_action):
        data = buf.get()
        loss_pi, pi_info = compute_loss_pi(data)
        pi_optimizer.zero_grad()
        loss_pi.backward()
        mpi_avg_grads(ac.pi)
        pi_optimizer.step()

        # Value function learning
        for _ in range(80):
            loss_v = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

            loss_cost_v = compute_loss_cost_v(data)
            cost_vf_optimizer.zero_grad()
            loss_cost_v.backward()
            mpi_avg_grads(ac.vc)
            cost_vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info['ent'], pi_info['cf']
        logger.store(LossPi=loss_pi.item(), LossV=loss_v.item(), LossCostV=loss_cost_v.item(),
                     KL=kl, Entropy=ent, ClipFrac=cf)

    def save(args, save_name, model, wandb, ep=None):
        import os
        save_dir = './trained_models/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not ep == None:
            torch.save(model, save_dir + args.proj_name + '_' + args.run_name + save_name + str(ep) + ".pt")
            wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
        else:
            torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
            wandb.save(save_dir + args.run_name + save_name + ".pth")

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_cost, ep_len, crash_counter, tra_counter = env.reset(), 0, 0, 0, 0, 0

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):
        wandb.watch(ac, log="gradients", log_freq=10)
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                o = o[1:].flatten()
                a, v, vc, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
                re_a = a

                if re_action:
                    re_a = env.check_action(int(a))
                    pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, info = env.step(int(a))
                c = info['cost']
                if info.get('crashed', False):
                    crash_counter += 1

                r = r - c
                ep_ret += r
                ep_cost += c
                ep_len += 1

                # save and log
                buf.store(o, a, r, c, v, vc, logp, re_a)
                logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        o = o[1:].flatten()
                        _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v, vc = 0, 0
                    buf.finish_path(last_val=v, last_cost_val=vc)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                        tra_counter += 1
                    o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

            # Perform PPO update!
            update(re_action)

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)
                save(config, save_name="_", model=ac, wandb=wandb, ep=config.seed)

            crash_ratio = mpi_sum(crash_counter) / mpi_sum(tra_counter)
            logger.store(Crash_counter=mpi_sum(crash_counter))
            crash_counter, tra_counter = 0, 0
            # Log info about epoch
            # ep_ret_stats = logger.get_stats('EpRet')
            # ep_cost_stats = logger.get_stats('EpCost')
            # ep_len_stats = logger.get_stats('EpLen')
            # loss_pi_stats = logger.get_stats('LossPi')
            # wandb.log({
            #     "AverageRewards": ep_ret_stats[0] if len(ep_ret_stats) > 0 else 0,
            #     "AverageCost": ep_cost_stats[0] if len(ep_cost_stats) > 0 else 0,
            #     "EpLen": ep_len_stats[0] if len(ep_len_stats) > 0 else 0,
            #     "Steps": (epoch + 1) * steps_per_epoch,
            #     "Policy Loss": loss_pi_stats[0] if len(loss_pi_stats) > 0 else 0,
            #     "Crash_ratio": crash_ratio,
            # })

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Crash_counter')
            logger.log_tabular('Crash_ratio', crash_ratio)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossCostV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='merge_game_env-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=5) # 1,2,3,4,5
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--max_ep_len', type=int, default=4000)
    parser.add_argument('--num_steps', type=int, default=5e5)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--cost_limit', type=float, default=0.01)
    parser.add_argument('--safe_check', type=bool, default=False)
    parser.add_argument('--run_name', type=str, default='IPO')
    parser.add_argument('--proj_name', type=str, default='Baseline')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(data_dir=os.path.join('data',args.proj_name), 
                                        exp_name=args.run_name, 
                                        seed=args.seed)

    epochs = int(args.num_steps / args.steps_per_epoch)

    ipo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma, max_ep_len=args.max_ep_len,
        seed=args.seed, steps_per_epoch=args.steps_per_epoch, epochs=epochs, cost_limit=args.cost_limit,
        re_action=args.safe_check, logger_kwargs=logger_kwargs, config=args)
