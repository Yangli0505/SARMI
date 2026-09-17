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


class TRPOBuffer:
    """
    与 PPOBuffer 一致，用于 GAE(λ) 优势估计；保留 cost 通道与 re_action（可选）。
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        obs_dim = (obs_dim[0] - 1) * obs_dim[1]
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.re_a_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, crew, val, cval, logp, re_a):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew
        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        self.re_a_buf[self.ptr] = re_a
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        # 对 cost-adv 不做标准化（或只做去均值），保持与 PPO 版本一致
        self.cadv_buf = (self.cadv_buf - cadv_mean)

        data = dict(
            obs=self.obs_buf, act=self.act_buf,
            ret=self.ret_buf, cret=self.cret_buf,
            adv=self.adv_buf, cadv=self.cadv_buf,
            logp=self.logp_buf, re_a=self.re_a_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def flat_params(module):
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def flat_grad(y, module, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(y, [p for p in module.parameters() if p.requires_grad],
                                retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.contiguous().view(-1) for g in grads])


def set_params(module, flat_vector):
    idx = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat_vector[idx:idx+n].view_as(p))
        idx += n


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (torch.dot(p, Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.dot(r, r)
        if rr_new < tol:
            break
        beta = rr_new / (rr + 1e-10)
        p = r + beta * p
        rr = rr_new
    return x


def trpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97, max_ep_len=1000,
         target_kl=0.01, backtrack_coeff=0.8, backtrack_iters=10, damping=0.1,
         vf_lr=1e-3, train_v_iters=80, logger_kwargs=dict(), save_freq=1,
         cost_limit=25.0, re_action=False, config=None):
    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    env = env.unwrapped
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, ac.vc])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = TRPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # 价值函数优化器
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)

    # 拉格朗日乘子参数（无约束参数 -> softplus）
    penalty_param = torch.tensor(1.0, requires_grad=True).float()
    penalty_optimizer = Adam([penalty_param], lr=5e-2)

    logger.setup_pytorch_saver(ac)

    def surrogate_losses(data, penalty_param_tensor):
        obs, act, adv, cadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'], data['logp']
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        Lr = (ratio * adv).mean()
        Lc = (ratio * cadv).mean()

        lam_val = softplus(penalty_param_tensor)
        surr = Lr - lam_val * Lc   # maximize
        # 额外信息
        with torch.no_grad():
            ent = pi.entropy().mean()
            approx_kl = (logp_old - logp).mean()
            clipfrac = (ratio.gt(1+0.2) | ratio.lt(1-0.2)).float().mean()  # 非关键指标，仅作参考
        info = dict(Lr=Lr.item(), Lc=Lc.item(), ent=ent.item(), kl=approx_kl.item(), cf=clipfrac.item())
        return surr, info

    def mean_kl(old_pi_dist, new_pi_dist):
        # KL(old || new)，对每个样本求和后取均值（对多维动作）
        kl = torch.distributions.kl_divergence(old_pi_dist, new_pi_dist)
        # kl 可能为 (batch, actdim) 或 (batch,)，统一做 sum(-1) 再 mean
        if kl.dim() == 2:
            kl = kl.sum(-1)
        return kl.mean()

    def get_fvp_func(obs_batch):
        # Fisher 向量积函数：Av = (∇^2 KL) v + damping * v
        @torch.no_grad()
        def build_old_dist(obs):
            # 旧策略分布（冻结），不跟踪梯度
            return ac.pi._distribution(obs)

        # 旧分布（detach）
        with torch.no_grad():
            old_dist = build_old_dist(obs_batch)

        def fvp(v):
            # 计算 new_dist = ac.pi._distribution(obs)
            new_dist = ac.pi._distribution(obs_batch)
            kl = mean_kl(old_dist, new_dist)

            # 一阶梯度 g_kl = ∇_θ KL
            grads = torch.autograd.grad(kl, [p for p in ac.pi.parameters() if p.requires_grad],
                                        create_graph=True, retain_graph=True)
            flat_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads])

            # 向量积 (∇^2 KL) v = ∇ (g_kl · v)
            kl_v = (flat_grad_kl * v).sum()
            hvp_grads = torch.autograd.grad(kl_v, [p for p in ac.pi.parameters() if p.requires_grad],
                                            retain_graph=True)
            flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])
            return flat_hvp + damping * v
        return fvp

    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        v_loss = ((ac.v(obs) - ret) ** 2).mean()
        vc_loss = ((ac.vc(obs) - cret) ** 2).mean()
        return v_loss, vc_loss

    def line_search(old_params, full_step, max_backtracks, accept_ratio, data, old_pi_info):
        """
        线搜索：沿自然梯度方向回退，满足 KL 限制且 surrogate 改善。
        """
        fval_old = -old_pi_info['surr']  # 我们对 -surr 做“loss”比较（越小越好）
        for stepfrac in [backtrack_coeff ** i for i in range(max_backtracks)]:
            new_params = old_params + stepfrac * full_step
            set_params(ac.pi, new_params)
            # 重新计算 surr 与 KL
            surr_new, info_new = surrogate_losses(data, penalty_param)
            with torch.no_grad():
                # 用旧分布估 KL：old_dist (detach) 与 new_dist
                old_dist = old_pi_info['old_dist']
                new_dist = ac.pi._distribution(data['obs'])
                kl_new = mean_kl(old_dist, new_dist).item()
                improve = (surr_new.item() - old_pi_info['surr'])
                loss_improve = -surr_new.item() - fval_old  # 越负越好
                # 接受条件：KL 合规 + surr 改善
                if (kl_new <= target_kl) and (improve > 0) and (loss_improve < 0):
                    return True, new_params, dict(kl=kl_new, improve=improve, info=info_new)
        # 回退失败，恢复旧参数
        set_params(ac.pi, old_params)
        return False, old_params, dict(kl=None, improve=0.0, info=None)

    def update(re_action_flag):
        # 统计当前 epoch 的平均成本，用于更新 λ
        cur_cost = logger.get_stats('EpCost')[0]

        data = buf.get()
        data['cur_cost'] = cur_cost

        # 先更新 penalty_param（拉格朗日乘子无约束参数）
        # maximize  penalty_param * (cur_cost - cost_limit)
        loss_penalty = -penalty_param * (cur_cost - cost_limit)
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        # 单参数也用 mpi_avg_grads 以保持接口统一
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()
        lam_val = softplus(penalty_param).item()
        logger.store(Penalty=lam_val)

        # ========= TRPO 主更新：策略 =========
        # 计算当前 surr 及信息
        surr, surr_info = surrogate_losses(data, penalty_param)
        with torch.no_grad():
            old_dist = ac.pi._distribution(data['obs'])

        old_params = flat_params(ac.pi).clone()

        # loss := -surr（做最小化）
        loss = -surr
        g = flat_grad(loss, ac.pi, retain_graph=True)
        # 自然梯度方向：-H^{-1} g
        fvp = get_fvp_func(data['obs'])
        step_dir = conjugate_gradient(fvp, g, iters=10, tol=1e-10)
        # 方向取负号（因为是下降 g）
        step_dir = -step_dir

        # 计算步长：sqrt(2*delta / (s^T H s))
        shs = (step_dir * fvp(step_dir)).sum()
        step_scale = torch.sqrt(2 * torch.as_tensor(target_kl) / (shs + 1e-10))
        full_step = step_scale * step_dir

        # 线搜索
        accept, new_params, ls_info = line_search(
            old_params, full_step, backtrack_iters, 0.1,
            data,
            dict(surr=surr.item(), old_dist=old_dist)
        )
        if not accept:
            # 若线搜索失败，仍然记录 KL 与改变量为 0
            pi_kl = 0.0
            pi_improve = 0.0
        else:
            pi_kl = ls_info['kl']
            pi_improve = ls_info['improve']

        # ========= 价值函数更新 =========
        for _ in range(train_v_iters):
            v_loss, vc_loss = compute_loss_v(data)
            vf_optimizer.zero_grad()
            v_loss.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            vc_loss.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

        # 记录日志
        logger.store(
            LossPi=-surr.item(),   # 等价地汇报“loss”
            LossV=v_loss.item(),
            LossVC=vc_loss.item(),
            KL=pi_kl if pi_kl is not None else 0.0,
            Entropy=surr_info['ent'],
            ClipFrac=surr_info['cf'],  # TRPO 不裁剪，这里仅供参考
            DeltaLossPi=-pi_improve if accept else 0.0,
        )

    # ========= 训练主循环 =========
    start_time = time.time()
    o, ep_ret, ep_cret, ep_len, crash_counter, tra_counter = env.reset(), 0, 0, 0, 0, 0

    with wandb.init(project=config.proj_name, name=config.run_name, config=config):
        wandb.watch(ac, log="gradients", log_freq=10)

        epochs_total = epochs
        for epoch in range(epochs_total):
            for t in range(local_steps_per_epoch):
                o_proc = o[1:].flatten()
                a, v, vc, logp = ac.step(torch.as_tensor(o_proc, dtype=torch.float32))
                re_a = a

                if re_action:
                    re_a = env.check_action(int(a), "mpc")

                next_o, r, d, info = env.step(int(a))
                c = info.get('cost', 0.0)
                if info.get('crashed', False):
                    crash_counter += 1

                # reward-shaping：r - c（保持与 PPO 版本一致）
                r_eff = r - c

                ep_ret += r_eff
                ep_cret += c
                ep_len += 1

                buf.store(o_proc, a, r_eff, c, v, vc, logp, re_a)

                logger.store(VVals=v)
                logger.store(CVVals=vc)

                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == local_steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    if timeout or epoch_ended:
                        o_proc = o[1:].flatten()
                        _, v, vc, _ = ac.step(torch.as_tensor(o_proc, dtype=torch.float32))
                    else:
                        v, vc = 0, 0
                    buf.finish_path(last_val=v, last_cval=vc)
                    if terminal:
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cret)
                        tra_counter += 1
                    o, ep_ret, ep_cret, ep_len = env.reset(), 0, 0, 0

            # 策略/价值更新
            update(re_action)

            # 保存模型
            if (epoch % save_freq == 0) or (epoch == epochs_total - 1):
                logger.save_state({'env': env}, None)
                # 与 PPO 版本一致的保存函数语义
                save_dir = './trained_models/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(ac, os.path.join(save_dir, f"{config.proj_name}_{config.run_name}_{config.seed}.pt"))
                wandb.save(os.path.join(save_dir, f"{config.proj_name}_{config.run_name}_{config.seed}.pt"))

            crash_ratio = (mpi_sum(crash_counter) / max(mpi_sum(tra_counter), 1)) if tra_counter > 0 else 0.0
            logger.store(Crash_counter=mpi_sum(crash_counter))
            crash_counter, tra_counter = 0, 0

            wandb.log({
                "AverageRewards": logger.get_stats('EpRet')[0],
                "AverageCost": logger.get_stats('EpCost')[0],
                "Penalty": logger.get_stats('Penalty')[0],
                "EpLen": logger.get_stats('EpLen')[0],
                "Steps": (epoch + 1) * steps_per_epoch,
                "Policy Loss": logger.get_stats('LossPi')[0],
                "Crash_ratio": crash_ratio,
            })

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Penalty')
            logger.log_tabular('Crash_counter')
            logger.log_tabular('Crash_ratio', crash_ratio)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossVC', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='merge_game_env-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=5)  # 1,2,3,4,5
    parser.add_argument('--cpu', type=int, default=1) # 4
    parser.add_argument('--max_ep_len', type=int, default=3000)
    parser.add_argument('--num_steps', type=float, default=5e5)
    parser.add_argument('--steps_per_epoch', type=int, default=4000) # 4000
    parser.add_argument('--cost_limit', type=float, default=0.1)
    parser.add_argument('--safe_check', type=bool, default=False)
    parser.add_argument('--run_name', type=str, default='TRPO')
    parser.add_argument('--proj_name', type=str, default='Lagrangian')
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--backtrack_coeff', type=float, default=0.8)
    parser.add_argument('--backtrack_iters', type=int, default=8)  # 10
    parser.add_argument('--damping', type=float, default=0.5) # 0.1
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--train_v_iters', type=int, default=80)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.proj_name + '_' + args.run_name, args.seed)

    epochs = int(args.num_steps / args.steps_per_epoch)

    trpo(lambda: gym.make(args.env),
         actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, lam=0.97, max_ep_len=args.max_ep_len,
         seed=args.seed, steps_per_epoch=args.steps_per_epoch, epochs=epochs,
         target_kl=args.target_kl, backtrack_coeff=args.backtrack_coeff,
         backtrack_iters=args.backtrack_iters, damping=args.damping,
         vf_lr=args.vf_lr, train_v_iters=args.train_v_iters,
         cost_limit=args.cost_limit, re_action=args.safe_check,
         logger_kwargs=logger_kwargs, config=args)