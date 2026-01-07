from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
import SAC.SAC_utils as utils
from SAC.SAC_critic import DoubleQCritic as critic_model
from SAC.SAC_actor import DiagGaussianActor as actor_model


class SAC(object):
    """SAC algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        learnable_temperature=True,
        save_every=0,
        load_model=False,
        save_directory=Path("src/drl_navigation_ros2/models/SAC"),
        model_name="SAC",
        load_directory=Path("src/drl_navigation_ros2/models/SAC"),
        hidden_layers=[1024, 512],
        action_noise_std=0.2,  # 动作噪声标准差
        base_state_dim=None,  # 基础状态维度（单个时间步的状态向量长度），当使用历史state时使用
        actor_only=False,  # 是否只创建actor模型（用于数据收集进程，节省显存）
        actor_grad_clip_value=0.0,  # Actor网络梯度裁剪值（>0时启用梯度裁剪，将梯度裁剪到[-actor_grad_clip_value, actor_grad_clip_value]范围；0或负数表示不进行梯度裁剪）
        critic_grad_clip_value=0.0,  # Critic网络梯度裁剪值（>0时启用梯度裁剪，将梯度裁剪到[-critic_grad_clip_value, critic_grad_clip_value]范围；0或负数表示不进行梯度裁剪）
    ):
        super().__init__()
        self.state_dim = state_dim
        # base_state_dim用于prepare_state中计算max_bins，如果没有指定则使用state_dim
        self.base_state_dim = base_state_dim if base_state_dim is not None else state_dim
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        self.actor_only = actor_only  # 标记是否只使用actor模型
        # 记录一次训练中采样与更新的耗时（秒），用于性能分析
        self.last_sample_time = 0.0
        self.last_update_time = 0.0

        def _init_weights_he(module: nn.Module):
            """使用 He(Kaiming) 初始化线性层权重（仅在不加载已有模型时使用）"""
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # 创建actor模型（总是需要）
        self.actor = actor_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        # 若不加载已有模型，则对 Actor 使用 He 初始化
        if not load_model:
            _init_weights_he(self.actor)
        # print(f"Actor model initialized")
        
        # 只有在非actor_only模式下才创建critic和target_critic
        if not self.actor_only:
            self.critic = critic_model(
                obs_dim=self.state_dim,
                action_dim=action_dim,
                hidden_layers=hidden_layers,
            ).to(self.device)
            # 若不加载已有模型，则对 Critic 使用 He 初始化
            if not load_model:
                _init_weights_he(self.critic)
            # print(f"Critic model initialized")
        
            self.critic_target = critic_model(
                obs_dim=self.state_dim,
                action_dim=action_dim,
                hidden_layers=hidden_layers,
            ).to(self.device)
            # 初始时让 target_critic 与 critic 保持一致（无论是否使用 He 初始化）
            self.critic_target.load_state_dict(self.critic.state_dict())
            # print(f"Critic target model initialized")

            if load_model:
                # 当需要加载已有模型时，不再额外进行 He 初始化，直接加载权重
                self.load(filename=self.model_name, directory=load_directory)

            self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -action_dim

            # optimizers
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr, betas=actor_betas
            )

            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr, betas=critic_betas
            )

            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=alpha_lr, betas=alpha_betas
            )

            self.critic_target.train()
            self.critic.train(True)
            # 完整模式下，actor设置为train模式（用于训练）
            self.actor.train(True)
        else:
            # actor_only模式下，只初始化log_alpha（用于act方法，但不用于训练）
            self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
            self.log_alpha.requires_grad = False  # 不需要梯度
            self.target_entropy = -action_dim
            # actor_only模式下不需要优化器，因为不会进行训练
            # actor_only模式下，actor应该设置为eval模式，因为只用于推理，这样可以提高性能
            self.actor.eval()
            
            if load_model:
                self.load(filename=self.model_name, directory=load_directory)
        
        self.step = 0
        self.action_noise_std = action_noise_std
        self.actor_grad_clip_value = actor_grad_clip_value if actor_grad_clip_value > 0 else None  # 只有>0时才启用梯度裁剪
        self.critic_grad_clip_value = critic_grad_clip_value if critic_grad_clip_value > 0 else None  # 只有>0时才启用梯度裁剪
    
        # print(f"SAC initialized")


    def save(self, filename, directory):
        """保存模型权重"""
        if self.actor_only:
            # actor_only模式下只保存actor权重
            torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
            print(f"Saved actor model to: {directory}")
        else:
            # 完整模式下保存所有组件
            torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
            torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))
            torch.save(
                self.critic_target.state_dict(),
                "%s/%s_critic_target.pth" % (directory, filename),
            )
            print(f"Saved models to: {directory}")

    def load(self, filename, directory):
        """加载模型权重"""
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        if not self.actor_only:
            # 只有在非actor_only模式下才加载critic权重
            self.critic.load_state_dict(
                torch.load("%s/%s_critic.pth" % (directory, filename))
            )
            self.critic_target.load_state_dict(
                torch.load("%s/%s_critic_target.pth" % (directory, filename))
            )
        print(f"Loaded weights from: {directory}")

    def train(self, replay_buffer, iterations, batch_size):
        critic_losses = []
        actor_losses = []
        critic_grads = []
        actor_grads = []
        entropies = []
        alpha_grads = []
        sample_time_total = 0.0
        update_time_total = 0.0
        
        for _ in range(iterations):
            critic_loss, actor_loss, critic_grad, actor_grad, entropy, alpha_grad, sample_dt, total_dt = self.update(
                replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
            )
            sample_time_total += sample_dt
            update_time_total += total_dt
            if critic_loss is not None:
                critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)
            if critic_grad is not None:
                critic_grads.append(critic_grad)
            if actor_grad is not None:
                actor_grads.append(actor_grad)
            if entropy is not None:
                entropies.append(entropy)
            if alpha_grad is not None:
                alpha_grads.append(alpha_grad)

        self.step += 1

        # 记录最近一次训练的耗时统计，便于外部打印
        self.last_sample_time = sample_time_total
        self.last_update_time = update_time_total

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
        
        # 计算平均损失
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else None
        
        # 计算平均梯度（裁剪前后的总体L2范数）
        if critic_grads:
            # 收集每次迭代的裁剪前后值，计算平均值
            avg_critic_grad_before = sum(g['before'] for g in critic_grads) / len(critic_grads)
            avg_critic_grad_after = sum(g['after'] for g in critic_grads) / len(critic_grads)
            avg_critic_grad = {
                'before': avg_critic_grad_before,
                'after': avg_critic_grad_after
            }
        else:
            avg_critic_grad = None
        
        if actor_grads:
            # 收集每次迭代的裁剪前后值，计算平均值
            avg_actor_grad_before = sum(g['before'] for g in actor_grads) / len(actor_grads)
            avg_actor_grad_after = sum(g['after'] for g in actor_grads) / len(actor_grads)
            avg_actor_grad = {
                'before': avg_actor_grad_before,
                'after': avg_actor_grad_after
            }
        else:
            avg_actor_grad = None
        
        # 计算平均熵值
        avg_entropy = sum(entropies) / len(entropies) if entropies else None
        
        # 计算平均alpha梯度范数
        avg_alpha_grad = sum(alpha_grads) / len(alpha_grads) if alpha_grads else None
        
        return avg_critic_loss, critic_losses, avg_actor_loss, actor_losses, avg_critic_grad, avg_actor_grad, avg_entropy, avg_alpha_grad

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, add_noise):
        # SAC是随机策略，探索性由策略分布本身控制（通过目标熵值调整alpha）
        # 如果add_noise=True，使用策略分布采样；如果False，使用策略均值（确定性）
        # 注意：额外的动作噪声在SAC中通常是不必要的，因为策略本身已经是随机的
        if add_noise:
            # 使用策略分布采样，探索性由策略本身的方差控制
            return self.act(obs, sample=True)
        else:
            # 评估时使用确定性动作（策略均值）
            return self.act(obs, sample=False)

    def act(self, obs, sample=False):
        # 在推理时禁用梯度计算，提高性能
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            
            # 检查输入观察值是否包含NaN或Inf
            if not torch.isfinite(obs).all():
                # 如果输入包含NaN/Inf，使用零动作作为安全回退
                print(f"警告: act方法收到包含NaN/Inf的观察值，使用零动作作为回退。NaN count: {(~torch.isfinite(obs)).sum().item()}")
                obs = torch.zeros_like(obs)
            
            obs = obs.unsqueeze(0)
            
            try:
                dist = self.actor(obs)
                action = dist.sample() if sample else dist.mean
                
                # 注意：actor网络的输出（action）不检查NaN/Inf，只检查输入
                
                action = action.clamp(*self.action_range)
                assert action.ndim == 2 and action.shape[0] == 1
                return utils.to_np(action[0])
            except (ValueError, RuntimeError) as e:
                # 如果Actor网络出现问题，使用零动作作为安全回退
                print(f"警告: Actor网络出错 ({e})，使用零动作作为回退")
                action_shape = (1, self.action_dim)
                action = torch.zeros(action_shape, device=self.device)
                return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ((1 - done) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 全局L2范数裁剪（Global Norm Clipping）：将所有参数梯度拼接成一个向量，计算总体L2范数
        # 如果超过阈值，按比例缩放所有梯度（保持相对比例不变）
        if self.critic_grad_clip_value is not None:
            # clip_grad_norm_会执行裁剪，并返回裁剪前的总范数
            # 注意：clip_grad_norm_内部计算范数的方式可能与手动计算有细微差别（浮点数精度）
            # 因此使用clip_grad_norm_返回的值作为裁剪前的范数，这样更准确
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.critic_grad_clip_value,
                error_if_nonfinite=False  # 允许非有限值，避免抛出异常
            ).item()
            
            # clip_grad_norm_的行为：
            # - 如果总L2范数 > max_norm，则按比例缩放所有梯度，使得总L2范数 = max_norm
            # - 如果总L2范数 <= max_norm，则不进行任何操作
            # 因此，如果裁剪前范数 > 阈值，裁剪后应该等于阈值；否则应该等于裁剪前
            if grad_norm_before > self.critic_grad_clip_value:
                grad_norm_after = self.critic_grad_clip_value
            else:
                grad_norm_after = grad_norm_before
        else:
            # 如果没有启用裁剪，手动计算总体L2范数
            total_norm_squared = 0.0
            for param in self.critic.parameters():
                if param.grad is not None:
                    param_norm_squared = param.grad.data.norm(2) ** 2
                    total_norm_squared += param_norm_squared.item()
            grad_norm_before = total_norm_squared ** (1. / 2) if total_norm_squared > 0 else 0.0
            grad_norm_after = grad_norm_before
        
        # 返回裁剪前后的总体L2范数作为统计值
        critic_grad_norm = {
            'before': grad_norm_before,
            'after': grad_norm_after
        }
        
        self.critic_optimizer.step()
        
        return critic_loss.item(), critic_grad_norm

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 全局L2范数裁剪（Global Norm Clipping）：将所有参数梯度拼接成一个向量，计算总体L2范数
        # 如果超过阈值，按比例缩放所有梯度（保持相对比例不变）
        if self.actor_grad_clip_value is not None:
            # clip_grad_norm_会执行裁剪，并返回裁剪前的总范数
            # 注意：clip_grad_norm_内部计算范数的方式可能与手动计算有细微差别（浮点数精度）
            # 因此使用clip_grad_norm_返回的值作为裁剪前的范数，这样更准确
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.actor_grad_clip_value,
                error_if_nonfinite=False  # 允许非有限值，避免抛出异常
            ).item()
            
            # clip_grad_norm_的行为：
            # - 如果总L2范数 > max_norm，则按比例缩放所有梯度，使得总L2范数 = max_norm
            # - 如果总L2范数 <= max_norm，则不进行任何操作
            # 因此，如果裁剪前范数 > 阈值，裁剪后应该等于阈值；否则应该等于裁剪前
            if grad_norm_before > self.actor_grad_clip_value:
                grad_norm_after = self.actor_grad_clip_value
            else:
                grad_norm_after = grad_norm_before
        else:
            # 如果没有启用裁剪，手动计算总体L2范数
            total_norm_squared = 0.0
            for param in self.actor.parameters():
                if param.grad is not None:
                    param_norm_squared = param.grad.data.norm(2) ** 2
                    total_norm_squared += param_norm_squared.item()
            grad_norm_before = total_norm_squared ** (1. / 2) if total_norm_squared > 0 else 0.0
            grad_norm_after = grad_norm_before
        
        # 返回裁剪前后的总体L2范数作为统计值
        actor_grad_norm = {
            'before': grad_norm_before,
            'after': grad_norm_after
        }
        
        self.actor_optimizer.step()

        # 计算熵值（entropy = -log_prob的平均值）
        entropy = -log_prob.mean().item()
        
        # 计算alpha梯度的L2范数
        # 注意：log_alpha是一个标量参数（只有一个元素），所以其L2范数就是其绝对值的平方根
        # 对于单个标量，L2范数 = |grad|，这与actor/critic的全局L2范数计算方式一致（只是只有一个参数）
        alpha_grad_norm = None
        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            
            # 计算log_alpha参数的梯度L2范数
            # log_alpha是标量参数，所以直接计算其梯度的L2范数（对于标量，就是绝对值）
            if self.log_alpha.grad is not None:
                # 使用与actor/critic相同的方式计算L2范数（虽然只有一个参数，但保持一致性）
                alpha_grad_norm = self.log_alpha.grad.data.norm(2).item()
            else:
                alpha_grad_norm = 0.0
            
            self.log_alpha_optimizer.step()
        
        return actor_loss.item(), actor_grad_norm, entropy, alpha_grad_norm

    def update(self, replay_buffer, step, batch_size):
        t0 = time.time()
        batch_data = replay_buffer.sample_batch(batch_size)
        sample_dt = time.time() - t0
        if batch_data is None:
            print(f"警告: 缓冲区大小({replay_buffer.size()})小于批次大小({batch_size})，跳过训练")
            return None, None, None, None, sample_dt, time.time() - t0
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
        ) = batch_data

        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)

        critic_loss, critic_grad = self.update_critic(state, action, reward, next_state, done, step)

        actor_loss = None
        actor_grad = None
        entropy = None
        alpha_grad = None
        if step % self.actor_update_frequency == 0:
            actor_loss, actor_grad, entropy, alpha_grad = self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        
        total_dt = time.time() - t0
        return critic_loss, actor_loss, critic_grad, actor_grad, entropy, alpha_grad, sample_dt, total_dt

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        # update the returned data from ROS into a form used for learning in the current model
        latest_scan = np.array(latest_scan) #latest_scan为180度方向激光扫描点离智能体的距离

        # inf_mask = np.isinf(latest_scan) # 得到距离为无限的扫描点的下标
        # latest_scan[inf_mask] = self.scan_range # 将所有距离为无限的扫描点的距离设置为scan_range（最大有效距离）

        scan_len = len(latest_scan)
        
        max_bins = self.base_state_dim - 5 # 最大的分箱数（激光扫描被分成几个区域），使用base_state_dim而非state_dim
        bin_size = int(np.ceil(scan_len / max_bins)) # 计算每个分箱的扫描点数量

        # Initialize the list to store the minimum values of each bin
        min_obs_distance = []

        # Loop through the data and create bins
        for i in range(0, scan_len, bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, scan_len - i)]
            # Find the minimum value in the current bin and append it to the min_obs_distance list
            min_obs_distance.append(min(bin))
        state = min_obs_distance + [distance, cos, sin] + [action[0], action[1]]
        assert len(state) == self.base_state_dim, f"len(state) must be {self.base_state_dim}, but got {len(state)}"
        terminal = 1 if collision or goal else 0

        return state, terminal

    def state_dict(self, actor_only=False):
        """返回模型的状态字典
        
        Args:
            actor_only: 如果为True，只返回actor的权重（用于数据收集进程）
        """
        if actor_only or self.actor_only:
            # 只返回actor权重
            return {
                'actor': self.actor.state_dict()
            }
        else:
            # 返回完整模型权重
            return {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'log_alpha': self.log_alpha,
                'step': self.step
            }
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典
        
        支持加载完整模型或仅actor权重（当state_dict只包含actor键时）
        """
        # 加载actor权重（总是需要）
        self.actor.load_state_dict(state_dict['actor'])
        
        # 只有当state_dict包含其他组件时才加载（非actor_only模式）
        if not self.actor_only and 'critic' in state_dict and 'critic_target' in state_dict:
            self.critic.load_state_dict(state_dict['critic'])
            self.critic_target.load_state_dict(state_dict['critic_target'])
            if 'log_alpha' in state_dict:
                self.log_alpha = state_dict['log_alpha']
            if 'step' in state_dict:
                self.step = state_dict['step']


