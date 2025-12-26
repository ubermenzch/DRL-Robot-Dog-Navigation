from pathlib import Path
import time

import numpy as np
import torch
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
        hidden_dim=1024,
        hidden_depth=2,
        action_noise_std=0.2,  # 动作噪声标准差
        base_state_dim=None,  # 基础状态维度（单个时间步的状态向量长度），当使用历史state时使用
        actor_only=False,  # 是否只创建actor模型（用于数据收集进程，节省显存）
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
        
        # 创建actor模型（总是需要）
        self.actor = actor_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        # print(f"Actor model initialized")
        
        # 只有在非actor_only模式下才创建critic和target_critic
        if not self.actor_only:
            self.critic = critic_model(
                obs_dim=self.state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
            ).to(self.device)
            # print(f"Critic model initialized")

            self.critic_target = critic_model(
                obs_dim=self.state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
            ).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            # print(f"Critic target model initialized")

            if load_model:
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
        sample_time_total = 0.0
        update_time_total = 0.0
        
        for _ in range(iterations):
            critic_loss, actor_loss, sample_dt, total_dt = self.update(
                replay_buffer=replay_buffer, step=self.step, batch_size=batch_size
            )
            sample_time_total += sample_dt
            update_time_total += total_dt
            if critic_loss is not None:
                critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)

        self.step += 1

        # 记录最近一次训练的耗时统计，便于外部打印
        self.last_sample_time = sample_time_total
        self.last_update_time = update_time_total

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)
        
        # 计算平均损失
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else None
        
        return avg_critic_loss, critic_losses, avg_actor_loss, actor_losses

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
            obs = obs.unsqueeze(0)
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1
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
        self.critic_optimizer.step()
        
        return critic_loss.item()

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
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        
        return actor_loss.item()

    def update(self, replay_buffer, step, batch_size):
        t0 = time.time()
        batch_data = replay_buffer.sample_batch(batch_size)
        sample_dt = time.time() - t0
        if batch_data is None:
            print(f"警告: 缓冲区大小({replay_buffer.size()})小于批次大小({batch_size})，跳过训练")
            return None, sample_dt, time.time() - t0
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

        critic_loss = self.update_critic(state, action, reward, next_state, done, step)

        actor_loss = None
        if step % self.actor_update_frequency == 0:
            actor_loss = self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        
        total_dt = time.time() - t0
        return critic_loss, actor_loss, sample_dt, total_dt

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


