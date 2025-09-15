



from pathlib import Path

from TD3.TD3 import TD3
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
import utils
from pretrain_utils import Pretraining
from datetime import datetime

max_velocity = 1.0 # 最大速度
neglect_angle = 0 # 前方视野左右两边忽略的角度（单位：度）
max_yawrate = 20.0 # 最大偏航率（单位：度/秒）
scan_range = 5
is_code_debug = False  # 是否为调试代码
is_eval = False  # whether to run evaluation after training
def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    if torch.cuda.is_available():
        print("GPU CUDA Ready")
    else:
        print("CPU CUDA Ready")
    # how many episodes to use to run evaluation 
    nr_eval_episodes = 10  # 用训练好的模型运行nr_eval_episodes个完整的回合（episodes）以评估模型性能。默认值为10
    # 训练整体思路是：一个轮次中，让智能体跑多个回合，这多个回合的数据在训练中被使用，训练时通过抽取回合数据来更新模型
    # max number of epochs. 
    max_epochs = 500  # 设置整个训练过程的最大迭代次数。默认值为100。
    # starting epoch number
    epoch = 0  # 当前训练轮次计数器
    # how many episodes to run in single epoch
    episodes_per_epoch = 70  # 在1轮（epoch）内收集episodes_per_epoch个回合的经验数据用于训练。默认值为70.
    # starting episode number
    episode = 0  # 当前轮次的回合数。默认值0.
    # train and update network parameters every n episodes
    train_every_n = 2  # 每train_every_n个回合后执行一次训练。默认值为2.
    # how many batches to use for single training cycle
    training_iterations = 200  # 每次执行训练时，用随机抽取的批次数据更新模型training_iterations次。默认值为500.
    # batch size for each training iteration
    batch_size = 40  # 从经验池中随机抽取batch_size条经验作为一个训练批次来对模型参数进行更新。默认值为40.
    # maximum number of steps in single episode
    max_steps = 3000  # 限制每个回合最多执行max_steps步操作，超过会强制结束回合。默认值为300.
    # starting step number
    steps = 0  # 当前回合的步数。默认值为0.
    # whether to load experiences from assets/data.yml
    load_saved_buffer = False  # 是否加载预存经验池。默认值为True。
    # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretrain = False  # 是否执行预训练。默认值为True，同时要求load_saved_buffer也必须为True。
    # number of training iterations to run during pre-training
    pretraining_iterations = (
        50  # 预训练迭代次数，默认值50
    )
    load_model = True
    
    # save the model every n training cycles
    save_every = 10  # 每save_every次训练后保存一次模型，默认值100

    load_save_path = Path("src/drl_navigation_ros2/models/SAC")

    if is_code_debug:
        max_epochs = 10
        episodes_per_epoch = 5
        train_every_n = 1
        training_iterations = 10
        load_saved_buffer = False
        pretrain = False

    if pretrain:
        load_saved_buffer=True
    else:
        load_saved_buffer=False

    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=load_model,
        save_directory = load_save_path,
        load_directory = load_save_path,
    )  # instantiate a model
    print("Model Loaded")
    ros = ROS_env(neglect_angle = neglect_angle,
    scan_range = scan_range,
    max_steps = max_steps,
    max_target_dist = 15.0,
    init_target_distance = 2.0,
    )  # instantiate ROS environment
    print("ROS Environment Initialized")
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=5e3, random_seed=42),
            reward_function=ros.get_reward,
        )  # instantiate pre-trainind
        print("Replay Buffer Loading")
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        print("Replay Buffer Loaded")
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training
        print("Load Saved Buffer Done")
    else:
        replay_buffer = ReplayBuffer(
            buffer_size=5e5, random_seed=42
        )  # if not experiences are loaded, instantiate an empty buffer
    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state
    #print("latest_scan_len:",len(latest_scan))
    print("="*20+f"Epoch 0"+"="*20)
    total_reward = 0.0
    episode_reward = 0.0
    while epoch < max_epochs:  # train until max_epochs is reached
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment
        action = model.get_action(state, True)  # get an action from the model
        a_in = utils.action_limit(
            action, max_velocity=max_velocity, max_yawrate=max_yawrate
        )
        #a_in = [1.0,0.0]
        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )  # get data from the environment
        episode_reward += reward
        #print("cos:", cos, "sin:", sin, "distance:", distance)
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            total_reward += episode_reward
            # target_distance
            td = np.linalg.norm([ros.target[0] - ros.episode_start_position[0], ros.target[1] - ros.episode_start_position[1]])
            if goal:
                episode_ending = "Goal"
            elif collision:
                episode_ending = "Collision"
            else:
                episode_ending= "Timeout"
            current_time = datetime.now()
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} Episode: {episode} Reward: {episode_reward:.2f} Target Distance: {td:.2f} Steps: {steps} End: {episode_ending}")
            episode_reward = 0.0

            episode += 1
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            if episode % train_every_n == 0:
                # print(
                #     f"Training model at epoch {epoch}, episode {episode}"
                # )
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters
            steps = 0
        else:
            steps += 1
        if (episode + 1) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            print(f"Epoch {epoch} Average Reward: {total_reward / episodes_per_epoch}")
            collision_rate = ros.collision_count / ros.env_count if ros.env_count > 0 else 0
            goal_rate = ros.goal_count / ros.env_count if ros.env_count > 0 else 0
            print(f"Goal rate: {goal_rate:.3f} Collision rate: {collision_rate:.3f} Timeout rate: {(1-goal_rate-collision_rate):.3f}")
            total_reward = 0.0
            epoch += 1
            print("="*20+f"Epoch {epoch}"+"="*20)
            if is_eval:
                eval(
                    model=model,
                    env=ros,
                    scenarios=eval_scenarios,
                    epoch=epoch,
                    max_steps=max_steps,
                )  # run evaluation


def eval(model, env, scenarios, epoch, max_steps):
    """Function to run evaluation"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
    avg_reward = 0.0
    col = 0
    gl = 0
    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            action = model.get_action(state, False)
            a_in = utils.action_limit(
                action, max_velocity=max_velocity, max_yawrate=max_yawrate
            )
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            col += collision
            gl += goal
    avg_reward /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
