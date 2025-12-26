from typing import List
from tqdm import tqdm
import yaml


class Pretraining:
    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function

    def load_buffer(self):
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                print("It will take minutes.Please be patient...")
                samples = yaml.full_load(file)
                print("full_loaded")
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    latest_scan = sample["latest_scan"]
                    distance = sample["distance"]
                    cos = sample["cos"]
                    sin = sample["sin"]
                    collision = sample["collision"]
                    goal = sample["goal"]
                    action = sample["action"]

                    state, terminal = self.model.prepare_state(
                        latest_scan, distance, cos, sin, collision, goal, action
                    )

                    if terminal:
                        continue

                    next_sample = samples[i + 1]
                    next_latest_scan = next_sample["latest_scan"]
                    next_distance = next_sample["distance"]
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_collision = next_sample["collision"]
                    next_goal = next_sample["goal"]
                    next_action = next_sample["action"]
                    next_state, next_terminal = self.model.prepare_state(
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                        next_collision,
                        next_goal,
                        next_action,
                    )
                    # 与ROS环境一致的奖励函数签名：
                    # get_reward(goal, collision, action, latest_scan, distance, cos, sin)
                    reward = self.reward_function(
                        next_goal,
                        next_collision,
                        action,
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                    )
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")
