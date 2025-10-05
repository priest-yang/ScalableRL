# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import torch
import argparse
import tqdm

from lwlab.distributed.proxy import RemoteEnv
from lwlab.utils.config_loader import config_loader

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

from lerobot.lwrl.sim.lwlab.env_lwlab import (
    make_lwlab_robot_env, 
    make_lwlab_processors,
    step_lwlab_env_and_process_transition,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LeRobot Dataset Collection Script")
    
    # Eval configuration
    parser.add_argument("--n_steps", type=int, default=100,
                       help="Number of steps to evaluate")

    # model configuration
    parser.add_argument("--model_path", type=str, 
                       default="/home/johndoe/Documents/lerobot-hilserl/outputs/train/2025-10-05/00-57-03_lwlab_lerobot_pickup_100env_nsteps3/checkpoints/0090000/pretrained_model",
                       help="Model path")

    # hilserl args (for obs processing)
    #! need to type in the command line explicitly (from lerobot-hilserl)
    parser.add_argument("--config_path", type=str, help="Hilserl arguments", required=True)
    
    return parser.parse_args()

def eval_policy(env, env_processor, action_processor, policy, n_steps, cfg):
    sum_success = 0
    sum_total = 0

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    # Process initial observation
    transition = create_transition(
        observation=obs,
        reward=torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device),
        done=torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
        truncated=torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
        info=info)
    transition = env_processor(transition)

    for _ in tqdm.tqdm(range(n_steps)):
        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }
        action = policy.select_action(batch=observation)
        new_transition = step_lwlab_env_and_process_transition(
            env=env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        done = new_transition.get(TransitionKey.DONE, torch.tensor(False))
        truncated = new_transition.get(TransitionKey.TRUNCATED, torch.tensor(False))
        info = new_transition.get(TransitionKey.INFO, {})

        if info == {}:
            print("Warning: info is empty")
        num_success = info.get('is_success', torch.zeros_like(done, device=env.device, dtype=torch.bool)).sum().item()
        num_total = torch.logical_or(done, truncated).sum().item()

        sum_success += num_success
        sum_total += num_total

        transition = new_transition

    # print success rate
    print(f"Success rate: {sum_success / sum_total}")
    print(f"Total steps: {sum_total}")
    print(f"Success steps: {sum_success}")

        
@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    args = parse_arguments()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Making environments...")
    env, teleop_device = make_lwlab_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_lwlab_processors(env, teleop_device, cfg.env, cfg.policy.device)

    print("Making policy...")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    print("Loading policy...")
    policy.from_pretrained(args.model_path, local_files_only=True)
    policy.eval()

    print("Evaluating policy...")
    eval_policy(env, env_processor, action_processor, policy=policy, n_steps=args.n_steps, cfg=cfg)


if __name__ == "__main__":
    main()
