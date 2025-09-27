import torch
from dataclasses import dataclass, field
from lwlab.distributed.proxy import RemoteEnv
import random
import argparse
import os
from tqdm import tqdm
from lwlab.utils.config_loader import config_loader
import json

from policy.maniskill_ppo.agent import PPOArgs, PPO, observation


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default="lerobot_liftobj_visual_hilserl_play", help="task config")


# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

args_cli.device = f"cuda:0"


def main(args):
    env = RemoteEnv.make(address=('0.0.0.0', 50000), authkey=b'lightwheel')
    env = env.unwrapped
    next_obs, _ = env.reset()
    next_obs = observation(next_obs['policy'])

    agent = PPO(env, next_obs, args, args_cli.device, train=False)
    if args_cli.checkpoint:
        agent.load_model(args_cli.checkpoint)
        

    eval_iter = 100
    success_count = 0
    episode_count = 0
    # env.env.cfg.execute_mode = ExecuteMode.EVAL
    with torch.inference_mode():
        for _ in tqdm(range(eval_iter), desc="Evaluation Progress"):
            action = agent.agent.get_action(next_obs, deterministic=True)
            # action = torch.zeros_like(action, device=env_cfg.sim.device)
            next_obs, _, terminations, truncations, infos = env.step(action)
            next_obs = observation(next_obs['policy'])

            if args_cli.check_success:
                success_count += infos['is_success'].sum().item()
                episode_count += (terminations | truncations).sum().item()
        if args_cli.check_success:
            success_rate = success_count / (episode_count + 1e-8)
            parent_dir = os.path.dirname(args_cli.checkpoint)
            result_path = os.path.join(parent_dir, "result.json")
            print(f"success_rate: {success_rate}")
            with open(result_path, "w") as f:
                json.dump({"success_rate": success_rate}, f)
    env.reset()



@dataclass
class Args:
    # env_id: str
    # """The environment id to train on"""
    # env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    args = Args()
    main(args.ppo)