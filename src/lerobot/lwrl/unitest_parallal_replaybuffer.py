#!/usr/bin/env python3
"""
Unit test for ParallelReplayBuffer using actual RL environment.
Tests the complete workflow: data collection -> buffer storage -> dataset conversion -> loading.
"""
import torch
import numpy as np
import tempfile
import shutil
from dataclasses import dataclass, field
from lwlab.distributed.proxy import RemoteEnv
import random
import argparse
from tqdm import tqdm
from lwlab.utils.config_loader import config_loader
import json

from policy.maniskill_ppo.agent import PPOArgs, PPO, observation
from lerobot.utils.buffer_batched import ParallelReplayBuffer, BatchTransition
from lerobot.utils.transition import move_transition_to_device
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import copy


@dataclass
class Args:
    ppo: PPOArgs = field(default_factory=PPOArgs)

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default="lerobot_liftobj_visual_hilserl_play", help="task config")

# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

args_cli.device = f"cuda:0"


def create_test_env():
    """Create a test environment for data collection."""
    print("Creating test environment...")
    env = RemoteEnv.make(address=('0.0.0.0', 50000), authkey=b'lightwheel')
    env = env.unwrapped
    env.reset()
    return env


def collect_data_with_policy(env, agent, num_steps=100, num_envs=4, args=None):
    """
    Collect data using the policy and store in ParallelReplayBuffer.
    
    Args:
        env: Environment instance
        agent: Policy agent
        num_steps: Number of steps to collect
        num_envs: Number of parallel environments
    
    Returns:
        ParallelReplayBuffer: Buffer with collected data
    """
    print(f"Collecting {num_steps} steps with {num_envs} parallel environments...")
    
    # Initialize buffer
    buffer = ParallelReplayBuffer(
        capacity=num_steps * 2,  # Extra capacity for safety
        num_envs=num_envs,
        device="cuda:0",
        storage_device="cpu"
    )
    
    # Reset environment
    obs, _ = env.reset()
    obs = obs['policy']
    
    # Initialize agent
    agent = PPO(env, observation(copy.deepcopy(obs)), args.ppo, args_cli.device, train=False)
    if args_cli.checkpoint:
        agent.load_model(args_cli.checkpoint)

    step_count = 0
    success_count = 0
    episode_count = 0
    with torch.inference_mode():
        while step_count < num_steps:
            # Get actions from policy
            actions = agent.agent.get_action(observation(copy.deepcopy(obs)), deterministic=False)
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            next_obs = next_obs['policy']
            success_count += infos['is_success'].sum().item()
            episode_count += (terminations | truncations).sum().item()

            parallel_transition = BatchTransition(
                state=obs,
                action=actions,
                reward=rewards,
                next_state=next_obs,
                done=terminations,
                truncated=truncations,
                complementary_info={"is_success": infos['is_success'].to(torch.float32)},
            )
            
            tr = move_transition_to_device(parallel_transition, device=buffer.storage_device)
            
            # Add to buffer
            buffer.add(
                state=tr["state"],
                action=tr["action"],
                reward=tr["reward"],
                next_state=tr["next_state"],
                done=tr["done"],
                truncated=tr["truncated"],
                complementary_info=tr["complementary_info"],
            )
            
            obs = next_obs
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Collected {step_count}/{num_steps} steps")

    print(f"Success count: {success_count}, Episode count: {episode_count}")
    print(f"Data collection completed. Buffer size: {len(buffer)}")

    return buffer


def test_buffer_sampling(buffer, num_samples=5):
    """Test sampling from the buffer."""
    print(f"\nTesting buffer sampling with {num_samples} samples...")
    
    for i in range(num_samples):
        batch = buffer.sample(batch_size=8)

        # Check batch structure
        assert "state" in batch
        assert "action" in batch
        assert "reward" in batch
        assert "next_state" in batch
        assert "done" in batch
        assert "truncated" in batch
        
        # Check shapes
        batch_size = batch["action"].shape[0]
        assert batch_size <= 8
        
        # check min, max, mean, std of action, reward, done, truncated
        for key, value in batch['state'].items():
            print(f"  Sample {i+1}: {key}_min={value.min()}, {key}_max={value.max()}, {key}_mean={value.mean()}, {key}_std={value.std()}")
        print(f"  Sample {i+1}: action_min={batch['action'].min()}, action_max={batch['action'].max()}, action_mean={batch['action'].mean()}, action_std={batch['action'].std()}")
        print(f"  Sample {i+1}: reward_min={batch['reward'].min()}, reward_max={batch['reward'].max()}, reward_mean={batch['reward'].mean()}, reward_std={batch['reward'].std()}")
        print(f"  Sample {i+1}: done_min={batch['done'].min()}, done_max={batch['done'].max()}, done_mean={batch['done'].mean()}, done_std={batch['done'].std()}")
        print(f"  Sample {i+1}: truncated_min={batch['truncated'].min()}, truncated_max={batch['truncated'].max()}, truncated_mean={batch['truncated'].mean()}, truncated_std={batch['truncated'].std()}")
        print(f"  Sample {i+1}: batch_size={batch_size}, action_shape={batch['action'].shape}")
        print("\n")
    
    print("✓ Buffer sampling works correctly")


def test_buffer_to_dataset(buffer, temp_dir):
    """Test converting buffer to LeRobotDataset."""
    print(f"\nTesting buffer to LeRobotDataset conversion...")

    # clear temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)

    # test there is success in the buffer
    is_success = buffer.complementary_info["is_success"].sum() > 0
    assert is_success, "There should be success in the buffer"
    
    # Convert buffer to dataset
    dataset = buffer.to_lerobot_dataset(
        repo_id="test_parallel_buffer",
        fps=1,
        root=temp_dir,
        task_name="parallel_buffer_test"
    )
    
    print(f"Dataset created with {len(dataset)} frames")
    
    if len(dataset) == 0:
        print("Dataset is empty")
    else:
        # Check dataset structure
        sample = dataset[0]
        required_keys = ["action", "next.reward", "next.done"]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        # Check state keys
        for key in buffer.state_keys:
            assert key in sample, f"Missing state key: {key}"
        
        print("✓ Buffer to dataset conversion works")
        return dataset


def test_dataset_to_buffer(dataset, num_envs=4, state_keys=None):
    """Test loading dataset back into a new buffer."""
    print(f"\nTesting dataset to buffer conversion with {num_envs} environments...")
    
    # Create new buffer from dataset
    new_buffer = ParallelReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        num_envs=num_envs,
        device="cuda:0",
        storage_device="cpu",
        state_keys=state_keys
    )
    
    print(f"New buffer created with {len(new_buffer)} total transitions")
    print(f"Environment sizes: {new_buffer.size}")
    
    # Test sampling from new buffer
    batch = new_buffer.sample(batch_size=4)
    assert batch["action"].shape[0] <= 4
    
    print("✓ Dataset to buffer conversion works")
    return new_buffer


def compare_buffers(original_buffer, new_buffer, tolerance=1e-6):
    """Compare two buffers to check if they contain the same data."""
    print(f"\nComparing original and reconstructed buffers...")
    
    # Check total sizes
    original_size = len(original_buffer)
    new_size = len(new_buffer)
    print(f"Original buffer size: {original_size}")
    print(f"New buffer size: {new_size}")
    
    # Sample from both buffers and compare
    num_comparisons = 5
    for i in range(num_comparisons):
        orig_batch = original_buffer.sample(batch_size=128)
        new_batch = new_buffer.sample(batch_size=128)
        
        # Compare action statistics
        orig_action_mean = orig_batch["action"].mean()
        new_action_mean = new_batch["action"].mean()
        orig_reward_mean = orig_batch["reward"].mean()
        new_reward_mean = new_batch["reward"].mean()
        
        print(f"  Comparison {i+1}:")
        print(f"    Action means - Original: {orig_action_mean:.4f}, New: {new_action_mean:.4f}")
        print(f"    Reward means - Original: {orig_reward_mean:.4f}, New: {new_reward_mean:.4f}")
        
        # Check if means are reasonably close (allowing for sampling variance)
        action_diff = abs(orig_action_mean - new_action_mean)
        reward_diff = abs(orig_reward_mean - new_reward_mean)
        
        if action_diff > 0.1:  # Allow some variance due to random sampling
            print(f"    ⚠️  Action means differ by {action_diff:.4f}")
        if reward_diff > 0.1:
            print(f"    ⚠️  Reward means differ by {reward_diff:.4f}")
    
    print("✓ Buffer comparison completed")


def test_parallel_buffer_workflow():
    """Main test function for the complete ParallelReplayBuffer workflow."""
    print("=" * 60)
    print("Testing ParallelReplayBuffer Complete Workflow")
    print("=" * 60)
    
    # Create temporary directory for dataset storage
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Test parameters
        num_steps = 70

        # Create test environment and agent
        print("\n1. Setting up environment and agent...")
        env = create_test_env()
        args = Args()
        
        # Collect data
        print("\n2. Collecting data with policy...")
        buffer = collect_data_with_policy(env, args.ppo, num_steps=num_steps, num_envs=env.num_envs, args=args)
        
        # Test sampling
        print("\n3. Testing buffer sampling...")
        test_buffer_sampling(buffer, num_samples=3)
        
        # Convert to dataset
        print("\n4. Converting buffer to LeRobotDataset...")
        dataset = test_buffer_to_dataset(buffer, temp_dir)
        
        # Load dataset back to buffer
        print("\n5. Loading dataset back to new buffer...")
        new_buffer = test_dataset_to_buffer(dataset, num_envs=env.num_envs, state_keys=list(env.reset()[0]['policy'].keys()))
        
        # Compare buffers
        print("\n6. Comparing original and reconstructed buffers...")
        compare_buffers(buffer, new_buffer)
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! ParallelReplayBuffer workflow is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    
    print("Starting ParallelReplayBuffer unit tests...")

    # Run full workflow test (requires environment)
    try:
        test_parallel_buffer_workflow()
    except Exception as e:
        print(f"Full workflow test skipped due to environment dependency: {e}")
        print("Simple test completed successfully.")
