from torchvision.transforms.v2.functional import elastic
import gymnasium as gym
import numpy as np
from typing import Any, Sequence
from lerobot.envs.utils import preprocess_observation
import torch
import torch.nn.functional as F
import einops

class LwLabObservationProcessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, ENV_STATE_KEYS=None, OBS_STATE_KEYS=None, CAMERA_KEYS=None, features=None):
        super().__init__(env)
        prev_space = self.observation_space
        new_space = {}

        num_envs = env.num_envs
        policy_observation_space = prev_space["policy"]

        # environment_state
        # all available keys except cameras
        self.ENV_STATE_KEYS = ENV_STATE_KEYS if ENV_STATE_KEYS is not None else [key for key in policy_observation_space if key not in ["camera"]]
        self.ENV_STATE_SHAPE = [num_envs, 0]
        # observation_state
        self.OBS_STATE_KEYS = OBS_STATE_KEYS if OBS_STATE_KEYS is not None else ["joint_pos"]
        self.OBS_STATE_SHAPE = [num_envs, 0]
        # camera_keys
        self.CAMERA_KEYS = CAMERA_KEYS
        
        for key in policy_observation_space:
            if "camera" in key:
                new_space[f"observation.images.{key}"] = gym.spaces.Box(
                    0.0, 255.0, shape=policy_observation_space[key].shape, dtype=np.uint8
                )

            elif len(policy_observation_space[key].shape) == 2: # num_envs * dim, otherwise need to handle differently
                if key in self.ENV_STATE_KEYS:
                    self.ENV_STATE_SHAPE[1] += policy_observation_space[key].shape[0]
                if key in self.OBS_STATE_KEYS:
                    self.OBS_STATE_SHAPE[1] += policy_observation_space[key].shape[0]

        new_space["observation.environment_state"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.ENV_STATE_SHAPE,
            dtype=np.float32,
        )
        new_space["observation.state"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.OBS_STATE_SHAPE,
            dtype=np.float32,
        )
        self.features = features
        self.observation_space = gym.spaces.Dict(new_space)

    def observation(self, observations: dict[str, Any]) -> dict[str, Any]:
        # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
        """Convert environment observation to LeRobot format observation.
        Args:
            observation: Dictionary of observation batches from a Gym vector environment.
        Returns:
            Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
        """
        # map to expected inputs for the policy
        return_observations = {}

        if self.CAMERA_KEYS is not None:
            imgs = {
                f"observation.images.{key}": img for key, img in observations["policy"].items() if key in self.CAMERA_KEYS
            }
        else:
            imgs = {
                f"observation.images.{key}": img for key, img in observations["policy"].items() if "camera" in key
            }

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            # img = torch.from_numpy(img) already done in the environment

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            if imgkey in self.features:
                # resize to match the target feature shape (width and height only)
                target_shape = self.features[imgkey].shape
                batch_size, c, h, w = img.shape
                
                # Extract target height and width (assuming shape is [C, H, W] or [H, W, C])
                if len(target_shape) >= 2:
                    if len(target_shape) == 3:  # [C, H, W] format
                        target_h, target_w = target_shape[1], target_shape[2]
                    else:  # [H, W] format
                        target_h, target_w = target_shape[0], target_shape[1]
                    
                    # Only resize if dimensions are different
                    if h != target_h or w != target_w:
                        img = F.interpolate(
                            img, 
                            size=(target_h, target_w), 
                            mode='bilinear', 
                            align_corners=False
                        )
                else:
                    # Fallback to reshape for non-image features
                    img = img.reshape(batch_size, *target_shape)

            return_observations[imgkey] = img

        # handle environment_state
        env_state = torch.concat([observations["policy"][key] for key in self.ENV_STATE_KEYS], dim=1)
        return_observations["observation.environment_state"] = env_state
        # handle observation_state
        obs_state = torch.concat([observations["policy"][key] for key in self.OBS_STATE_KEYS], dim=1)
        return_observations["observation.state"] = obs_state

        return return_observations


class TorchBox(gym.spaces.Box):
    """
    A version of gym.spaces.Box that handles PyTorch tensors.

    This class extends gym.spaces.Box to work with PyTorch tensors,
    providing compatibility between NumPy arrays and PyTorch tensors.
    """

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: Sequence[int] | None = None,
        np_dtype: np.dtype | type = np.float32,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the PyTorch-compatible Box space.

        Args:
            low: Lower bounds of the space.
            high: Upper bounds of the space.
            shape: Shape of the space. If None, inferred from low and high.
            np_dtype: NumPy data type for internal storage.
            torch_dtype: PyTorch data type for tensor conversion.
            device: PyTorch device for returned tensors.
            seed: Random seed for sampling.
        """
        super().__init__(low, high, shape=shape, dtype=np_dtype, seed=seed)
        self.torch_dtype = torch_dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        Sample a random point from the space.

        Returns:
            A PyTorch tensor within the space bounds.
        """
        arr = super().sample()
        return torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)

    def contains(self, x: torch.Tensor) -> bool:
        """
        Check if a tensor is within the space bounds.

        Args:
            x: The PyTorch tensor to check.

        Returns:
            Boolean indicating whether the tensor is within bounds.
        """
        # Move to CPU/numpy and cast to the internal dtype
        arr = x.detach().cpu().numpy().astype(self.dtype, copy=False)
        return super().contains(arr)

    def seed(self, seed: int | np.random.Generator | None = None):
        """
        Set the random seed for sampling.

        Args:
            seed: The random seed to use.

        Returns:
            List containing the seed.
        """
        super().seed(seed)
        return [seed]

    def __repr__(self) -> str:
        """
        Return a string representation of the space.

        Returns:
            Formatted string with space details.
        """
        return (
            f"TorchBox({self.low_repr}, {self.high_repr}, {self.shape}, "
            f"np={self.dtype.name}, torch={self.torch_dtype}, device={self.device})"
        )


class LwLabTorchActionWrapper(gym.Wrapper):
    """
    Wrapper that changes the action space to use PyTorch tensors.

    This wrapper modifies the action space to return PyTorch tensors when sampled
    and handles converting PyTorch actions to NumPy when stepping the environment.
    """

    def __init__(self, env: gym.Env, device: str):
        """
        Initialize the PyTorch action space wrapper.

        Args:
            env: The environment to wrap.
            device: The PyTorch device to use for tensor operations.
        """
        super().__init__(env)
        self.action_space = TorchBox(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.device = env.device if device is None else device

    def step(self, action: torch.Tensor):
        """
        Step the environment with a PyTorch tensor action.

        This method handles conversion from PyTorch tensors to NumPy arrays
        for compatibility with the underlying environment.

        Args:
            action: PyTorch tensor action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device)
        return self.env.step(action)

class LwlabSparseRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

    def step(self, action: torch.Tensor):
        """
        Override the Reward function to 0-1 sparse reward
        """
        obs, reward, done, truncated, info = self.env.step(action)
        reward = info['log']['Episode_Termination/success']
        truncated = truncated or done

        return obs, reward, done, truncated, info