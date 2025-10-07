import torch
import argparse
import tqdm
import random
import os
import pickle as pkl
import numpy as np

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

# ========== Tunables ==========
Q_MIN = 0.0
Q_MAX = 1.25
BATCH_SIZE = 512

# Camera/view defaults for the 3D plot
ELEV = 28
AZIM = -158
ROLL = 19
FIGSIZE = (8, 6)
FPS = 1  # default if you don't pass another
# =============================

def parse_arguments():
    """Parse command line arguments"""
    import datetime
    _parser = argparse.ArgumentParser(description="LeRobot Q-heat Trajectory Visualization",
        add_help=False)

    # dataset configuration
    _parser.add_argument("--dataset_root", type=str,
                        default="/home/johndoe/Documents/lerobot-hilserl/datasets/lerobot_lift_viz",
                        help="Dataset root directory")

    # visualization configuration
    _parser.add_argument("--n_trajectories", type=int, default=50,
                        help="Number of trajectories to evaluate")

    _parser.add_argument("--downsample_rate", type=int, default=2,
                        help="Downsample rate")

    _parser.add_argument("--debug", action="store_true",
                        help="Debug mode (show an example plot interactively to adjust view)")

    _parser.add_argument("--fps", type=int, default=FPS, help="FPS for the output video")

    # model configuration
    _parser.add_argument("--model_dir", type=str,
                        default="/home/johndoe/Documents/lerobot-hilserl/outputs/train/2025-10-06/02-16-15_lwlab_lerobot_pickup_100env_nsteps3_130episodes/actor/checkpoints/",
                        help="Directory that contains step subfolders with .pkl checkpoints")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    _parser.add_argument("--output_dir", type=str,
                        default=f"./outputs/visualization/{current_time}",
                        help="Output directory")

    # hilserl args (for obs processing)
    #! need to type in the command line explicitly (from lerobot-hilserl)
    # _parser.add_argument("--config_path", type=str, help="Hilserl arguments", required=True)

    # Optional: override Q min/max via CLI if you want
    _parser.add_argument("--q_min", type=float, default=Q_MIN)
    _parser.add_argument("--q_max", type=float, default=Q_MAX)

    return _parser.parse_known_args()


def generate_trajectories(dataset_root, n_trajectories=50, downsample_rate=5):
    dataset = LeRobotDataset(root=dataset_root, repo_id="lerobot_lift_viz")
    trajs = []
    traj = []
    episode_index = dataset[0]["episode_index"]

    for frame in tqdm.tqdm(dataset, desc="Generating trajectories"):
        if frame["episode_index"] != episode_index:
            if len(traj) > 0:
                trajs.append(traj[::downsample_rate])
            traj = []
            episode_index = frame["episode_index"]
        traj.append(frame)
    if len(traj) > 0:
        trajs.append(traj)
    if len(trajs) > n_trajectories:
        trajs = random.sample(trajs, n_trajectories)
    return trajs

def stack_batch(batch_list, device="cpu"):
    observations = {
        key: torch.stack([frame[key] for frame in batch_list]).to(device)
        for key in batch_list[0].keys() if "observation" in key
    }
    actions = torch.stack([frame["action"] for frame in batch_list]).to(device)
    return observations, actions

@torch.no_grad()
def inference_trajectories(critic, trajs):
    viz_trajs = []
    device = next(iter(critic.parameters())).device
    critic.eval()
    # run batched inference
    for traj in tqdm.tqdm(trajs, desc="Inferring Q on trajectories"):
        viz_traj = []
        for i in range(0, len(traj), BATCH_SIZE):
            batch_list = traj[i:i+BATCH_SIZE]
            observations, actions = stack_batch(batch_list, device=device)
            q_values = critic(observations, actions).detach().cpu()
            # Shape normalization: [ensemble, B] or [B, heads] -> mean per item
            if q_values.ndim > 1:
                q_values = q_values.mean(dim=0)
            # ensure plain tensor shape [B]
            q_values = q_values.view(-1)

            for frame, q_value in zip(batch_list, q_values):
                # eef_pos expects xyz
                pos = frame["complementary_info.ee_pose"][:3]
                if torch.is_tensor(pos):
                    pos = pos.detach().cpu().numpy()
                viz_traj.append({
                    "eef_pos": np.asarray(pos, dtype=np.float32),  # (3,)
                    "q_value": float(q_value.item()),
                })
        viz_trajs.append(viz_traj)
    return viz_trajs

def _set_axes_equal_3d(ax, xyz):
    """Set 3D axes to equal aspect based on data points Nx3."""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    spans = (maxs - mins)
    radius = max(spans) / 2.0
    x, y, z = centers
    ax.set_xlim(x - radius, x + radius)
    ax.set_ylim(y - radius, y + radius)
    ax.set_zlim(z - radius, z + radius)

def fig_to_rgb(fig):
    """
    Return (H, W, 3) uint8 RGB for any MPL backend.
    """
    import numpy as np
    # 1) Try current canvas (works on Agg/Qt5Agg, etc.)
    try:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3).copy()
    except Exception:
        # 2) Fallback: render with Agg offscreen
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        # drop alpha if present
        if buf.shape[-1] == 4:
            buf = buf[..., :3]
        return buf.copy()

def plot_3d_trajectory(trajs, q_min=Q_MIN, q_max=Q_MAX, elev=ELEV, azim=AZIM, roll=ROLL, figsize=FIGSIZE, debug=False):
    """
    Plot a single model's set of trajectories with heat coloring by Q.
    `trajs` is a list of trajectories; each trajectory is a list of dicts with keys:
       - "eef_pos": (3,) array-like
       - "q_value": float
    Returns: RGB numpy array for this frame (H, W, 3), dtype=uint8
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib import cm, colors

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # collect all points to set equal aspect
    all_pts = []
    for traj in trajs:
        for p in traj:
            all_pts.append(p["eef_pos"])
    if len(all_pts) == 0:
        # empty safety
        all_pts = np.zeros((2, 3), dtype=np.float32)
    else:
        all_pts = np.stack(all_pts, axis=0)
    _set_axes_equal_3d(ax, all_pts)

    # Normalizer for Q
    norm = colors.Normalize(vmin=q_min, vmax=q_max, clip=True)
    cmap = matplotlib.colormaps["viridis"]

    # draw each trajectory as small colored segments
    for traj in trajs:
        if len(traj) < 2:
            continue
        pts = np.stack([t["eef_pos"] for t in traj], axis=0)  # (T, 3)
        qs = np.array([t["q_value"] for t in traj], dtype=np.float32)  # (T,)

        # segments between consecutive points
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)  # (T-1, 2, 3)

        # segment color uses the Q at the *end* point (or mean of ends)
        q_seg = (qs[:-1] + qs[1:]) * 0.5  # (T-1,)
        colors_rgba = cmap(norm(q_seg))  # (T-1, 4)

        lc = Line3DCollection(segs, colors=colors_rgba, linewidths=1.5, alpha=0.9)
        ax.add_collection3d(lc)

        # Optionally scale alpha/linewidth by Q as in your JAX ref:
        # alpha = 0.2 + 0.6 * norm(q_seg)
        # lc.set_alpha(alpha)

        # mark start/end (small)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], marker='o', s=6, c='white', edgecolors='k', linewidths=0.3)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], marker='^', s=10, c='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)  # roll needs newer Matplotlib; if not, drop arg
    except TypeError:
        ax.view_init(elev=elev, azim=azim)

    # colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Q-value')

    fig.tight_layout()

    if debug:
        import matplotlib.pyplot as plt
        plt.show()

    # Convert canvas to RGB array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = fig_to_rgb(fig)
    img = img.reshape(h, w, 3).copy()
    # clear the figure
    fig.clear()
    plt.close(fig)
    return img

def generate_video(episodes, output_dir, q_min=Q_MIN, q_max=Q_MAX, fps=FPS, debug=False):
    """
    Build a video over *models* (episodes is list over models/checkpoints),
    where each element is a list of trajectories (each trajectory is list of {eef_pos, q_value} dicts).
    """
    import imageio.v2 as imageio

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    images = []
    for i, trajs in enumerate(tqdm.tqdm(episodes, desc="Rendering frames")):
        img = plot_3d_trajectory(
            trajs,
            q_min=q_min,
            q_max=q_max,
            elev=ELEV,
            azim=AZIM,
            roll=ROLL,
            figsize=FIGSIZE,
            debug=(debug and i == 0),
        )
        frame_path = os.path.join(frames_dir, f"{i:05d}.png")
        imageio.imwrite(frame_path, img)
        images.append(img)

    # write video
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "video.mp4")
    imageio.mimsave(video_path, images, fps=fps)
    print(f"[OK] Saved video to: {video_path}")


def main(cfg: TrainRLServerPipelineConfig, args):
    cfg.validate()

    # Allow overriding Q range at runtime
    q_min = args.q_min
    q_max = args.q_max

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # generate trajectories (positions+actions)
    trajs = generate_trajectories(
        dataset_root=args.dataset_root,
        n_trajectories=args.n_trajectories,
        downsample_rate=args.downsample_rate
    )

    print("Making policy...")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    # collect all model checkpoint files (*.pkl) inside step subfolders
    all_model_paths = []
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    for steps in sorted(os.listdir(args.model_dir)):
        step_dir = os.path.join(args.model_dir, steps)
        if not os.path.isdir(step_dir):
            continue
    
        model_path = os.path.join(step_dir, "pretrained_model")
        all_model_paths.append(model_path)

    all_model_paths = sorted(all_model_paths)
    print(f"Found {len(all_model_paths)} models")

    os.makedirs(args.output_dir, exist_ok=True)

    all_viz_trajs = []
    for model_path in tqdm.tqdm(all_model_paths, desc="Loading checkpoints + scoring"):
        policy = policy.from_pretrained(model_path)
        policy = policy.eval().to(device)
        critic = policy.critic_ensemble
        viz_trajs = inference_trajectories(critic, trajs)
        all_viz_trajs.append(viz_trajs)

    # optional: quick debug view for first frame to tune camera
    if args.debug and len(all_viz_trajs) > 0:
        _ = plot_3d_trajectory(all_viz_trajs[-1], q_min=q_min, q_max=q_max, debug=True)

    generate_video(all_viz_trajs, args.output_dir, q_min=q_min, q_max=q_max, fps=args.fps, debug=args.debug)

if __name__ == "__main__":

    import sys
    viz_args, remaining = parse_arguments()

    # temporarily hand the remaining args to the LeRobot parser-decorated entrypoint
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]] + remaining

    @parser.wrap()
    def _entry(cfg: TrainRLServerPipelineConfig):
        cfg.validate()
        main(cfg, viz_args)

    _entry()


"""
sample usage:
python visualization.py \
    --model_dir /home/johndoe/Documents/lerobot-hilserl/outputs/train/2025-10-06/02-16-15_lwlab_lerobot_pickup_100env_nsteps3_130episodes/actor/checkpoints/ \
    --dataset_root /home/johndoe/Documents/lerobot-hilserl/datasets/lerobot_lift_viz \
    --output_dir /home/johndoe/Documents/lerobot-hilserl/outputs/ \
    --n_trajectories 50 \
    --downsample_rate 2 \
    --fps 1 \
    --debug
"""