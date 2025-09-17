
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取episodes.jsonl文件
episode_lengths = []
with open('/home/johndoe/Documents/lerobot_data/aractingi/franka_sim_pick_lift_6/meta/episodes.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        episode_lengths.append(data['length'])

# 计算统计信息
mean_length = np.mean(episode_lengths)
std_length = np.std(episode_lengths)
min_length = np.min(episode_lengths)
max_length = np.max(episode_lengths)

print(f"Episode Length Statistics:")
print(f"  Mean length: {mean_length:.2f}")
print(f"  Standard deviation: {std_length:.2f}")
print(f"  Min length: {min_length}")
print(f"  Max length: {max_length}")
print(f"  Total episodes: {len(episode_lengths)}")

# Create chart
plt.figure(figsize=(12, 8))

# Draw histogram
plt.hist(episode_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)

# Add mean line
plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.2f}')

# Add standard deviation lines
plt.axvline(mean_length + std_length, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+1σ: {mean_length + std_length:.2f}')
plt.axvline(mean_length - std_length, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'-1σ: {mean_length - std_length:.2f}')

# Set chart properties
plt.xlabel('Episode Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Episode Length Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add statistics text box
stats_text = f'Total episodes: {len(episode_lengths)}\nMean: {mean_length:.2f}\nStd: {std_length:.2f}\nRange: {min_length}-{max_length}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()