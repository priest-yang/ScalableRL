from lwlab.distributed.proxy import RemoteEnv
env = RemoteEnv.make(address=('0.0.0.0', 50000), authkey=b'lightwheel')
env = env.unwrapped
env.reset()

import torch
for i in range(100):
    action = env.action_space.sample()
    action = torch.from_numpy(action).to(env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n\nTotal_steps:{i}, done:{terminated}, truncated:{truncated}\n\n")

import pdb; pdb.set_trace()

env.reset()