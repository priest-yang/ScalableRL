from lwlab.distributed.proxy import RemoteEnv
env = RemoteEnv.make(address=('0.0.0.0', 50000), authkey=b'lightwheel')
env = env.unwrapped
env.reset()

import torch
for i in range(300):
    action = env.action_space.sample()
    action = torch.from_numpy(action).to(env.device)
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nTotal_steps:{i}, done:{done}, truncated:{truncated}, info_timeout: {info['log']['Episode_Termination/time_out_']}\n")
    if done or info['log']['Episode_Termination/time_out_']!=0.0:
        print(f"\n\nTruncated at step {i}\n\n")
        import pdb; pdb.set_trace()

env.reset()