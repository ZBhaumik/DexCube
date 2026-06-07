from orca_sim import OrcaHandRightCubeOrientation
import torch

env = OrcaHandRightCubeOrientation(version="v2", render_mode="human")

nominal = env.nominal_reset_options()
obs, info = env.reset(options=nominal)

for step in range(1000):
    action = torch.zeros(env.action_space.shape, dtype=torch.float32).numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()