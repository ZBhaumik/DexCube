from orca_sim import OrcaHandRightCubeOrientation
import time

env = OrcaHandRightCubeOrientation(
    version="v2",
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
    render_mode="human",  # if supported
)

obs, info = env.reset(seed=0)

print("Observation:", obs)

for step in range(1000):
    action = env.action_space.sample()  # random action

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()  # if rendering is not automatic

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.02)  # slow things down so you can see it

env.close()