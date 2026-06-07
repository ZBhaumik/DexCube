from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from orca_sim import OrcaHandRightCubeOrientation
import numpy as np
from collections import deque

def make_env():
    return Monitor(OrcaHandRightCubeOrientation(version="v2", render_mode="human"))

env = DummyVecEnv([make_env])

vecnorm_path = "./checkpoints/vecnormalize_250000_steps.pkl"
model_path = "./checkpoints/sac_cube_orientation_250000_steps.zip"

env = VecNormalize.load(vecnorm_path, env)
env.training = False
env.norm_reward = False
model = SAC.load(model_path, env=env)

num_episodes = 50

successes = deque()
drops = deque()
timeouts = deque()
alignments = deque()

obs = env.reset()

episode_count = 0
import time
while episode_count < num_episodes:
    time.sleep(0.03)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    for i, d in enumerate(done):
        if d:
            episode_count += 1
            inf = info[i]
            reason = inf.get("term_reason", "timeout")

            successes.append(float(reason == "success"))
            drops.append(float(reason == "dropped"))
            timeouts.append(float(reason == "timeout"))

            if "red_face_up_alignment" in inf:
                alignments.append(inf["red_face_up_alignment"])
            print(f"Episode {episode_count}: " f"{reason} | " f"align={inf.get('red_face_up_alignment', None)}")

print("\n===== EVALUATION RESULTS =====")
print(f"Success rate: {np.mean(successes):.3f}")
print(f"Drop rate:    {np.mean(drops):.3f}")
print(f"Timeout rate: {np.mean(timeouts):.3f}")

if len(alignments) > 0:
    print(f"Mean alignment: {np.mean(alignments):.3f}")