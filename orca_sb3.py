from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from orca_sim import OrcaHandRightCubeOrientation

from collections import deque
import numpy as np
import os

class DiagnosticsCallback(BaseCallback):
    def __init__(self, window=100):
        super().__init__()
        self.window = window

        self.successes = deque(maxlen=window)
        self.drops = deque(maxlen=window)
        self.timeouts = deque(maxlen=window)
        self.alignments = deque(maxlen=window)

    def _on_step(self):
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for info, done in zip(infos, dones):

            if done:
                reason = info.get("term_reason", "timeout")

                self.successes.append(float(reason == "success"))
                self.drops.append(float(reason == "dropped"))
                self.timeouts.append(float(reason == "timeout"))

            if done and "red_face_up_alignment" in info:
                self.alignments.append(info["red_face_up_alignment"])

        if len(self.successes) > 0:
            self.logger.record("custom/success_rate", np.mean(self.successes))
            self.logger.record("custom/drop_rate", np.mean(self.drops))
            self.logger.record("custom/timeout_rate", np.mean(self.timeouts))

        if len(self.alignments) > 0:
            self.logger.record("custom/mean_alignment", np.mean(self.alignments))

        return True

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, save_freq):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"vecnormalize_{self.num_timesteps}_steps.pkl"
            )
            self.training_env.save(path)

        return True

def make_env():
    return Monitor(
        OrcaHandRightCubeOrientation(
            version="v2",
            render_mode=None,
            # initial_red_face="random",
            # cube_pos_xy_jitter=0.01,
        )
    )

env = DummyVecEnv([make_env])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10_000,
    verbose=1,
    tensorboard_log="./tb_logs"
)

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="./checkpoints/",
    name_prefix="sac_cube_orientation"
)

vecnorm_callback = SaveVecNormalizeCallback(save_path="./checkpoints/", save_freq=25000)

callback = CallbackList([DiagnosticsCallback(), checkpoint_callback, vecnorm_callback])

model.learn(total_timesteps=500000, callback=callback)

model.save("sac_cube_orientation_final")
env.save("vecnormalize_final.pkl")