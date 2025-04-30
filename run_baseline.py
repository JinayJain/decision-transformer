import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from util import build_env, seed_everything


def main():
    seed_everything(42)
    gym.register_envs(ale_py)
    env = build_env(1, is_eval=True)
    model = DQN.load("artifacts/dqn_breakout")
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode="human")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
