import retro
from gym.wrappers import TimeLimit
from RandomAgent import MarioDiscretizer
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gym.wrappers import ResizeObservation, GrayScaleObservation
from gym.envs.classic_control import rendering
import numpy as np
from RandomAgent import repeat_upsample

model = PPO.load("tmp/best_model.zip")
#model=PPO.load("SuperMarioBros-Nes.zip")
def main():
    steps = 0
    viewer = rendering.SimpleImageViewer(maxwidth=1000)
    env = retro.make(game='SuperMarioBros-Nes')
    env = MarioDiscretizer(env)
    env = TimeLimit(env,4000)
    #env = MaxAndSkipEnv(env, 2)
    env=ResizeObservation(env,84)
    env=GrayScaleObservation(env, keep_dim=True)
    
    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        rgb = env.render('rgb_array')
        upscaled=repeat_upsample(rgb,4, 6)
        viewer.imshow(upscaled)
        if done:
            obs = env.reset()
        steps += 1
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Final Info")
    print(info)
    env.close()


if __name__ == "__main__":
    main()