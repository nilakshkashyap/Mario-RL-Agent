import retro
import gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from gym.wrappers import TimeLimit
from gym.envs.classic_control import rendering
import numpy as np

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!").format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

class CustomDiscretizer(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class MarioDiscretizer(CustomDiscretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[['RIGHT'],['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT','A','B'],['A'],['B']])

def main():
    steps = 0
    env = retro.make(game='SuperMarioBros-Nes', use_restricted_actions=retro.Actions.DISCRETE)
    print('retro.Actions.DISCRETE action_space', env.action_space)
    env.close()
    viewer = rendering.SimpleImageViewer(maxwidth=1000)
    env = retro.make(game='SuperMarioBros-Nes')
    env = MarioDiscretizer(env)
    print('MarioDiscretizer action_space', env.action_space)
    #env = MaxAndSkipEnv(env, 2)
    env=TimeLimit(env,2000)
    obs = env.reset()
    print(obs.shape)
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        #env.render()
        rgb = env.render('rgb_array')
        upscaled=repeat_upsample(rgb,3, 4)
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
