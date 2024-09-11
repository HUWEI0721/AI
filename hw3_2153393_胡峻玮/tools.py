import cv2
import numpy as np
import gym
from gym import spaces
from collections import deque
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, x_values, scores, epsilons, filename, bestScore, time, lines=None):
        self.fig, self.ax = plt.subplots()
        self.ax2 = self.ax.twinx()

        self.x_values = x_values
        self.scores = scores
        self.epsilons = epsilons
        self.filename = filename
        self.lines = lines
        self.bestScore = bestScore
        self.totalTime = time

    def plot(self):
        self.ax.plot(self.x_values, self.epsilons, color="C0")
        self.ax.set_xlabel("Frame", color="C0")
        self.ax.set_ylabel("Epsilon", color="C0")
        self.ax.tick_params(axis='x', colors="C0")
        self.ax.tick_params(axis='y', colors="C0")

        self.ax2.plot(self.x_values, self.scores, color="C1")
        self.ax2.set_xlabel("Frame", color="C1")
        self.ax2.set_ylabel("Score", color="C1")
        self.ax2.tick_params(axis='x', colors="C1")
        self.ax2.tick_params(axis='y', colors="C1")
        self.ax2.yaxis.tick_right()

        if self.lines is not None:
            for line in self.lines:
                self.ax.axvline(x=line)

        self.ax.set_title(f'Best Score: {self.bestScore}, Total Time: {self.totalTime:.2f} Min')

        plt.savefig(self.filename)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.choice(range(1, self.noop_max + 1))
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        self._obs_buffer = []
        obs = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, (84, 84), interpolation=cv2.INTER_AREA)
        return new_frame[:, :, None].astype(np.uint8)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class BufferWrapper(gym.Wrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.n_steps = n_steps
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(shp[0], shp[1], shp[2] * n_steps), dtype=np.float32)
        self.frames = deque([], maxlen=n_steps)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames.extend([obs] * self.n_steps)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_steps
        return np.concatenate(self.frames, axis=-1)


# Use this modified BufferWrapper in your make_env function
def make_env(env_name, render_mode):
    if render_mode == 'human':
        env = gym.make(env_name, render_mode='human')
    elif render_mode == None:
        env = gym.make(env_name)
    env = NoopResetEnv(env, noop_max=30)
    env = SkipEnv(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = PreProcessFrame(env)
    env = ScaleFrame(env)
    env = ClipRewardEnv(env)
    env = BufferWrapper(env, 4)
    return env