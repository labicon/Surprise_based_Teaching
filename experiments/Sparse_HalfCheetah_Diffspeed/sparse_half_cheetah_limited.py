import numpy as np
import gym
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class SparseHalfCheetahLimitedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        if abs(xposafter) > 5.0:
            reward = (xposafter - xposbefore) / self.dt
        else:
            reward = 0.0
        # Limit the velocity
        vel = (xposafter - xposbefore) / self.dt
        if abs(vel) > 3.0:
        	done = True
        else:
        	done = False
        return ob, reward, done, dict(reward_run=reward, reward_ctrl=0.0)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
