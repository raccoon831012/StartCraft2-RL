import numpy as np
from gym import spaces
from pysc2.lib import actions, features
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv, BaseMovement3dEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'DefeatRoaches'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_SELECT_POINT = actions.FUNCTIONS.select_point.id

_SELECT_RECT = actions.FUNCTIONS.select_rect.id

_NO_OP = actions.FUNCTIONS.no_op.id
CMD = {0:_SELECT_ARMY,1:_SELECT_POINT,2:_ATTACK_SCREEN}

class DefeatRoaches1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)
    
    def _get_observation_space(self):
        screen_shape = (4, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space

    def _extract_observation(self, obs):
        shape = (1, ) + self.observation_space.shape[1:]
        obs = np.concatenate((
            obs.observation["feature_screen"][_PLAYER_RELATIVE].reshape(shape),
            obs.observation['feature_screen'][_UNIT_HIT_POINTS].reshape(shape),
            obs.observation['feature_screen'][_SELECTED].reshape(shape),
            obs.observation['feature_screen'][_UNIT_TYPE].reshape(shape)
        ), axis=0)
        return obs
    
    def _translate_action(self, action):
        if action < 0 or action > self.action_space.n or _ATTACK_SCREEN not in self.available_actions:
            return [_NO_OP]

        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        target = list(np.unravel_index(action, screen_shape))
        #print("location",target)
        return [_ATTACK_SCREEN, _NOT_QUEUED, target]

class DefeatRoaches2dEnv(DefeatRoaches1dEnv):
    def _get_action_space(self):
        return spaces.MultiDiscrete([len(CMD),84,84])
    
    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        #print([CMD[action[0]], action[1:]])
        if CMD[action[0]] not in self.available_actions:
            return [_NO_OP]
        elif CMD[action[0]] == _SELECT_ARMY:
            return [_SELECT_ARMY, _SELECT_ALL]
        elif CMD[action[0]] == _NO_OP:
            return [_NO_OP]

        return [CMD[action[0]], _NOT_QUEUED, action[1:]]

class DefeatRoaches3dEnv(BaseMovement3dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)
    
    def step(self, action):
        action = self._translate_action(action)
        #print(action[0])
        obs, reward, done, info = self._safe_step(action)
        if obs is None:
            return None, 0, True, {}
        screen_obs , mini_obs = self._extract_observation(obs)
        return screen_obs , mini_obs, reward, done, info

    def _get_action_space(self):
        return spaces.MultiDiscrete([2,84,84,84,84])
    def _translate_action(self, action):
        command = _SELECT_RECT
        if action[0] > 0:
            command = _ATTACK_SCREEN
        else:
            command = _SELECT_RECT
        if command not in self.available_actions:
            return [_NO_OP]
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        if command == _ATTACK_SCREEN:
            return [_ATTACK_SCREEN, _NOT_QUEUED, action[1:]]
        elif command == _SELECT_RECT:
            return [_SELECT_RECT, _NOT_QUEUED, action[1:3],action[3:]]
        #print("location",target)
        return [_NO_OP]