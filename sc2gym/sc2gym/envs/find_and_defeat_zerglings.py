import numpy as np
from gym import spaces
from pysc2.lib import actions, features
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv
from sc2gym.envs.collect_mineral_shards import CollectMineralShardsGroupsEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'FindAndDefeatZerglings'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_MINIMAP_PLAYER_RELATIVE_SCALE = features.MINIMAP_FEATURES.player_relative.scale

_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

CMD = {0:_SELECT_ARMY,1:_SELECT_POINT,2:_ATTACK_SCREEN,3:_NO_OP,4:_MOVE_CAMERA}

class FindAndDefeatZerglings1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

    def _translate_action(self, action):
        if action < 0 or action > self.action_space.n:
            return [_NO_OP]
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        target = list(np.unravel_index(action, screen_shape))
        #print("location",target)
        return [_ATTACK_SCREEN, _NOT_QUEUED, target]

class FindAndDefeatZerglingsGroupsEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

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
        elif CMD[action[0]] == _MOVE_CAMERA:
            for ix, act in enumerate(action):
                if ix!=0 and act < 20 or act > 50:
                    return [_NO_OP]
            return [CMD[action[0]], action[1:]]
        return [CMD[action[0]], _NOT_QUEUED, action[1:]]
    
    def _get_observation_space(self):
        screen_shape = (3, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space

    def _extract_observation(self, obs):
        #feature_minimap/feature_screen
        shape = (1, ) + self.observation_space.shape[1:]
        
        obs = np.concatenate((
            obs.observation["feature_screen"][_PLAYER_RELATIVE].reshape(shape),
            obs.observation['feature_screen'][_UNIT_HIT_POINTS].reshape(shape),
            obs.observation['feature_screen'][_SELECTED].reshape(shape),
        ), axis=0)
        return obs

class FindAndDefeatZerglingsGroups2dEnv(FindAndDefeatZerglingsGroupsEnv):
    def step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._safe_step(action)
        if obs is None:
            return None, 0, True, {}
        screen_obs , mini_obs = self._extract_observation(obs)
        return screen_obs , mini_obs, reward, done, info

    def _post_reset(self):
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        screen_obs , mini_obs = self._extract_observation(obs)
        return screen_obs , mini_obs

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self._get_observation_space()
        else:
            return self._observation_space

    def _get_observation_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"]
        minimap_shape = self.observation_spec[0]["feature_minimap"]
        #print(minimap_shape)
        screen_space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        minimap_shape = spaces.Box(low=0, high=_MINIMAP_PLAYER_RELATIVE_SCALE, shape=minimap_shape, dtype=np.int32)
        return screen_space,minimap_shape
        
    def _extract_observation(self, obs):
        #obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        #obs = np.array(obs.reshape(self.observation_space.shape))
        screen_obs = obs.observation["feature_screen"]
        mini_obs = obs.observation["feature_minimap"]
        return screen_obs , mini_obs