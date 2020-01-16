"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
from gym import spaces
from pysc2.lib import actions, features
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv, BaseMovement3dEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'CollectMineralShards'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

_SELECTED = features.SCREEN_FEATURES.selected.index

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_GROUP_RECALL = [0]
_GROUP_SET = [1]

_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_SINGLE = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id


class CollectMineralShards1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

    def _post_reset(self):
        obs = self._init_control_groups()
        obs = self._extract_observation(obs[0])
        return obs

    def _init_control_groups(self):
        obs = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [0]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [2]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [1]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [3]])
        obs = self._safe_step([_NO_OP]) 
        return obs
    
    def step(self, action):
        if _MOVE_SCREEN not in self.available_actions:
            self._init_control_groups()
        return super().step(action)

    def _get_action_space(self):
        return spaces.Discrete(3*84*84)

    def _translate_action(self, action):
        if action < 0 or action > self.action_space.n:
            return [_NO_OP]

        target = list(np.unravel_index(action, (3,84,84)))
        self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [target[0] + 1]])
        return [_MOVE_SCREEN, _NOT_QUEUED, target[1:]]

    def _get_observation_space(self):
        screen_shape = (2, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space

    def _extract_observation(self, obs):
        shape = (1, ) + self.observation_space.shape[1:]
        obs = np.concatenate((
            obs.observation["feature_screen"][_PLAYER_RELATIVE].reshape(shape),
            obs.observation['feature_screen'][_SELECTED].reshape(shape)
        ), axis=0)
        return obs


class CollectMineralShards2dEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

class CollectMineralShardsGroupsEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

    def _post_reset(self):
        obs = self._init_control_groups()
        obs = self._extract_observation(obs[0])
        return obs

    def _init_control_groups(self):
        obs = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [0]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [2]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [1]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [3]])
        obs = self._safe_step([_NO_OP]) 
        return obs
    
    def step(self, action):
        if _MOVE_SCREEN not in self.available_actions:
            self._init_control_groups()
        return super().step(action)

    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.MultiDiscrete([2] + [s-1 for s in screen_shape])

    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [action[0] + 2]])
        return [_MOVE_SCREEN, _NOT_QUEUED, action[1:]]

    def _get_observation_space(self):
        screen_shape = (2, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space

    def _extract_observation(self, obs):
        shape = (1, ) + self.observation_space.shape[1:]
        obs = np.concatenate((
            obs.observation["feature_screen"][_PLAYER_RELATIVE].reshape(shape),
            obs.observation['feature_screen'][_SELECTED].reshape(shape)
        ), axis=0)
        return obs

class CollectMineralShardsGroups2dEnv(BaseMovement3dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)

    def _post_reset(self):
        obs = self._init_control_groups()
        obs = self._extract_observation(obs[0])
        return obs

    def _init_control_groups(self):
        obs = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [0]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [2]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [1]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [3]])
        obs = self._safe_step([_NO_OP]) 
        return obs
        
    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.MultiDiscrete([2] + [s-1 for s in screen_shape])

    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [action[0] + 2]])
        return [_MOVE_SCREEN, _NOT_QUEUED, action[1:]]

class CollectMineralShardsGroups3dEnv(CollectMineralShardsGroupsEnv):

    def _get_action_space(self):
        return spaces.MultiDiscrete([84,84])
    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [2]])
        return [_MOVE_SCREEN, _NOT_QUEUED, action[:]]
        
