import numpy as np
import random
from gym import spaces
from pysc2.lib import actions, features,units
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv

_MAP_NAME = 'CollectMineralsAndGas'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_FOOD_USED = features.Player.food_used
_FOOD_CAP = features.Player.food_cap

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_HARVEST_GATHER_SCREEN  = actions.FUNCTIONS.Harvest_Gather_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NO_OP = actions.FUNCTIONS.no_op.id

_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id # 建立兵營的id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id # 建立供應站的id
_RALLY_WORKERS_SCREEN = actions.FUNCTIONS.Rally_Workers_screen.id

_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id # 訓練兵的快捷鍵
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id #訓練農夫的快捷鍵

_TERRAN_SCV = units.Terran.SCV.value #45
_NEUTRAL_MINERAL_FIELD = units.Neutral.MineralField.value #341

_SELECT_ALL = [0]
_QUEUED = [1]
_NOT_QUEUED = [0]
_SCREEN = [0]

CMD = {0:_SELECT_POINT,1:_TRAIN_SCV,2:_HARVEST_GATHER_SCREEN,3:_BUILD_REFINERY_SCREEN,4:_RALLY_WORKERS_SCREEN,5:_SELECT_IDLE_WORKER}
class CollectMineralsAndGas1dEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)
    def _post_reset(self):
        obs, _, _, _ = self._safe_step([_NO_OP])
        if _SELECT_RECT in self.available_actions:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            start = [unit_x[0],unit_y[0]]
            end = [unit_x[-1],unit_y[-1]]
            obs, _, _, _ = self._safe_step([_SELECT_RECT,_SELECT_ALL,start,end])
        if _HARVEST_GATHER_SCREEN in self.available_actions:
            unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
            target = [unit_x[10], unit_y[10]]
            obs, _, _, _ = self._safe_step([_HARVEST_GATHER_SCREEN, _NOT_QUEUED, target])
        
        obs, _, _, _ = self._safe_step([_NO_OP])
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
        if unit_y.any():
            target = [unit_x[len(unit_x)//2], unit_y[len(unit_y)//2]]
            #return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            obs, reward, done, info = self._safe_step([_SELECT_POINT, _SCREEN, target])
        self.obs_player = obs.observation['player']
        obs = self._extract_observation(obs)
        return obs

    def step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._safe_step(action)
        self.obs_player = obs.observation['player']
        if obs is None:
            return None, 0, True, {}
        obs = self._extract_observation(obs)
        return obs, reward, done, info

    def _get_observation_space(self):
        screen_shape = (4, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
        return space
    
    def _extract_observation(self, obs):
        #feature_minimap/feature_screen
        shape = (1, ) + self.observation_space.shape[1:]
        
        obs = np.concatenate((
            obs.observation["feature_screen"][_PLAYER_RELATIVE].reshape(shape),
            obs.observation['feature_screen'][_UNIT_HIT_POINTS].reshape(shape),
            obs.observation['feature_screen'][_SELECTED].reshape(shape),
            obs.observation['feature_screen'][_UNIT_TYPE].reshape(shape),

        ), axis=0)
        return obs

    def _get_action_space(self):
        return spaces.MultiDiscrete([len(CMD),84,84])

    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        command = CMD[action[0]]
        queued = _NOT_QUEUED
        location = action[1:]
        if command not in self.available_actions or command == _NO_OP:
            return [_NO_OP]
        #elif command == _SELECT_POINT:
        #    return [command, _SCREEN, location]
        elif command == _SELECT_IDLE_WORKER:
            return [command, _SELECT_ALL]
        elif command == _TRAIN_SCV:
            return [command, _QUEUED]
        #elif command == _BUILD_REFINERY_SCREEN:
        #    return [command, _SCREEN, location]
        #elif command == _RALLY_WORKERS_SCREEN:
        return [command, queued, location]