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
from pysc2.lib import actions, features, units
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv
from sc2gym.envs.collect_mineral_shards import CollectMineralShardsGroupsEnv

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import app
import random
from random import randint
#import pandas as pd
import math
import time

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'BuildAndDefeat'
#_MAP_NAME  = 'Simple64'
#_MAP_NAME  = 'FindAndDefeatZerglings'
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id # 建立兵營的id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id # 建立供應站的id
_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_FOOD_USED = features.Player.food_used
_FOOD_CAP = features.Player.food_cap
_HARVEST_GATHER_SCREEN  = actions.FUNCTIONS.Harvest_Gather_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_MINIMAP_PLAYER_RELATIVE_SCALE = features.MINIMAP_FEATURES.player_relative.scale
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id # 訓練兵的快捷鍵
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id # 訓練死神的快捷鍵
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id #訓練農夫的快捷鍵
_TERRAN_SCV = units.Terran.SCV.value #45
_NEUTRAL_MINERAL_FIELD = units.Neutral.MineralField.value #341
_NO_OP = actions.FUNCTIONS.no_op.id
_MARINE = 48
_ZERGLING = 105
_CommandCenter = 18
_Hydralisk = 107
_Observer = 82
_Barracks = 21
_PhotonCannon = 66
_SpineCrawler = 98
_SporeCrawler = 99
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SCREEN = [0]
_QUEUED = [1]
#CMD = {0:_SELECT_ARMY,1:_SELECT_POINT,2:_ATTACK_SCREEN,3:_NO_OP,4:_MOVE_CAMERA}
Defeat_CMD = {0:_SELECT_ARMY,1:_SELECT_POINT,2:_ATTACK_SCREEN,3:_NO_OP,4:_MOVE_CAMERA}
Build_CMD = {0:_SELECT_POINT,1:_BUILD_BARRACKS,2:_BUILD_SUPPLYDEPOT,3:_TRAIN_MARINE,4:_TRAIN_SCV,5:_SELECT_IDLE_WORKER}
Bottom_CMD = [Defeat_CMD,Build_CMD]
CMD = {0:_SELECT_POINT,1:_BUILD_SUPPLYDEPOT,2:_BUILD_BARRACKS,3:_TRAIN_MARINE,4:_TRAIN_SCV,5:_TRAIN_REAPER
,6:_SELECT_IDLE_WORKER,7:_ATTACK_SCREEN,8:_MOVE_CAMERA,9:_SELECT_ARMY,10:_TERRAN_SCV,11:_MOVE_SCREEN}

class BuildAndDefeat1dEnv(BaseMovement1dEnv):
   def __init__(self, **kwargs):
      self.scv_number = 12
      super().__init__(map_name=_MAP_NAME, **kwargs)

   def _translate_action(self, action):
      if action < 0 or action > self.action_space.n:
         return [_NO_OP]
      screen_shape = self.observation_spec[0]["feature_screen"][1:]
      target = list(np.unravel_index(action, screen_shape))
      #print("location",target)
      return [_ATTACK_SCREEN, _NOT_QUEUED, target]

class BuildAndDefeatGroupsEnv(BaseMovement2dEnv):
   def __init__(self, **kwargs):
      self.scv_number = 12
      super().__init__(map_name=_MAP_NAME, **kwargs)

   def _get_action_space(self):
      return spaces.MultiDiscrete([len(CMD),84,84])
   
   def _translate_action(self, action):
      for ix, act in enumerate(action):
         if act < 0 or act > self.action_space.nvec[ix]:
            return [_NO_OP]
      command = CMD[action[0]]
      #print(command)
      queued = _NOT_QUEUED
      location = action[1:]
      if command not in self.available_actions or command == _NO_OP:
         #print("false:", command)
         return [_NO_OP]
      elif command == _SELECT_POINT:
         return [command, _SCREEN, location]
      elif command == _SELECT_IDLE_WORKER:
         return [command, _SELECT_ALL]
      elif command == _TRAIN_MARINE or command == _TRAIN_REAPER:
         return [command, _QUEUED]
      elif command == _TRAIN_SCV:
         if self.scv_number > 17:
               return [_NO_OP]
         self.scv_number+=1
         return [command, _QUEUED]
      elif command == _BUILD_BARRACKS:
         return [command, _SCREEN, location]    
      elif command == _BUILD_SUPPLYDEPOT:
         if self.obs_player[_FOOD_CAP] - self.obs_player[_FOOD_USED] <10:
               return [command, queued, location]
         else: 
               return [_NO_OP]
      elif command == _SELECT_ARMY:
         return [command, _SELECT_ALL]
      elif command == _ATTACK_SCREEN or command == _MOVE_SCREEN:
         #print(command, _NOT_QUEUED, location)
         return [command, _NOT_QUEUED, location]
      elif command == _MOVE_CAMERA:
         for loc in location:
            if loc < 20 or loc > 50:
               return [_NO_OP]
         return [_MOVE_CAMERA, location]
      return [command, queued, location]   

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
      ), axis=0)
      return obs

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
         target = [unit_x[len(unit_x)//2], unit_y[len(unit_y)//2]]
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
      if obs is None:
         return None, 0, True, {}
      self.obs_player = obs.observation['player']
      obs = self._extract_observation(obs)
      return obs, reward, done, info

class BuildAndDefeatGroups2dEnv(BuildAndDefeatGroupsEnv):
   def step(self, action):
      action = self._translate_action(action)
      obs, reward, done, info = self._safe_step(action)
      if obs is None:
         return None,None, 0, True, {}
      self.obs_player = obs.observation['player']
      screen_obs , mini_obs = self._extract_observation(obs)
      return screen_obs , mini_obs, reward, done, info

   def _post_reset(self):
      obs, reward, done, info = self._safe_step([_NO_OP])
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
      screen_space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape, dtype=np.int32)
      minimap_shape = spaces.Box(low=0, high=_MINIMAP_PLAYER_RELATIVE_SCALE, shape=minimap_shape, dtype=np.int32)
      return screen_space , minimap_shape
        
   def _extract_observation(self, obs):
      #obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
      #obs = np.array(obs.reshape(self.observation_space.shape))
      screen_obs = obs.observation["feature_screen"]
      mini_obs = obs.observation["feature_minimap"]
      return screen_obs , mini_obs

class BuildAndDefeatGroups3dEnv(BuildAndDefeatGroups2dEnv):
   def step(self,agent,action):
      Sub_CMD = Bottom_CMD[agent]
      command = Sub_CMD[action[0]]
      for index,cmd in CMD.items():
         if cmd == command:
            action[0] = index
      action = self._translate_action(action)
      obs, reward, done, info = self._safe_step(action)

      if obs is None:
         return None,None, 0, True, {}
      self.obs_player = obs.observation['player']
      screen_obs , mini_obs = self._extract_observation(obs)
      return screen_obs , mini_obs, reward, done, info