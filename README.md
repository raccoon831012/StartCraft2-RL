# StartCraft2-RL

簡介
-----
使用RL中的PPO玩google星海所提供的的小遊戲[1]或是自製遊戲的小遊戲

使用環境
----
    Anaconda3 
    Python 3.6
    Pysc2 2.0
    Tensorflow 1.13.1
    Sc2gym-master https://github.com/islamelnabarawy/sc2gym (本專案中的sc2gym檔案已添加全部小遊戲環境)

環境安裝
----
    1.	Anaconda安裝
        官方網站：https://www.anaconda.com/distribution/
    2.	創建虛擬環境
        conda create -n (env_name) pip python=3.6
    3.	啟動虛擬環境
        activate (env_name)
    4.	安裝tensorflow
        pip install tensorflow==1.13.1
    5.	安裝pysc2
        開啟anaconda prompt，cd至下載之pysc2資料夾
        pip install –e . 進行安裝
    6.	安裝Starcraft2
        下載Battle.net-Setup.exe並執行安裝
        注意：此步驟需要網路支援安裝
        登入平台後下載星海爭霸2遊戲
    7.	安裝訓練地圖
        下載mini_games資料夾並放置到您的StarcraftII/Maps/ 資料夾中
    8.	測試Pysc2安裝成功
        python -m pysc2.bin.agent --map Simple64
    9.	安裝sc2-gym
        下載sc2gym-master後cd sc2gym-master 
        cd sc2gym-master 
        pip install –e .
    10.	修改agent以及model
    11.	更改訓練地圖
        甲、Pysc2 將地圖新增至Maps後，修改pysc2/maps/minigames.py，新增欲使用之地圖
        乙、Sc2gym-master修改sc2gym/__init__.py 新增register以及修改/envs中的agent，依照class順序綁定v0.v1.v2….
        丙、以預設好的環境中有分三類：單一維度、有小地圖及沒有小地圖，請根據該環境類型用不同的程式碼進行訓練
        i.	單一維度
            SC2MoveToBeacon-v0
            SC2CollectMineralShards-v0
            SC2FindAndDefeatZerglings-v0
            SC2DefeatRoaches-v0
            SC2DefeatZerglingsAndBanelings-v0
        ii.	沒有小地圖
            SC2MoveToBeacon-v1
            SC2CollectMineralShards-v1
            SC2CollectMineralShards-v2
            SC2CollectMineralShards-v4
            SC2FindAndDefeatZerglings-v1
            SC2DefeatRoaches-v1
            SC2DefeatZerglingsAndBanelings-v1
            SC2CollectMineralsAndGas-v0
            SC2BuildMarines-v0
            SC2BuildMarinesBigMap-v0
            BuildAndDefeat-v0
            BuildAndDefeat-v1
            BuildAndDefeat-v2
        iii.	有小地圖
            SC2MoveToBeacon-v2
            SC2CollectMineralShards-v3
            SC2FindAndDefeatZerglings-v2
            SC2DefeatRoaches-v2
            SC2DefeatZerglingsAndBanelings-v2
            SC2BuildMarines-v1 
            SC2BuildMarinesBigMap-v1
            BuildAndDefeat-v3
            BuildAndDefeat-v4
RL使用訓練
----
    1.	進入SC2_RL資料夾
    2.	從頭開始訓練根據上述的環境不同選擇不同的資料夾及程式碼
      甲、單一維度
          進入PPO的資料夾點選main.py修改Train=True
          cmd上執行python main.py
      乙、沒有小地圖
          進入PPO的資料夾點選main_group.py修改Train=True
          cmd上執行python main_group.py
      丙、有小地圖
          進入SC2_PPO的資料夾點選main_sc2.py修改Train=True
          cmd上執行python main_sc2.py
    3.  訓練後不再訓練觀看結果 
          將main.py, main_group.py或是main_sc2.py中的Train = False並且執行即可觀看

參考資料
-----
[1]StarCraft II: A New Challenge for Reinforcement Learning
https://arxiv.org/abs/1708.04782
