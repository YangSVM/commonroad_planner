# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object
import os
from detail_central_vertices import detail_cv
from intersection_planner import distance_lanelet
import numpy as np
import matplotlib.pyplot as plt

from grid_lanelet import edit_scenario4test
from grid_lanelet import lanelet_network2grid
from grid_lanelet import ego_pos2tree
from MCTs_02 import NaughtsAndCrossesState
from MCTs_02 import mcts
from MCTs_02 import vehicles

if __name__ == '__main__':
    # -------------- 固定写法。从common road中读取场景 -----------------
    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/NGSIM/US101')
    # 文件名
    id_scenario = 'USA_US101-16_1_T-1'
    path_scenario = os.path.join(path_scenario_download, id_scenario + '.xml')
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    # -------------- 读取结束 -----------------
    # 原有场景车辆太多。删除部分车辆
    ego_pos_init = planning_problem.initial_state.position
    scenario = edit_scenario4test(scenario, ego_pos_init)
    # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
    lanelet_network  = scenario.lanelet_network
    lanelet_id_matrix  = lanelet_network2grid(lanelet_network)
    print('lanelet_id_matrix: ', lanelet_id_matrix)

    # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
    T = 50           # 5*0.1=0.5.返回0.5s时周车的状态。注意下面函数返回的自车状态仍然是初始时刻的。
    grid, ego_d, obstacle_states =ego_pos2tree(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, T)
    print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)
    v_ego = planning_problem.initial_state.velocity
    print('自车初始速度： ', v_ego)
    print('他车矩阵', obstacle_states)

    a = [[100, 5], [100, 5], [50, 10], [50, 10], [50, 10]]
    ini = vehicles(a)
    b = [2, 20, 15]
    initialState = NaughtsAndCrossesState(b)
    searcher = mcts(iterationLimit=5000)  # 改变循环次数或者时间
    action = searcher.search(initialState=initialState)  # 一整个类都是其状态
    print(action.act)
