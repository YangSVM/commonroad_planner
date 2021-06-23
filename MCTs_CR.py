# -*- coding: UTF-8 -*-
# 写于0623，用于调用 MCTs_v2 以及 grid_lanelet 两个代码的接口。

from typing import NewType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import str_len
from intersection_planner import conf_agent_checker
from detail_central_vertices import detail_cv
from intersection_planner import distance_lanelet

from grid_lanelet import edit_scenario4test
from grid_lanelet import lanelet_network2grid
from grid_lanelet import ego_pos2tree
from MCTs_v2 import NaughtsAndCrossesState
from MCTs_v2 import mcts

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
    lanelet_network = scenario.lanelet_network
    lanelet_id_matrix = lanelet_network2grid(lanelet_network)
    # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
    grid, d = ego_pos2tree(ego_pos_init, lanelet_id_matrix, lanelet_network)
    print('grid', grid, 'd', d)

    b = [2, 50, 15]
    ini = [[80, 10], [80, 10], [0, 0], [0, 0], [0, 0]]
    initialState = NaughtsAndCrossesState(b, ini)
    searcher = mcts(iterationLimit=5000)  # 改变循环次数或者时间
    action = searcher.search(initialState=initialState)  # 一整个类都是其状态
    print(action.act)
