# -*- coding: UTF-8 -*-
from typing import NewType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import str_len

from utils import plot_lanelet_network


def lanelet2grid(lanelet_network, ego_pos):
    '''明阳需求。将lanelet_network转成网格地图。v1仅适用于矩形直道场景。并道太复杂，暂不考虑。
    Args:
        lanelet_network: commonroad lanelet_network
        ego_pos: 自车位置. [x, y]
    Return:
        grid: m(车道数)*n(纵向lanelet数目)矩阵. 值域 [0,1,-1]。-1不可通行，0可通行，1自车所在lanelet。
        lanelet_id： 对应的车道的lanelet_id
    '''
    # 自车lanelet
    lanelet_ego_id = lanelet_network.find_lanelet_by_position([ego_pos])[0][0]
    lanelet_ego = lanelet_network.find_lanelet_by_id(lanelet_ego_id)
    

    lanelet_id = []
    j  = 0
    i = 0
    # lanelet00. 左上角的lanelet
    lanelet00 = lanelet_ego
    while len(lanelet00.predecessor) > 0:
        lanelet00_id = lanelet00.predecessor[0]
        lanelet00 = lanelet_network.find_lanelet_by_id(lanelet00_id)
        j += 1

    while lanelet00.adj_left is not None:
        if lanelet00.adj_left_same_direction:
            lanelet00_id = lanelet00.adj_left
            lanelet00 = lanelet_network.find_lanelet_by_id(lanelet00_id)            
            i += 1
    
    # 外循环，遍历所有车道数目（相邻车道）
    current_lanelet_id = lanelet00.lanelet_id
    while current_lanelet_id is not None:
        current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet_id)
        tmp_lanelet = current_lanelet
        lanelet_id_row = []
        lanelet_id_row.append(tmp_lanelet.lanelet_id)
        
        # 内循环。遍历子节点
        while len(tmp_lanelet.successor) >0:
            tmp_lanelet_id = tmp_lanelet.successor[0]
            tmp_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet_id)
            lanelet_id_row.append(tmp_lanelet.tmp_lanelet_id)
        lanelet_id.append(lanelet_id_row)

        # 外循环。遍历右节点
        if current_lanelet.adj_right_same_direction:
            current_lanelet_id = current_lanelet.adj_right
        else:
            break

    # 转为numpy矩阵
    lanelet_id = np.array(lanelet_id)

    # 初始化。全部可通行
    grid = np.zeros(lanelet_id.shape)
    # 自车位置为1
    grid[i, j] = 1

    return grid, lanelet_id

     


if __name__=='__main__':
    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/tiecun/codes/commonroad/commonroad-scenarios/scenarios/NGSIM/US101')
    # 文件名
    id_scenario = 'USA_US101-16_1_T-1'

    path_scenario =os.path.join(path_scenario_download, id_scenario + '.xml')
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # lanelet_network
    lanelet_network = scenario.lanelet_network

    # plot_lanelet_network(lanelet_network)
    # 
    ego_pose = scenario.obstacles[0].state_at_time(5).position
    grid, lanelet_id  = lanelet2grid(lanelet_network, ego_pose)
    
