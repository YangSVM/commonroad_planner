# -*- coding: UTF-8 -*-
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


def lanelet_network2grid(lanelet_network):
    '''明阳需求。将lanelet_network转成网格地图。v1仅适用于矩形直道场景。并道太复杂，暂不考虑。
    param: lanelet_network: commonroad lanelet_network
    return: grid: m(车道数)*n(纵向lanelet数目)矩阵. 值域 [0,1,-1]。-1不可通行，0可通行，1自车所在lanelet。
    return: lanelet_id： 对应的车道的lanelet_id
    '''
    lanelet_id = []
    # lanelet00. 左上角的lanelet
    lanelet00 = lanelet_network.lanelets[0]
    while len(lanelet00.predecessor) > 0:
        lanelet00_id = lanelet00.predecessor[0]
        lanelet00 = lanelet_network.find_lanelet_by_id(lanelet00_id)

    while lanelet00.adj_left is not None:
        if lanelet00.adj_left_same_direction:
            lanelet00_id = lanelet00.adj_left
            lanelet00 = lanelet_network.find_lanelet_by_id(lanelet00_id)            
    
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
    lanelet_id = np.array(lanelet_id, dtype=int)

    # 初始化。全部可通行
    grid = np.zeros(lanelet_id.shape)

    return grid, lanelet_id


def ego_pos2tree(ego_pos, grid, lanelet_ids, lanelet_network):
    lanelet_id = lanelet_network.find_lanelet_by_position([ego_pos])[0]
    for lanelet_id_i in lanelet_id:
        index = np.where(lanelet_ids == lanelet_id_i)
        grid[index] =1
    # 取lanelet_ids中左上角的中心线为frenet坐标系参考线，返回自车当前的s
    lanelet00 = lanelet_network.find_lanelet_by_id(lanelet_ids[0, 0])
    center_vertices = lanelet00.center_vertices
    cv, _,s_cv = detail_cv(center_vertices)
    # 找最近点
    d = distance_lanelet(cv, s_cv, center_vertices[0,:],ego_pos)

    return grid, d






if __name__=='__main__':
    # 测试MCTS算法


    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/NGSIM/US101')
    # 文件名
    id_scenario = 'USA_US101-16_1_T-1'

    path_scenario =os.path.join(path_scenario_download, id_scenario + '.xml')
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # 可视化目前场景

    # plt.figure(figsize=(25, 10))
    # for  i in range(100):
    #     plt.clf()
    #     draw_parameters = {'time_begin': i}

    #     draw_object(scenario, draw_params=draw_parameters)
    #     draw_object(planning_problem_set)
    #     plt.gca().set_aspect('equal')
    #     plt.pause(0.01)

    # 获取初始位置
    ego_init_pos = planning_problem.initial_state.position


    # ------------ 删除车辆，使得每个车道最多一辆车 ------------------------------------
    # scenario.remove_obstacle(obstacle id)
    lanelet_network = scenario.lanelet_network

    lanelet_ids = [14,17,20,23,26]
    conf_point = []
    for i in range(len(lanelet_ids)):
        conf_point.append(ego_init_pos)

    dict_lanelet_agent = conf_agent_checker(scenario, lanelet_ids, conf_point, 0)
    obstacle_remain = [agent for agent in dict_lanelet_agent.values()]
    obstacle_remove = []
    for i in range(len(scenario.obstacles)):
        obstacle_id = scenario.obstacles[i].obstacle_id
        if obstacle_id in obstacle_remain:
            continue
        obstacle_remove.append(obstacle_id)
    for obstalce_id_remove in obstacle_remove:
        scenario.remove_obstacle(scenario.obstacle_by_id(obstalce_id_remove))

    # --------------------------------------------删除车辆完成。可视化 ------------------------------
    plt.figure(figsize=(25, 10))
    # 画一小段展示一下
    for  i in range(10):
        plt.clf()
        draw_parameters = {
            'time_begin':i, 
            'scenario':
            { 'dynamic_obstacle': { 'show_label': True, },
                'lanelet_network':{'lanelet':{'show_label': False,  },} ,
            },
        }

        draw_object(scenario, draw_params=draw_parameters)
        draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.pause(0.001)
        # plt.show()
    plt.close()

    # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置

    ego_pos = planning_problem.initial_state.position

    grid, lanelet_id  = lanelet_network2grid(lanelet_network)
    grid, d =ego_pos2tree(ego_pos, grid, lanelet_id, lanelet_network)
    print('grid',grid,'d',d)
