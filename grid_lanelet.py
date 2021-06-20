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

    return lanelet_id


def ego_pos2tree(ego_pos,  lanelet_id_matrix, lanelet_network, scenario):
    '''通过自车位置，标记对应的1。并且返回自车位置相对于左上角lanelet的中心线的frenet坐标系的距离。
    param: ego_pos:车辆位置.[x,y]
    param: lanelet_ids:lanelet网格矩阵。每个位置为对应的lanelet_id.
    param: lanlet_network. common-road lanelet_network
    return: grid: 与lanelet_ids维度相同的矩阵。自车在的位置为1，其余为0
    return: ego_s: 自车位置相对于左上角lanelet的中心线的frenet坐标系的沿中心线方向距离
    return: obstacle_states: 他车状态矩阵。m(车道数)*2. 表示每条道路上车的状态，每条道路上最多一辆车。如果该车道上没有车，则是[-1,-1]
        如果有车，则是[s, v]。s表示frenet坐标系(s-d)坐标值.v是他车沿车方向速度。

    '''
    grid = np.zeros(lanelet_id_matrix.shape)
    lanelet_id = lanelet_network.find_lanelet_by_position([ego_pos])[0]
    for lanelet_id_i in lanelet_id:
        index = np.where(lanelet_id_matrix == lanelet_id_i)
        grid[index] =1
    # 取lanelet_ids中左上角的中心线为frenet坐标系参考线，返回自车当前的s
    lanelet00 = lanelet_network.find_lanelet_by_id(lanelet_id_matrix[0, 0])
    center_vertices = lanelet00.center_vertices
    cv, _,s_cv = detail_cv(center_vertices)
    # 找最近点
    ego_s = distance_lanelet(cv, s_cv, center_vertices[0,:],ego_pos)

    shape = lanelet_id_matrix.shape
    obstacle_states = -1*np.ones((shape[0],2))
    obstacles = scenario.obstacles
    for i in range(len(lanelet_id_matrix)):
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id_matrix[i, 0])
        obstacles_on = lanelet.get_obstacles(obstacles)
        if len(obstacles_on) == 0:
            continue
        elif len(obstacles_on) ==1:
            obstacle = obstacles_on[0]
        else:
            print('warninig: 多辆车')
            obstacle = obstacles_on[0]
        state = obstacle.state_at_time(0)
        s = distance_lanelet(cv, s_cv, center_vertices[0,:],state.position)
        obstacle_states[i, 0] = s
        obstacle_states[i, 1] = state.velocity


    return grid, ego_s, obstacle_states


def edit_scenario4test(scenario, ego_init_pos):
    # ------------ 删除车辆，使得每个车道最多一辆车 ------------------------------------
    # scenario.remove_obstacle(obstacle id)
    lanelet_network = scenario.lanelet_network

    lanelet_ids = [14,17,20,23,26]
    conf_point = []
    for i in range(len(lanelet_ids)):
        conf_point.append(ego_init_pos)

    obstacle_remain = [252, 254,234,126]
    obstacle_remove = []
    for i in range(len(scenario.obstacles)):
        obstacle_id = scenario.obstacles[i].obstacle_id
        if obstacle_id in obstacle_remain:
            continue
        obstacle_remove.append(obstacle_id)
    for obstalce_id_remove in obstacle_remove:
        scenario.remove_obstacle(scenario.obstacle_by_id(obstalce_id_remove))
    return scenario

if __name__=='__main__':
    # 测试MCTS算法

    # -------------- 固定写法。从common road中读取场景 -----------------
    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/NGSIM/US101')
    # 文件名
    id_scenario = 'USA_US101-16_1_T-1'
    path_scenario =os.path.join(path_scenario_download, id_scenario + '.xml')    
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    # -------------- 读取结束 -----------------

    # 原有场景车辆太多。删除部分车辆
    ego_pos_init = planning_problem.initial_state.position
    scenario = edit_scenario4test(scenario, ego_pos_init)

    # ---------------可视化修改后的场景 ------------------------------
    plt.figure(figsize=(25, 10))
    # 画一小段展示一下
    for  i in range(0):
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
    # ---------------可视化 end ------------------------------

    # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
    lanelet_network  = scenario.lanelet_network
    lanelet_id_matrix  = lanelet_network2grid(lanelet_network)
    print('lanelet_id_matrix: ', lanelet_id_matrix)

    # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
    grid, ego_d, obstacle_states =ego_pos2tree(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario)
    print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)
    v_ego = planning_problem.initial_state.velocity
    print('自车初始速度： ', v_ego)
    print('他车矩阵', obstacle_states)
    print('单独提取',obstacle_states[0][0])

