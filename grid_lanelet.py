# -*- coding: UTF-8 -*-
from typing import Mapping
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object

import os

from numpy.lib.function_base import gradient


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


def ego_pos2tree(ego_pos,  lanelet_id_matrix, lanelet_network, scenario, T):
    '''通过自车位置，标记对应的1。并且返回自车位置相对于左上角lanelet的中心线的frenet坐标系的距离。
    param: ego_pos:车辆位置.[x,y]
    param: lanelet_ids:lanelet网格矩阵。每个位置为对应的lanelet_id.
    param: lanlet_network. common-road lanelet_network
    param: T: 仿真步长。乘以0.1即为时间。
    return: grid_ego_matrix: 与lanelet_ids维度相同的矩阵。自车在的位置为1，其余为0
    return: ego_s: 自车位置相对于左上角lanelet的中心线的frenet坐标系的沿中心线方向距离
    return: obstacle_states: 他车状态矩阵。[m(车道数), 2]矩阵. 表示每条道路上车的状态，每条道路上最多一辆车。如果该车道上没有车，则是[-1,-1]
        如果有车，则是[s, v]。s表示frenet坐标系(s-d)坐标值.v是他车沿车方向速度。

    '''
    grid_ego_matrix = np.zeros(lanelet_id_matrix.shape)
    lanelet_id = lanelet_network.find_lanelet_by_position([ego_pos])[0]
    for lanelet_id_i in lanelet_id:
        index = np.where(lanelet_id_matrix == lanelet_id_i)
        grid_ego_matrix[index] =1
    # 取lanelet_ids中左上角的中心线为frenet坐标系参考线，返回自车当前的s
    lanelet00 = lanelet_network.find_lanelet_by_id(lanelet_id_matrix[0, 0])
    center_vertices = lanelet00.center_vertices
    cv, _,s_cv = detail_cv(center_vertices)
    # 找最近点
    ego_s = distance_lanelet(cv, s_cv, center_vertices[0,:],ego_pos)

    shape = lanelet_id_matrix.shape

    obstacles = scenario.obstacles
    n_obstacles = len(obstacles)
    obstacle_states = -1*np.ones((n_obstacles,3))

    for i_ob in range(n_obstacles):
        obstacle = obstacles[i_ob]
        state = obstacle.state_at_time(T)
        if state is None:
            print('obstacle dead.')
            continue
        lanelet_id = lanelet_network.find_lanelet_by_position([state.position])[0]
        lane_lat_n_ = np.where(lanelet_id_matrix==lanelet_id)
        if len(lane_lat_n_)==0:
            print('info: obstacle ', obstacle.obstacle_id, ' not in lanelet_id_matix ')
            continue
        else:
            lane_lat_n = lane_lat_n_[0]
            obstacle_states[i_ob, 0] = lane_lat_n
        s = distance_lanelet(cv, s_cv, center_vertices[0,:],state.position)
        obstacle_states[i_ob, 1] = s
        obstacle_states[i_ob, 2] = state.velocity


    obstacle_states_in = obstacle_states[obstacle_states[:,0] != -1, :]


    return grid_ego_matrix, ego_s, obstacle_states_in


# def get_ego_init_state(init_state, grid_ego_matrix):
#     '''输入planning problem的state，以及提取的lanelet矩阵，输出相应
#     Params:
#         init_state: commonroad  planning problem init state
#         grid_ego_matrix: ego_pos2tree 输出的 grid_ego_matrix
#     Returns:
#         states_ego: (3,) [车道，位置，速度]。最左侧车道为0

#     '''
#     state = []
    
#     return state

def get_map_info(planning_problem, grid, lanelet00_cv_info, lanelet_id_matrix):
    map  = []
    n_lane = grid.shape[0]
    map.append(n_lane)
    goal_pos  = planning_problem.goal.state_list[0].position.center

    lanelet_id_goal = lanelet_network.find_lanelet_by_position([goal_pos])[0]
    lane_pos_ = np.where(lanelet_id_matrix==lanelet_id_goal)[0]
    if len(lane_pos_)==0:
        print('error cannot found goal lanelet position')
    else:
        lane_pos = lane_pos_[0]
    map.append(lane_pos)
    cv, _,  s_cv = lanelet00_cv_info
    goal_s = distance_lanelet(cv, s_cv, [cv[0][0], cv[1][0]],goal_pos)
    map.append(goal_s)


    return map


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
    # 提取自车初始状态
    # scenario = edit_scenario4test(scenario, ego_pos_init)

    # ---------------可视化修改后的场景 ------------------------------
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
    # ---------------可视化 end ------------------------------

    # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
    lanelet_network  = scenario.lanelet_network
    lanelet_id_matrix  = lanelet_network2grid(lanelet_network)
    print('lanelet_id_matrix: ', lanelet_id_matrix)

    # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
    T = 50           # 5*0.1=0.5.返回0.5s时周车的状态。注意下面函数返回的自车状态仍然是初始时刻的。
    grid, ego_d, obstacles =ego_pos2tree(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, T)
    # print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)

    v_ego = planning_problem.initial_state.velocity

    lanelet00_cv_info = detail_cv(lanelet_network.find_lanelet_by_id(lanelet_id_matrix[0, 0]).center_vertices)
    lane_ego_n_array, _ = np.where(grid == 1)

    map = get_map_info(planning_problem, grid, lanelet00_cv_info, lanelet_id_matrix)
    if len(lane_ego_n_array)>0:
        lane_ego_n = lane_ego_n_array[0]
    else:
        print('ego_lane not found. out of lanelet')
        lane_ego_n = -1
    state = [lane_ego_n, ego_d, v_ego]


    # 决策初始时刻目前无法给出，需要串起来后继续[TODO]
    T= 0        

    print('自车初始状态矩阵', state)
    print('地图信息', map)
    print('他车矩阵', obstacles)



