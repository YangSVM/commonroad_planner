# -*- coding: UTF-8 -*-
from logging import NOTSET
from typing import NewType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt

from conf_lanelet_checker import conf_lanelet_checker
from detail_central_vertices import detail_cv



# 计算 中心轨迹 的累积距离
def calc_distance_center_line(center_line):
    ds = np.linalg.norm(center_line[1:,] - center_line[:-1])
    s = np.array([0,  np.cumsum(ds)])
    return s


# 计算沿着道路中心线的路程. p2 - p1（正数说明p2在道路后方）
# 直线的时候，保证是直线距离；曲线的时候，近似正确
def distance_lanelet(center_line, s, p1, p2):
    '''
    Args: 
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    Return:

    '''
    # 规范化格式。必须是numpy数组。并且m*2维，m是点的数量
    if type(center_line) is not np.ndarray:
        center_line = np.array(center_line)
    if center_line.shape[1] !=2:
        center_line = center_line.T
    if center_line.shape[0] == 2:
        print('distance_lanelet warning! may wrong size of center line input. check the input style ')

    d1 = np.linalg.norm(center_line - p1, axis=1)
    i1 = np.argmin(d1)
    d2 = np.linalg.norm(center_line - p2, axis=1)
    i2 = np.argmin(d2)
    
    return s[i2] - s[i1]


def interp_zxc(lanelet_center_line):
    return lanelet_center_line


# 查找道路中的冲突车辆
def conf_agent_checker(scenario, conf_lanelets, conf_points, T):
    # T ：仿真步长
    # 找conf_lanelets中最靠近冲突点的车，为冲突车辆
    # 返回字典dict_lanelet_agent: lanelet-> obstacle_id。可以通过scenario.obstacle_by_id(obstacle_id)获得该障碍物
    # 返回字典dict_lanelet_d: lanelet - > distance。到达冲突点的距离
    lanelet_network = scenario.lanelet_network

    # lanelet = lanelet_network.lanelets[0]

    dict_lanelet_agent = {}         # 字典。key: lanelet, obs_id ;
    dict_lanelet_d ={}          # 字典。key: lanelet, value: distacne .到冲突点的路程
    dict_lanelet_s = {}         # 保存已有的中心线距离。避免重复计算。
    
    # 使用get_obstacles函数，判断lanelet上的车
    # for conf_lanelet_id in conf_lanelets:
    #     conf_lanelet = lanelet_network.find_lanelet_by_id(conf_lanelet_id)
    #     obstacle = conf_lanelet.get_obstacles(scenario.obstacles, T)

        
    n_obs = len(scenario.obstacles)
    # 暴力排查场景中的所有车
    for i in range(n_obs):
        state =  scenario.obstacles[i].state_at_time(T)
        # 当前时刻这辆车可能没有
        if state is None:
            continue
        pos = scenario.obstacles[i].state_at_time(T).position
        lanelet_ids = lanelet_network.find_lanelet_by_position([pos])[0]
        # 可能在多条车道上，现在每个都做检查
        for lanelet_id in lanelet_ids:
            # 修改。使用函数map_obstacle_to_lanelets
            # 不能仅用位置判断车道。车的朝向也需要考虑
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            res = lanelet.get_obstacles([ scenario.obstacles[i]], T)
            if  scenario.obstacles[i] not in res:
                continue
            # 考虑车的朝向和车道的朝向不能超过180度
            # Corner case: 在十字路口倒车
            if lanelet_id in conf_lanelets:
                lanelet_center_line = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

                # 插值函数
                lanelet_center_line, _,  new_add_len = detail_cv(lanelet_center_line)

                if lanelet_id not in dict_lanelet_d:
                    dict_lanelet_s[lanelet_id] = new_add_len
                conf_point = conf_points[conf_lanelets.index(lanelet_id)]
                d_obs2conf_point = distance_lanelet(lanelet_center_line, dict_lanelet_s[lanelet_id], pos, conf_point)
                
                # 车辆已经通过冲突点，跳过循环
                # 可能有问题...在冲突点过了一点点的车怎么搞？
                if d_obs2conf_point< -2 - scenario.obstacles[i].obstacle_shape.length/2:
                    break
                if lanelet_id not in dict_lanelet_d:
                    dict_lanelet_d[lanelet_id] = d_obs2conf_point
                    dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id
                else:           
                    if d_obs2conf_point < dict_lanelet_d[lanelet_id]:
                        dict_lanelet_d[lanelet_id] = d_obs2conf_point
                        dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id
        
    # 找每辆车，让主车先通过，需要多大的
    return dict_lanelet_agent 


# 返回：dict_lanelet_a。    # 字典。冲突的lanelet->需要协作的加速度
def compute_acc4cooperate(ego_state, ego_lanlet, conf_point, dict_lanelet_agent, scenario, T):
    pos = ego_state.pos
    v = ego_state.v

    dict_lanelet_a={}
    t4ego2pass = distance_lanelet(ego_lanlet, pos, conf_point) / v
    t_thre = 0.5
    t = t4ego2pass + t_thre
    for conf_lanelet, conf_agent_id in dict_lanelet_agent.items():
        conf_agent = scenario.obstacle_by_id(conf_agent_id)
        state = conf_agent.state_at_time(T)
        p, v = state.position, state.velocity
        s = distance_lanelet(conf_lanelet, p, conf_point)
        a = 2(s-v*t)/(t**2)
        dict_lanelet_a[conf_lanelet] = a
    return dict_lanelet_a

    
if __name__=='__main__':
    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/hand-crafted')
    # 文件名
    id_scenario = 'ZAM_Tjunction-1_282_T-1'

    path_scenario =os.path.join(path_scenario_download, id_scenario + '.xml')
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # lanelet_network
    lanelet_network = scenario.lanelet_network

    # intersection 信息
    intersections = lanelet_network.intersections
    for intersection in intersections:
        incomings = intersection.incomings
   
    incoming_lanelet_id_sub = 50195
    direction_sub = 1
    # cl_info: 两个属性。id: 直接冲突lanelet的ID list。conf_point：对应的冲突点坐标list。
    cl_info = conf_lanelet_checker(lanelet_network, incoming_lanelet_id_sub, direction_sub)
    
    # 潜在冲突lanelet
    conf_lanelet_potentials = []
    for conf_lanelet_id in cl_info. id:
        conf_lanlet = lanelet_network.find_lanelet_by_id(conf_lanelet_id)
        conf_lanelet_potentials.append(conf_lanlet.predecessor)
    T= [x for x in range(100)]
    # T = [70]
    for  i in T:
        dict_lanelet_agent = conf_agent_checker(scenario, cl_info.id, cl_info.conf_point, i)
        print('直接冲突车辆',dict_lanelet_agent)

        conf_potential_lanelets = []
        conf_potential_points = []
        for id, conf_point in zip(cl_info.id, cl_info.conf_point):
            conf_lanlet = lanelet_network.find_lanelet_by_id(id)
            id_predecessors = conf_lanlet.predecessor
            # 排除没有父节点的情况
            if id_predecessors is not None:
                # 多个父节点
                for id_predecessor in id_predecessors:
                    conf_potential_lanelets.append(id_predecessor)
                    conf_potential_points.append(conf_point)
        
        # 根据冲突点排序
        dict_lanelet_agent_potential = conf_agent_checker(scenario,conf_potential_lanelets, conf_potential_points, i)
        print('间接冲突车辆',dict_lanelet_agent_potential)

        plt.figure(1)

        plt.clf()
        draw_parameters = {
            'time_begin': i, 
            'scenario':
            { 'dynamic_obstacle': { 'show_label': True, },
                'lanelet_network':{'lanelet':{'show_label': False,  },} ,
            },
        }

        draw_object(scenario, draw_params=draw_parameters)
        draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.pause(0.01)
        # plt.show()
