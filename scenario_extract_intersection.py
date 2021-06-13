# -*- coding: UTF-8 -*-
from typing import NewType
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt


#  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
path_scenario_download = os.path.abspath('/home/tiecun/codes/commonroad/commonroad-scenarios/scenarios/hand-crafted')
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
    # 找lanelet中心线

    d1 = np.linalg.norm(center_line - p1, axis=1)
    i1 = np.argmin(d1)
    d2 = np.linalg.norm(center_line - p2, axis=1)
    i2 = np.argmin(d2)
    
    return s[i2] - s[i1]


def interp_zxc(lanelet_center_line):
    return lanelet_center_line


# 查找道路中的冲突车辆
def conf_agent_checker(conf_lanelet, scenario, conf_point, T):
    # T ：仿真步长
    # 找conf_lanelet中最靠近冲突点的车，为冲突车辆
    # 返回字典dict_lanelet_agent: lanelet-> obstacle_id。可以通过scenario.obstacle_by_id(obstacle_id)获得该障碍物
    # 返回字典dict_lanelet_d: lanelet - > distance。到达冲突点的距离
    lanelet_network = scenario.lanelet_network

    # lanelet = lanelet_network.lanelets[0]

    dict_lanelet_agent = {}         # 字典。key: lanelet, obs_id ;
    dict_lanelet_d ={}          # 字典。key: lanelet, value: distacne .到冲突点的路程
    dict_lanelet_s = {}         # 保存已有的中心线距离。避免重复计算。
    n_obs = len(scenario.obstacle)
    # 暴力排查场景中的所有车
    for i in n_obs:
        pos = scenario.obstacles[i].state_at_time(T).position
        lanelet_id = lanelet_network.find_lanelet_by_position([pos])

        # 修改。使用函数map_obstacle_to_lanelets
        if lanelet_id in conf_lanelet:
            lanelet_center_line = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

            # 插值函数
            lanelet_center_line = interp_zxc(lanelet_center_line)

            if lanelet_id not in dict_lanelet_d:
                dict_lanelet_s[lanelet_id] = calc_distance_center_line(lanelet_center_line)

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
    
    # 冲突对象检查
    conf_agent_checker(lanelet_network.lanelets())
    
