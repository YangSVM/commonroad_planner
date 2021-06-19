# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt

from conf_lanelet_checker import conf_lanelet_checker, potential_conf_lanelet_checker
from detail_central_vertices import detail_cv

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
from commonroad.visualization.mp_renderer import MPRenderer


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


def sort_conf_point(ego_pos, conf_points, cv, cv_s):
    ''' 给冲突点按照离自车的距离 由近到远 排序。如果自车越过该冲突点，则删除
    params: 
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    returns:
        id: conf_points 的下标排序
    '''
    distance = []
    for conf_point in conf_points:
        distance.append(distance_lanelet(cv, cv_s, ego_pos, conf_point))
    distance = np.array(distance)
    id = np.argsort(distance)
    id_reverse = id[::-1]
    distance_sorted = distance[id_reverse]
    id_reverse = id_reverse[distance_sorted>0]

    return id_reverse


def find_reference(s, ref_cv, ref_orientation,  ref_cv_len):
    ref_cv, ref_orientation,  ref_cv_len = np.array(ref_cv), np.array(ref_orientation), np.array(ref_cv_len)
    id = np.searchsorted(ref_cv_len, s)

    return ref_cv[:, id], ref_orientation[id]

class IntersectionPlanner():
    def __init__(self, scenario, state_init, goal) -> None:
        self.scenario = scenario
        self.state_init = state_init
        self.goal = goal


    def planner(self):
        '''轨迹规划器。返回轨迹
        Returns:
            trajectory: 自车轨迹。
        '''
        scenario = self.scenario
        lanelet_network = scenario.lanelet_network
        DT = scenario.dt
        
        # --------------- 检索地图，检查冲突lanelet和冲突点 ---------------------
        # 搜索结果： cl_info: ;conf_lanelet_potentials
        incoming_lanelet_id_sub = 50195
        direction_sub = 1
        # cl_info: 两个属性。id: 直接冲突lanelet的ID list。conf_point：对应的冲突点坐标list。
        cl_info = conf_lanelet_checker(lanelet_network, incoming_lanelet_id_sub, direction_sub)
        
        # 潜在冲突lanelet
        cl_potential_info = potential_conf_lanelet_checker(lanelet_network, cl_info)

        
        # ---------------- 运动规划 --------------
        ego_state = self.state_init
        
        # 计算车辆前进的参考轨迹
        cv1 = lanelet_network.find_lanelet_by_id(incoming_lanelet_id_sub).center_vertices
        cv2 = lanelet_network.find_lanelet_by_id(50209).center_vertices
        cv =  np.concatenate((cv1, cv2), axis=0)
        ref_cv, ref_orientation,  ref_s = detail_cv(cv)

        T= [x for x in range(100)]
        # T = [70]
        s = distance_lanelet(ref_cv, ref_s, cv1[0,:],ego_state.position) # 已经有参考轨迹，直接计算行驶路程

        a_max = 3
        state_list = []
        state_list.append(ego_state)
        for  i in T:
            dict_lanelet_agent= self.conf_agent_checker(cl_info.id, cl_info.conf_point, i)
            print('直接冲突车辆',dict_lanelet_agent)
            
            # 间接冲突车辆
            dict_lanelet_agent_potential = self.conf_agent_checker(cl_potential_info.id, cl_potential_info.conf_point, i)
            print('间接冲突车辆',dict_lanelet_agent_potential)

            # 冲突点排序. 所有的冲突点都在cl_info.
            
            # 运动规划
            isConfFound = False
            a_thre = 0          # 非交互式情况，协作加速度阈值(threshold) 设置为0
            seq_conf_point = sort_conf_point(ego_state.position, cl_info.conf_point, ref_cv, ref_s)
            for i_conf_point in seq_conf_point:
                lanelet_id = cl_info.id[i_conf_point]
                # 直接冲突
                if lanelet_id in dict_lanelet_agent.keys():
                    a  = self.compute_acc4cooperate(ego_state, ref_cv, ref_s, cl_info.conf_point[i_conf_point],
                     lanelet_id, dict_lanelet_agent[lanelet_id], scenario, i)
                    if a < a_thre:
                        print('直接冲突 - 避让这辆车', a)
                        v0 = ego_state.velocity
                        v = v0 - a_max *DT
                        if v<0:
                            v = 0
                        s += v*DT
                        position, orientation = find_reference(s, ref_cv, ref_orientation,  ref_s)
                        ego_state.position = position
                        ego_state.velocity = v
                        ego_state.orientation = orientation
                        ego_state.time_step = i
                        tmp_state = ego_state
                        state_list.append(tmp_state)
                    else:
                        print('直接冲突 - 加速通过', a)
                        v0 = ego_state.velocity
                        v = v0 + a_max *DT

                        s += v*DT
                        position, orientation = find_reference(s, ref_cv, ref_orientation,  ref_s)
                        ego_state.position = position
                        ego_state.velocity = v
                        ego_state.orientation = orientation
                        ego_state.time_step = i
                        tmp_state = ego_state
                        state_list.append(tmp_state)
                                        
                # 如果没有直接冲突车辆，看间接冲突车辆。
                else:
                    # 查找父节点
                    lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
                    for parent_lanelet_id in lanelet.predecessor:
                        if parent_lanelet_id not in dict_lanelet_agent_potential.keys():
                            continue
                        else:
                            a  = self.compute_acc4cooperate(ego_state, ref_cv, ref_s, cl_info.conf_point[i_conf_point], 
                            parent_lanelet_id, dict_lanelet_agent_potential[parent_lanelet_id], scenario, i,lanelet_id)
                            if a < a_thre:
                                print('间接冲突 - 避让这辆车')
                                v0 = ego_state.velocity
                                v = v0 - a_max *DT
                                if v<0:
                                    v = 0
                                s += v*DT
                                position, orientation = find_reference(s, ref_cv, ref_orientation,  ref_s)
                                ego_state.position = position
                                ego_state.velocity = v
                                ego_state.orientation = orientation
                                ego_state.time_step = i
                                tmp_state = ego_state
                                state_list.append(tmp_state)
                            else:
                                print('间接冲突 - 加速通过')
                                v0 = ego_state.velocity
                                v = v0 + a_max *DT

                                s += v*DT
                                position, orientation = find_reference(s, ref_cv, ref_orientation,  ref_s)
                                ego_state.position = position
                                ego_state.velocity = v
                                ego_state.orientation = orientation
                                ego_state.time_step = i
                                tmp_state = ego_state
                                state_list.append(tmp_state)
                            isConfFound = True
                            break
                    if isConfFound:
                        break

        # create the planned trajectory starting at time step 1
        ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])
        # create the prediction using the planned trajectory and the shape of the ego vehicle

        vehicle3 = parameters_vehicle3.parameters_vehicle3()
        ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
        ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                                    shape=ego_vehicle_shape)

        # the ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_type = ObstacleType.CAR
        ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                    obstacle_shape=ego_vehicle_shape, initial_state=self.state_init,
                                    prediction=ego_vehicle_prediction)
        return ego_vehicle


    def motion_planner(self, a):
        ''''根据他车协作加速度，规划自己的运动轨迹；
        params:
            a: 协作加速度
        returns:

        '''

    def conf_agent_checker(self, conf_lanelet_ids, conf_points, T):
        '''  找conf_lanelets中最靠近冲突点的车，为冲突车辆
        params:
            conf_lanelet_ids:  列表。
            conf_points: 
            T: 仿真时间步长
        returns:
            [!!!若该lanelet上没有障碍物，则没有这个lanelet的key。]
            字典dict_lanelet_agent: lanelet-> obstacle_id。可以通过scenario.obstacle_by_id(obstacle_id)获得该障碍物。
           [option] 非必要字典dict_lanelet_d: lanelet - > distance。障碍物到达冲突点的距离。负数说明过了冲突点一定距离
        '''
        scenario = self.scenario
        lanelet_network = scenario.lanelet_network

        dict_lanelet_agent = {}         # 字典。key: lanelet, obs_id ;
        dict_lanelet_d ={}          # 字典。key: lanelet, value: distacne .到冲突点的路程
        dict_lanelet_s = {}         # 保存已有的中心线距离。避免重复计算。
            
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
                if lanelet_id in conf_lanelet_ids:
                    lanelet_center_line = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

                    # 插值函数
                    lanelet_center_line, _,  new_add_len = detail_cv(lanelet_center_line)

                    if lanelet_id not in dict_lanelet_d:
                        dict_lanelet_s[lanelet_id] = new_add_len
                    conf_point = conf_points[conf_lanelet_ids.index(lanelet_id)]
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
            
        return dict_lanelet_agent

    def compute_acc4cooperate(self, ego_state, ref_cv, ref_s, conf_point, conf_lanelet_id,obstacle_id, scenario, T, successor_id = None):
        '''计算单辆车的协作加速度。用于之后的运动规划。协作加速度为，自车匀速到达冲突点，他车同时到达该点需要的加速度
        params:
            ego_state: common-road state。起码包含属性position, v,
            ref_cv, ref_s: 自车参考轨迹中心线，累计距离。
            conf_points 冲突点
            obstacle_id: 
            scenario: commonroad scenario
            T: 仿真步长
        returns:
            a    # 协作的加速度
        '''
        pos = ego_state.position
        v =30/3.6

        
        t4ego2pass = []
        if v ==0:
            v = v+1
        t4ego2pass.append(distance_lanelet(ref_cv, ref_s, pos, conf_point) / v)
        t4ego2pass = np.array(t4ego2pass)
        t_thre = 0.5
        t = t4ego2pass + t_thre

        conf_agent = scenario.obstacle_by_id(obstacle_id)
        state = conf_agent.state_at_time(T)
        p, v = state.position, state.velocity

        conf_lanelet = scenario.lanelet_network.find_lanelet_by_id(conf_lanelet_id)
        conf_cv = conf_lanelet.center_vertices

        if successor_id is not None:
            conf_lanelet_s = scenario.lanelet_network.find_lanelet_by_id(successor_id)
            conf_cv_s =conf_lanelet_s.center_vertices
            conf_cv = np.concatenate((conf_cv, conf_cv_s), axis=0)
        conf_cv, _, conf_s = detail_cv(conf_cv)

        s = distance_lanelet(conf_cv, conf_s, p, conf_point)
        a = 2*(s-v*t)/(t**2)
        return a


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

    state_init = planning_problem.initial_state
    goal  = [0,0]

    ip = IntersectionPlanner(scenario, state_init, goal)
    ego_vehicle = ip.planner()
    # scenario.add_objects()

    # ip.plot()

    # plt.figure(1)

    # plt.clf()
    # draw_parameters = {
    #     'time_begin': i, 
    #     'scenario':
    #     { 'dynamic_obstacle': { 'show_label': True, },
    #         'lanelet_network':{'lanelet':{'show_label': False,  },} ,
    #     },
    # }

    # draw_object(scenario, draw_params=draw_parameters)
    # draw_object(planning_problem_set)
    # plt.gca().set_aspect('equal')
    # plt.pause(0.01)
    # # plt.show()

    # plot the scenario and the ego vehicle for each time step
    plt.figure(1)
    for i in range(0, 40):
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i})
        ego_vehicle.draw(rnd, draw_params={'time_begin': i, 'dynamic_obstacle': {
            'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
                'facecolor': 'r'}}}}}})
        planning_problem_set.draw(rnd)
        rnd.render()
        plt.pause(0.01)
