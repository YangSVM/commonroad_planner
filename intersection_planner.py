# -*- coding: UTF-8 -*-
from typing import Iterable
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt

from CR_tools.utility import distance_lanelet
from conf_lanelet_checker import conf_lanelet_checker, potential_conf_lanelet_checkerv2
from detail_central_vertices import detail_cv
from route_planner import route_planner
from Lattice_CRv3 import Lattice_CRv3

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
from commonroad.visualization.mp_renderer import MPRenderer

'''
方法概述：
    - 寻找地图信息。已知自车前进路线，寻找可能有交叉的lanelet；
    - 将交叉的lanelet根据交叉点的远近进行排序
    - 寻找交叉lanelet上，在交叉点前的车
    - 提取两辆车，计算协作加速度
    - 根据协作加速度控制车辆
缺少要素：
    1. 如何系统的考虑前车的影响？目前场景没有前车: 考虑在前车期望速度。
    2. 全局lanelet规划器。比如路口在行驶的过程中，如何知道此时应该前行还是右转。

'''


class Ipaction():
    def __init__(self):
        self.v_end = 0
        self.a_end = 0
        self.delta_s = None
        self.frenet_cv = []
        self.T = None
        self.ego_state_init = []
        self.lanelet_id_target = None

def get_route_frenet_line(route, lanelet_network):
    ''' 获取route lanelt id的对应参考线
    '''
    # TODO: 是否存在route是多个左右相邻的并排车道的情况。此时不能将lanelet的中心线直接拼接
    cv = []
    for n in range(len(route)):
        if n == 0:
            cv = lanelet_network.find_lanelet_by_id(route[n]).center_vertices
        else:
            cv_temp = lanelet_network.find_lanelet_by_id(route[n]).center_vertices
            cv = np.concatenate((cv, cv_temp), axis=0)
    ref_cv, ref_orientation, ref_s = detail_cv(cv)
    ref_cv = np.array(ref_cv).T
    return ref_cv, ref_orientation, ref_s

def sort_conf_point(ego_pos, dict_lanelet_conf_point, cv, cv_s):
    """ 给冲突点按照离自车的距离 由近到远 排序。
    params:
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    returns:
        sorted_lanelet: conf_points 的下标排序
        i_ego: sorted_lanelet[i_ego]则是自车需要考虑的最近的lanelet
    """
    conf_points = list(dict_lanelet_conf_point.values())
    lanelet_ids = np.array(list(dict_lanelet_conf_point.keys()))
    distance = []
    for conf_point in conf_points:
        distance.append(distance_lanelet(cv, cv_s, ego_pos, conf_point))
    distance = np.array(distance)
    id = np.argsort(distance)

    distance_sorted = distance[id]
    index = np.where(distance_sorted > 0)[0]
    if len(index) == 0:
        i_ego = len(distance)
    else:
        i_ego = index.min()

    sorted_lanelet = lanelet_ids[id]

    return sorted_lanelet, i_ego


def find_reference(s, ref_cv, ref_orientation, ref_cv_len):
    ref_cv, ref_orientation, ref_cv_len = np.array(ref_cv), np.array(ref_orientation), np.array(ref_cv_len)
    id = np.searchsorted(ref_cv_len, s)
    if id >= ref_orientation.shape[0]:
        # print('end of reference line, please stop !')
        id = ref_orientation.shape[0] - 1
    return ref_cv[id, :], ref_orientation[id]


def front_vehicle_info_extraction(scenario, ego_pos, lanelet_route):
    '''lanelet_route第一个是自车车道。route直接往前找，直到找到前车。
    新思路：利用函数`find_lanelet_successors_in_range`寻找后继的lanelet节点。寻找在这些节点
    return:
        front_vehicle: dict. key: pos, vel, distance
    '''
    ln = scenario.lanelet_network
    front_vehicle = {}
    ref_cv, ref_orientation, ref_s = get_route_frenet_line(lanelet_route, ln)
    min_dhw = 500
    s_ego = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], ego_pos)
    lanelet_ids_ego = ln.find_lanelet_by_position([ego_pos])[0]
    # assert lanelet_ids_ego[0] in lanelet_route
    obstacles = scenario.obstacles
    for obs in obstacles:
        if obs.state_at_time(0):
            pos = obs.state_at_time(0).position
            # print(ln.find_lanelet_by_position([pos]))
            if not ln.find_lanelet_by_position([pos]) == [[]]:
                obs_lanelet_id = ln.find_lanelet_by_position([pos])[0][0]
                if obs_lanelet_id not in lanelet_route:
                    continue
            s_obs = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], pos)
            dhw = s_obs - s_ego
            if dhw < 0:
                continue
            if dhw < min_dhw:
                min_dhw = dhw
                front_vehicle['id'] = obs.obstacle_id
                front_vehicle['dhw'] = dhw
                front_vehicle['v'] = obs.state_at_time(0).velocity
                front_vehicle['state'] = obs.state_at_time(0)

    if len(front_vehicle) == 0:
        print('no front vehicle')
        front_vehicle['dhw'] = -1
        front_vehicle['v'] = -1

    return front_vehicle


class IntersectionInfo():
    ''' 提取交叉路口的冲突信息
    '''

    def __init__(self, cl) -> None:
        '''
        params:
            cl: Conf_Lanelet类
        '''
        self.dict_lanelet_conf_point = {}  # 直接冲突lanelet(与自车轨迹存在直接相交的lanelet,必定在路口内) ->冲突点坐标
        for i in range(len(cl.id)):
            self.dict_lanelet_conf_point[cl.id[i]] = cl.conf_point[i]

        self.dict_lanelet_agent = {}  # 场景信息。直接冲突lanelet - > 离冲突点最近的agent
        self.dict_parent_lanelet = {}  # 地图信息。间接冲突lanelet->直接冲突lanelet*列表*。(间接冲突是直接冲突的parent，一个间接可能对应多个直接)
        self.dict_lanelet_potential_agent = {}  # 间接冲突lanelet - > 离冲突点最近的agent。
        self.sorted_lanelet = []  # 直接冲突lanelet按照冲突点位置进行排序。
        self.i_ego = 0  # 自车目前通过了哪个冲突点。0代表在第一个冲突点之前
        self.sorted_conf_agent = []  # 最终结果：List：他车重要度排序
        self.dict_agent_lanelets = {}

    def extend2list(self, lanelet_network):
        '''为了适应接口。暂时修改
        '''
        conf_potential_lanelets = []
        conf_potential_points = []
        ids = self.dict_lanelet_conf_point.keys
        conf_points = self.dict_lanelet_conf_point.values

        for id, conf_point in zip(ids, conf_points):
            conf_lanlet = lanelet_network.find_lanelet_by_id(id)
            id_predecessors = conf_lanlet.predecessor
            # 排除没有父节点的情况
            if id_predecessors is not None:
                # 多个父节点
                for id_predecessor in id_predecessors:
                    conf_potential_lanelets.append(id_predecessor)
                    conf_potential_points.append(conf_point)
        return conf_potential_lanelets, conf_potential_points


class IntersectionPlanner():
    ''' 交叉路口规划器。
    过程说明：

    '''

    def __init__(self, scenario, route, ego_vehicle, lanelet_state) -> None:
        self.scenario = scenario
        self.ego_state = ego_vehicle.current_state  # 自车状态
        # self.goal = planning_problem.goal
        self.route = route
        self.ego_vehicle = ego_vehicle
        self.lanelet_state = lanelet_state

    def planning(self, T):
        '''轨迹规划器。返回轨迹。
        重要过程说明：
            cl_info: 两个属性。id: 直接冲突lanelet的ID list。conf_point：对应的冲突点坐标list。
            iiinfo: IntersectionInfo类。逐步计算相关变量。详见变量定义。
        Returns:
            trajectory: 自车轨迹。
        '''
        T = 0
        scenario = self.scenario
        lanelet_network = scenario.lanelet_network
        DT = scenario.dt
        # if self.ego_state.position[0] > 472200:
        #     print('conflict check!')

        # --------------- 检索地图，检查冲突lanelet和冲突点 ---------------------
        # 搜索结果： cl_info: ;conf_lanelet_potentials
        potential_ego_lanelet_id_list = scenario.lanelet_network.find_lanelet_by_position([self.ego_state.position])[0]
        for idx in potential_ego_lanelet_id_list:
            if idx in self.route:
                lanelet_id_ego = idx
        # route中没有 lanelet_id_ego: 
        # cl_info: 两个属性。id: 直接冲突lanelet的ID list。conf_point：对应的冲突点坐标list。
        cl_info = conf_lanelet_checker(lanelet_network, lanelet_id_ego, self.lanelet_state, self.route)

        iinfo = IntersectionInfo(cl_info)
        iinfo.dict_parent_lanelet = potential_conf_lanelet_checkerv2(lanelet_network, cl_info)

        # ---------------- 运动规划 --------------
        ego_state = self.ego_state

        # 计算车辆前进的参考轨迹。ref_cv： [n, 2]。参考轨迹坐标. 
        ref_cv, ref_orientation, ref_s = get_route_frenet_line(self.route, lanelet_network)

        # 在[T, T+400]的时间进行规划
        time = [x + T for x in range(1)]
        s = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], ego_state.position)  # 计算自车的frenet纵向坐标
        s_list = [s]
        state_list = []
        state_list.append(ego_state)
        for t in time:

            dict_lanelet_agent = self.conf_agent_checker(iinfo.dict_lanelet_conf_point, t)
            # print('直接冲突车辆', dict_lanelet_agent)
            iinfo.dict_lanelet_agent = dict_lanelet_agent

            # 间接冲突车辆
            dict_lanelet_potential_agent = self.potential_conf_agent_checker(iinfo.dict_lanelet_conf_point,
                                                                             iinfo.dict_parent_lanelet, self.route,
                                                                             t)
            # print('间接冲突车辆', dict_lanelet_potential_agent)
            iinfo.dict_lanelet_potential_agent = dict_lanelet_potential_agent

            # 运动规划
            isConfFound = False
            # 冲突点排序
            iinfo.sorted_lanelet, iinfo.i_ego = sort_conf_point(ego_state.position, iinfo.dict_lanelet_conf_point,
                                                                ref_cv, ref_s)

            # 按照冲突点先后顺序进行决策。找车，给冲突车辆排序
            sorted_conf_agent = []
            dict_agent_lanelets = {}
            for i_lanelet in range(iinfo.i_ego, len(iinfo.sorted_lanelet)):
                lanelet_id = iinfo.sorted_lanelet[i_lanelet]
                # 直接冲突
                if lanelet_id in dict_lanelet_agent.keys():
                    sorted_conf_agent.append(iinfo.dict_lanelet_agent[lanelet_id])
                    dict_agent_lanelets[sorted_conf_agent[-1]] = [lanelet_id]
                else:
                    # 查找父节点
                    lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
                    for parent_lanelet_id in lanelet.predecessor:
                        if parent_lanelet_id not in dict_lanelet_potential_agent.keys():
                            # 如果是None, 没有父节点，也会进入该循环
                            continue
                        else:
                            sorted_conf_agent.append(iinfo.dict_lanelet_potential_agent[parent_lanelet_id])
                            if sorted_conf_agent[-1] not in dict_agent_lanelets.keys():
                                dict_agent_lanelets[sorted_conf_agent[-1]] = [parent_lanelet_id, lanelet_id]
                            continue
            iinfo.sorted_conf_agent = sorted_conf_agent
            iinfo.dict_agent_lanelets = dict_agent_lanelets
            # print('车辆重要性排序：', iinfo.sorted_conf_agent)
            # print('对应车辆可能lanelet：', iinfo.dict_agent_lanelets)

            # 目前。根据未来两辆车进行决策。不够两辆车怎么搞？
            n_o = min(len(iinfo.sorted_conf_agent), 2)
            o_ids = []
            a = []
            dis_ego2cp = []
            for i in range(n_o):
                o_ids.append(iinfo.sorted_conf_agent[i])
                lanelet_ids = iinfo.dict_agent_lanelets[o_ids[i]]
                conf_point = iinfo.dict_lanelet_conf_point[lanelet_ids[-1]]
                a4c, dis_ego2cp_tmp = self.compute_acc4cooperate(ego_state, ref_cv, ref_s, conf_point, lanelet_ids,
                                                                 o_ids[i], t)
                a.append(a4c)
                dis_ego2cp.append(dis_ego2cp_tmp)

        # # ① ==== test planner
        #     # 前车信息提取
        #     front_vehicle = front_vehicle_info_extraction(scenario, lanelet_network, ego_state.position, self.route, T)
        #     next_state, s = self.motion_planner_test(a,
        #                                              ego_state, s,
        #                                              [ref_cv, ref_orientation, ref_s], t,
        #                                              [front_vehicle['dhw'], front_vehicle['v']],
        #                                              )
        #
        #     s_list.append(s)
        #     state_list.append(next_state)
        # # generate a ego vehicle for visualization
        # ego_vehicle = self.generate_ego_vehicle(state_list)
        #
        # # self.analysis_intersection(s_list, scenario)
        # return state_list[1]
        # # ① ===== test planner

        # ② ==== lattice planner
            front_vehicle = front_vehicle_info_extraction(scenario, ego_state.position, self.route)
            next_state, is_new_action_needed = self.motion_planner_lattice(a, dis_ego2cp, front_vehicle)
            state_list.append(next_state)
        return state_list[1]
        # ② ==== lattice planner

    def generate_ego_vehicle(self, state_list):
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
                                      obstacle_shape=ego_vehicle_shape, initial_state=self.ego_state,
                                      prediction=ego_vehicle_prediction)
        return ego_vehicle

    def analysis_intersection(self, s_list, scenario):
        '''分析自车运动轨迹，画出相应的s-t图
        params:
            ego_vehicle
        returns:
        '''

        return 0

    def motion_planner_test(self, a, ego_state0, s, ref_info, t, front_vehicle_info=None):
        ''''根据他车协作加速度，规划自己的运动轨迹；
        params:
            a: 协作加速度
            front_vehicle_info: 2维list。表示前车车距，前车车速。

        returns:
            ego_state: 自车下一时刻的状态
        '''
        a_thre = -4  # 非交互式情况，协作加速度阈值(threshold) 设置为0
        if len(a) > 1:
            a1 = a[0]
            a2 = a[1]
        elif len(a) == 1:
            a1 = a[0]
            a2 = a[0]
        else:
            a1 = 100
            a2 = 100

        # test planner
        if front_vehicle_info is None:  # no leading car
            front_vehicle_info = [0, 0]

        DT = self.scenario.dt
        v = ego_state0.velocity
        a_max = 3
        a_front = 3

        if a1 < a_thre or a2 < a_thre:
            # print(' 避让这辆车', a1, a2)
            a_conf = -a_max
        else:
            a_conf = a_max

        # 考虑前车
        dhw = front_vehicle_info[0]
        v_f = front_vehicle_info[1]
        if dhw > 0:  # 有前车
            s_t = 2 + max([0, v * 1.5 - v * (v - v_f) / 2 / (4 * 2) ** 0.5])
            a_front = 4 * (1 - (v / 60 * 3.6) ** 4 - (s_t / dhw) ** 2)

        # 取最小的加速度控制
        a = min([a_conf, a_front])
        print("加速度", a_conf, a_front)
        v_next = v + a * DT
        if v_next < 0:
            v_next = 0
        s += v_next * DT

        ref_cv, ref_orientation, ref_s = ref_info
        position, orientation = find_reference(s, ref_cv, ref_orientation, ref_s)
        tmp_state = State()
        tmp_state.position = position
        tmp_state.velocity = v_next
        tmp_state.orientation = orientation
        tmp_state.time_step = t
        tmp_state.acceleration = a
        # end of test planner
        return tmp_state, s

    def motion_planner_lattice(self, a, dis_ego2cp, front_veh):
        action = Ipaction()
        is_lane_change = False
        # 1. cv
        ln = self.scenario.lanelet_network
        ego_lanelet_list = ln.find_lanelet_by_position([self.ego_state.position])[0]
        ego_lanelet_id = list(set(self.route).intersection(set(ego_lanelet_list)))[0]
        ego_lanelet = ln.find_lanelet_by_id(ego_lanelet_id)
        next_lanelet = []
        next_lanelet_id = None
        action.frenet_cv = ego_lanelet.center_vertices
        if self.route.index(ego_lanelet_id) < len(self.route)-1:
            curret_index = self.route.index(ego_lanelet_id)
            next_lanelet_id = self.route[curret_index+1]
            next_lanelet = ln.find_lanelet_by_id(next_lanelet_id)
        if next_lanelet:

            action.frenet_cv = np.concatenate((action.frenet_cv, next_lanelet.center_vertices), axis=0)

            if ego_lanelet.adj_left:
                if next_lanelet_id == ego_lanelet.adj_left:
                    print('** left lane-change in ip')
                    is_lane_change = True
                    action.frenet_cv = ln.find_lanelet_by_id(ego_lanelet.adj_left).center_vertices
            if ego_lanelet.adj_right:
                if next_lanelet_id == ego_lanelet.adj_right:
                    print('** right lane-change in ip')
                    is_lane_change = True
                    action.frenet_cv = ln.find_lanelet_by_id(ego_lanelet.adj_right).center_vertices

        # 2. initial state
        ego_state_init = [0 for i in range(6)]
        ego_state_init[0] = self.ego_state.position[0]  # x
        ego_state_init[1] = self.ego_state.position[1]  # y
        ego_state_init[2] = self.ego_state.velocity  # velocity
        ego_state_init[3] = self.ego_state.acceleration  # accleration
        ego_state_init[4] = self.ego_state.orientation  # orientation.
        action.ego_state_init = ego_state_init

        v_end_limit = max(self.ego_state.velocity, 100 / 3.6)
        v_end_conf = v_end_limit  # deal with potential lane-crossing conflicts
        delta_s_conf = 200
        v_end_cf = v_end_limit  # deal with car-following
        delta_s_cf = 200

        # 3. planning distance and end velocity
        # considering lane-crossing conflicts
        a_thre = -2  # 非交互式情况，协作加速度阈值(threshold) 设置为0
        a1, a2 = 100, 100
        ttc = 100
        if len(a) > 1:
            a1 = a[0]
            a2 = a[1]
        elif len(a) == 1:
            a1 = a[0]
            a2 = a[0]
        if a1 < a_thre or a2 < a_thre:  # 避让
            v_end_conf = 10
            if a1 <= a2:
                delta_s_conf = dis_ego2cp[0] - 5
            elif a1 > a2:
                delta_s_conf = dis_ego2cp[1] - 5
        # considering car-following
        if not front_veh['v'] == -1:  # leading car exists
            dhw = front_veh['dhw']
            v_f = front_veh['v']
            ttc = dhw / (ego_state_init[2] - v_f)
            delta_s_cf = dhw
            v_end_cf = v_f

        # print('v_end_conf', v_end_conf)
        # print('v_end_cf', v_end_cf)
        # print('delta_s_conf', delta_s_conf)
        # print('delta_s_cf', delta_s_cf)
        action.delta_s = min(delta_s_conf, delta_s_cf)
        if action.delta_s == delta_s_conf:
            action.v_end = v_end_conf
        elif action.delta_s == delta_s_cf:
            action.v_end = v_end_cf

        # 4. planning horizon
        action.T = action.delta_s / (self.ego_state.velocity + action.v_end) * 2

        # 5. final acceleration
        action.a_end = 0

        # lane-change exception
        if is_lane_change:  # 换道的时候规划时域不能太短，固定为5s
            action.v_end = action.ego_state_init[2]
            if 0 < ttc < 4:
                action.v_end = 0
            action.T = 5
            action.delta_s = action.T * (action.v_end + action.ego_state_init[2]) / 2

        # cut action
        # if not is_lane_change:
        #     action.delta_s = action.delta_s / 4
        #     action.T = action.T / 4
        #     action.v_end = action.ego_state_init[2] + (action.v_end - action.ego_state_init[2]) / 4

        print('dis_ego2cp', dis_ego2cp)
        print('T', action.T)
        print('delta_s', action.delta_s)
        print('v_end', action.v_end)
        print('cv: from', action.frenet_cv[0, :], 'to ', action.frenet_cv[-1, :])

        # lattice planning
        lattice_planner = Lattice_CRv3(self.scenario, self.ego_vehicle)
        next_state, is_new_action_needed = lattice_planner.planner(action)
        return next_state, is_new_action_needed

    def conf_agent_checker(self, dict_lanelet_conf_points, T):
        """  找直接冲突点 conf_lanelets中最靠近冲突点的车，为冲突车辆
        params:
            dict_lanelet_conf_points: 字典。直接冲突点的lanelet_id->冲突点位置
            T: 仿真时间步长
        returns:
            [!!!若该lanelet上没有障碍物，则没有这个lanelet的key。]
            字典dict_lanelet_agent: lanelet-> obstacle_id。可以通过scenario.obstacle_by_id(obstacle_id)获得该障碍物。
           [option] 非必要字典dict_lanelet_d: lanelet - > distance。障碍物到达冲突点的距离。负数说明过了冲突点一定距离
        """
        scenario = self.scenario
        lanelet_network = scenario.lanelet_network
        conf_lanelet_ids = list(dict_lanelet_conf_points.keys())  # 所有冲突lanelet列表

        dict_lanelet_agent = {}  # 字典。key: lanelet, obs_id ;
        dict_lanelet_d = {}  # 字典。key: lanelet, value: distacne .到冲突点的路程

        n_obs = len(scenario.obstacles)
        # 暴力排查场景中的所有车
        for i in range(n_obs):
            state = scenario.obstacles[i].state_at_time(0)  # zxc:scenario是实时的，所有T都改成0
            # 当前时刻这辆车可能没有
            if state is None:
                continue
            pos = scenario.obstacles[i].state_at_time(0).position
            lanelet_ids = lanelet_network.find_lanelet_by_position([pos])[0]
            # 可能在多条车道上，现在每个都做检查
            for lanelet_id in lanelet_ids:
                # 不能仅用位置判断车道。车的朝向也需要考虑?暂不考虑朝向。因为这样写不美。可能在十字路口倒车等
                lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
                # 用自带的函数，检查他车是否在该lanelet上
                res = lanelet.get_obstacles([scenario.obstacles[i]], 0)
                if scenario.obstacles[i] not in res:
                    continue

                # 如果该车在 冲突lanelet上
                if lanelet_id in conf_lanelet_ids:
                    lanelet_center_line = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

                    # 插值函数
                    lanelet_center_line, _, lanelet_center_line_s = detail_cv(lanelet_center_line)

                    conf_point = dict_lanelet_conf_points[lanelet_id]
                    d_obs2conf_point = distance_lanelet(lanelet_center_line, lanelet_center_line_s, pos, conf_point)

                    # 车辆已经通过冲突点，跳过循环
                    # 可能有问题...在冲突点过了一点点的车怎么搞？
                    if d_obs2conf_point < -2 - scenario.obstacles[i].obstacle_shape.length / 2:
                        # 如果超过冲突点一定距离。不考虑该车
                        break
                    if lanelet_id not in dict_lanelet_d:
                        # 该lanelet上出现的第一辆车
                        dict_lanelet_d[lanelet_id] = d_obs2conf_point
                        dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id
                    else:
                        if d_obs2conf_point < dict_lanelet_d[lanelet_id]:
                            dict_lanelet_d[lanelet_id] = d_obs2conf_point
                            dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id

        return dict_lanelet_agent

    def potential_conf_agent_checker(self, dict_lanelet_conf_point, dict_parent_lanelet, ego_lanelets, T):
        '''找间接冲突lanelet.
        params:
            dict_lanelet_conf_point: intersectioninfo类成员。
            dict_parent_lanelet: 间接冲突lanelet->子节点列表。
            T:
        returns:
            dict_lanelet_potential_agent: 间接冲突lanelet->冲突智能体列表。
        '''
        # 即使一辆车多个意图可能相撞，但是只用取一个值就行。任一个冲突点都是靠近终点。

        dict_parent_conf_point = {}  # 可能冲突lanelet -> 随意一个冲突点；因为越靠近终点的就是最需要的车辆。
        for parent, kids in dict_parent_lanelet.items():
            for kid in kids:
                if kid in dict_lanelet_conf_point.keys():
                    dict_parent_conf_point[parent] = dict_lanelet_conf_point[kid]

        # 删除前车影响：
        for ego_lanelet in ego_lanelets:
            if ego_lanelet in dict_parent_conf_point.keys():
                dict_parent_conf_point.pop(ego_lanelet)

        dict_lanelet_potential_agent = self.conf_agent_checker(dict_parent_conf_point, T)

        return dict_lanelet_potential_agent

    def compute_acc4cooperate(self, ego_state, ref_cv, ref_s, conf_point, conf_lanelet_ids, obstacle_id, T):
        '''计算单辆车的协作加速度。用于之后的运动规划。协作加速度为，自车匀速到达冲突点，他车同时到达该点需要的加速度
        params:
            ego_state: common-road state。起码包含属性position, v,
            ref_cv, ref_s: 自车参考轨迹中心线，累计距离。
            conf_points 冲突点
            obstacle_id: 
            conf_lanelet_ids: 他车到达冲突点的lanelet列表。间接冲突车辆可能会经过多个lanelet才能到达冲突点
            T: 仿真步长
        returns:
            a    # 协作的加速度
        '''
        scenario = self.scenario
        pos = ego_state.position
        v = ego_state.velocity

        t4ego2pass = []
        if v == 0:
            v = v + 1
        d_ego2cp = distance_lanelet(ref_cv, ref_s, pos, conf_point)
        t4ego2pass.append(d_ego2cp / v)
        t4ego2pass = np.array(t4ego2pass)
        t_thre = 0
        t = t4ego2pass + t_thre

        conf_agent = scenario.obstacle_by_id(obstacle_id)
        state = conf_agent.state_at_time(0)
        p, v = state.position, state.velocity

        if np.linalg.norm(p - pos) < 5:
            print('warning: too close!')

        conf_cvs = []
        if not isinstance(conf_lanelet_ids, Iterable):
            conf_lanelet = scenario.lanelet_network.find_lanelet_by_id(conf_lanelet_ids)
            conf_cvs = conf_lanelet.center_vertices
        else:
            for conf_lanelet_id in conf_lanelet_ids:
                conf_lanelet = scenario.lanelet_network.find_lanelet_by_id(conf_lanelet_id)
                conf_cv = conf_lanelet.center_vertices
                conf_cvs.append(conf_cv)
            conf_cvs = np.concatenate(conf_cvs, axis=0)

        conf_cvs, _, conf_s = detail_cv(conf_cvs)

        s = distance_lanelet(conf_cvs, conf_s, p, conf_point)
        a = 2 * (s - v * t) / (t ** 2)
        return a, d_ego2cp
