from posixpath import ismount
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object
import os


from detail_central_vertices import detail_cv
from intersection_planner import distance_lanelet
import numpy as np
import matplotlib.pyplot as plt
from grid_lanelet import lanelet_network2grid
from grid_lanelet import get_obstacle_info
from grid_lanelet import get_map_info
from grid_lanelet import edit_scenario4test
from MCTs_v3pro import NaughtsAndCrossesState
from MCTs_v3pro import mcts
from grid_lanelet import get_frenet_lanelet_axis
from grid_lanelet import generate_len_map,find_target_frenet_axis,extract_speed_limit_from_traffic_sign
from MCTs_v3pro import output


class ActionAddition:
    def __init__(self) :
        self.v_end = -1
        self.a_end = -1
        self.delta_s = -1
        self.lanelet_id_target = -1
        self.T = 6
        self.ego_state_init = []
        self.frenet_cv = None
    def find_lanelet_id_target(self, s_goal, lanelet_id_matrix, n_target, ln: LaneletNetwork):
        ''' 寻找目标位置的lanelet ID。根据goad的车道，s坐标反推。
        Params:
            s_goal: 目标的s轴坐标
            lanelet_id_matrix: lanelet_id 矩阵
            n_target: goal的目标车道
        Return:
            lanelet_id_target: 目标的lanelet编号
        '''
        # 仅需要判断在哪个 len 区间中即可。
        len_map = generate_len_map(ln, lanelet_id_matrix, isContinous=False)
        n_s_lanelet = len(lanelet_id_matrix[n_target,:])
        index_target = -1
        for i in range(n_s_lanelet):
            len_lanelet = len_map[n_target][i]
            if  len_lanelet[0] < s_goal and s_goal<len_lanelet[1]:
                index_target = i
                break
        assert index_target!=-1
        lanelet_id_target = lanelet_id_matrix[n_target, index_target]
        assert lanelet_id_target!=-1
        return lanelet_id_target


class MCTs_CRv3():
    def __init__(self, scenario: Scenario, planning_problem: PlanningProblem, lanelet_route, ego_vehicle):
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.lanelet_route = lanelet_route
        self.ego_vehicle = ego_vehicle

    def cut_lanelet_route(self, ego_state):
        '''cut the straghtway of the scenario. 
        return:
            start_route_id. id of self.route
            end_lanelet_id.  id of self.route. 位于路口/汇入口 的进入lanelet。
        '''
        ln = self.scenario.lanelet_network
        # find current lanelet
        lanelet_id_ego = ln.find_lanelet_by_position([ego_state.position])[0][0]
        lanelet_ego = ln.find_lanelet_by_id(lanelet_id_ego)
    
        #1. find the nearest lanelet in self.lanelet_route:
        # 如果自车车道在lanelet route上

        # 1.1 找同向邻车道
        lanelets_id_adj = []           # 与lanelet_ego左右相邻的车道的ID
        
        # 从自车道一直往左遍历相邻车道
        tmp_lanelet = lanelet_ego
        while tmp_lanelet.adj_left is not None:
            if tmp_lanelet.adj_left_same_direction:     # 行驶方向相同
                tmp_lanelet_id = tmp_lanelet.adj_left
                lanelets_id_adj.append(tmp_lanelet_id)
                tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)       

        # 从自车道一直往右遍历相邻车道
        tmp_lanelet = lanelet_ego
        while tmp_lanelet.adj_right is not None:
            if tmp_lanelet.adj_right_same_direction:    # 行驶方向相同
                tmp_lanelet_id = tmp_lanelet.adj_right
                lanelets_id_adj.append(tmp_lanelet_id)
                tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)       
        
        # 2. 找自车最相近的lanelet_id
        start_lanelet_id = None

        if lanelet_id_ego in self.lanelet_route:
            start_lanelet_id = lanelet_id_ego
        else:
            print('warning. maybe wrong. mcts_cr cannot cut the lanelet right')
            for lanelet_id_adj in lanelets_id_adj:
                if lanelet_id_adj in self.lanelet_route:
                    start_lanelet_id = lanelet_id_adj 
                            
        # cannot cut the lanelet route
        assert start_lanelet_id is not None
        start_route_id = self.lanelet_route.index(start_lanelet_id)

        # 3. search for the end lanetlet_id
        end_lanelet_id = None
        is_meet_intersection = False
        # 3.1 按顺序check the route lanelet
        for i_route in range(start_route_id, len(self.lanelet_route)):
            tmp_lanelet_id_route = self.lanelet_route[i_route]
            # check if it is in incoming
            for idx_inter, intersection in enumerate(ln.intersections):
                incomings = intersection.incomings

                for idx_inc, incoming in enumerate(incomings):
                    incoming_lanelets = list(incoming.incoming_lanelets)            # 进入路口的lanelet
                    in_intersection_lanelets = list(incoming.successors_straight)   # 出路口的lanelet
                    # 如果self.route中有lanelet在路口上, 赋值为end_lanelet_id
                    if tmp_lanelet_id_route in incoming_lanelets or tmp_lanelet_id_route in in_intersection_lanelets:
                        end_lanelet_id = tmp_lanelet_id_route
                        is_meet_intersection = True
                        break
                if is_meet_intersection:
                    break
            if is_meet_intersection:
                break
        
        if not is_meet_intersection:
            end_lanelet_id = self.lanelet_route[-1]
        end_route_id = self.lanelet_route.index(end_lanelet_id)

        # if len(self.lanelet_route) == end_route_id:
        #     is_goal  = True
        # else:
        #     is_goal = False
        return start_route_id, end_route_id, is_meet_intersection

    def planner(self, T):
        T = 0
        planning_problem  = self.planning_problem
        scenario =self.scenario
        ego_vehicle = self.ego_vehicle
        start_route_id, end_route_id, is_meet_intersection = self.cut_lanelet_route(ego_vehicle.current_state)

        # 直接判断是否在终点lanelet 是否是 planning problem的goal
        is_goal = self.lanelet_route[end_route_id] in planning_problem.goal.lanelets_of_goal_position[0]

        ego_pos = self.ego_vehicle.current_state.position
        # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
        ln  = scenario.lanelet_network
        lanelet_id_matrix  = lanelet_network2grid(ln, self.lanelet_route[start_route_id:end_route_id+1])
        
        # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离

        # 自车在第几个车道，自车s坐标，障碍物矩阵
        lane_ego_n_array, s_ego, obstacles =get_obstacle_info(ego_pos, lanelet_id_matrix, ln, scenario, T)
        # print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)


        lanelet_ids_frenet_axis = get_frenet_lanelet_axis(lanelet_id_matrix)
        
        # 地图信息: [总车道数, 目标车道编号, 目标位置, 场景限速(m/s)]
        lanelet_id_goal = self.lanelet_route[end_route_id]
        map = get_map_info(is_goal,  lanelet_id_goal, lanelet_ids_frenet_axis, lanelet_id_matrix, ln, planning_problem,is_interactive=True)
        speed_limit = extract_speed_limit_from_traffic_sign(ln)
        map.append(speed_limit)

        if len(lane_ego_n_array)>0:
            lane_ego_n = lane_ego_n_array[0]
        else:
            print('ego_lane not found. out of lanelet')
            lane_ego_n = -1
        v_ego = ego_vehicle.current_state.velocity
        state = [lane_ego_n, s_ego, v_ego]

        # 获取可行地图信息
        map_info = generate_len_map(ln, lanelet_id_matrix)
        # print for debug
        print('lanelet_id_matrix: ', lanelet_id_matrix)
        print('自车初始状态矩阵: [车道，位置，速度]\n', state)
        print('地图信息: [总车道数, 目标车道编号, 目标位置, 场景限速(m/s)]\n', map)
        print('他车矩阵：[[所在车道编号，位置，速度]...]\n', obstacles)
        print('可用道路信息列表：[[该车道可用起点, 可用路段终点]...]\n', map_info)

 
        initialState = NaughtsAndCrossesState(state, map, obstacles, map_info)
        searcher = mcts(iterationLimit=5000)  # 改变循环次数或者时间
        action = searcher.search(initialState=initialState)  # 一整个类都是其状态
        out = output(state, action.act, speed_limit)
        print('out: ',out)  # 包括三个信息：[车道，纵向距离的增量，纵向车速]
        # print(action.act)
        
        # Motion planner 其他所需信息
        ego_state_init = [0 for i in range(6)]
        ego_state_init[0] = self.ego_vehicle.current_state.position[0]      # x
        ego_state_init[1] = self.ego_vehicle.current_state.position[1]      # y
        ego_state_init[2] = self.ego_vehicle.current_state.velocity             # velocity
        ego_state_init[3] = self.ego_vehicle.current_state.acceleration      # accleration
        ego_state_init[4] = self.ego_vehicle.current_state.orientation      # orientation. 

        # 曹磊使用
        # 
        action_addition = ActionAddition()
        action_addition.delta_s = out[1]
        action_addition.v_end = out[2]
        action_addition.ego_state_init = ego_state_init
        # action_output.lanelet_id_target = 
        s_goal = action_addition.delta_s + s_ego
        lanelet_id_target = action_addition.find_lanelet_id_target(s_goal, lanelet_id_matrix, out[0], ln)
        action_addition.lanelet_id_target = lanelet_id_target
        frenet_cv = find_target_frenet_axis(lanelet_id_matrix, lanelet_id_target, ln)
        action_addition.frenet_cv = frenet_cv

        # print('目标车道lanelet_id :\n', lanelet_id_target)
        # print('目标车道中心线数组维度大小：\n', frenet_cv.shape)
        # print('T', action_addition.T)
        # print('delta_s', action_addition.delta_s)
        # print('v_end', action_addition.v_end)


        return action.act, action_addition
