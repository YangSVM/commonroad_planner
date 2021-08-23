# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblem
import os

from detail_central_vertices import get_lane_feature
from detail_central_vertices import detail_cv
from CR_tools.utility import distance_lanelet
import numpy as np
import matplotlib.pyplot as plt

def find_adj_lanelets(ln:LaneletNetwork, lanelet_id, include_ego=True):
    '''find the adjecent lanelets with same direction given a lanelet id.
    Params:
        include_ego: 返回值是否包括输入的 lanelet_id 。
    return:
        返回按从左到右顺序的，相邻的lanelet的列表。
    '''
    lanelets_id_adj_left = []           # 与lanelet_ego左右相邻的车道的ID
    lanelets_id_adj_right = []           # 与lanelet_ego左右相邻的车道的ID
    n_left = 0
    n_right = 0
    tmp_lanelet_id = lanelet_id
    tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)
    while tmp_lanelet.adj_left is not None:
        if tmp_lanelet.adj_left_same_direction:
            n_left += 1
            tmp_lanelet_id = tmp_lanelet.adj_left
            lanelets_id_adj_left.append(tmp_lanelet_id)
            tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)       

    tmp_lanelet_id = lanelet_id
    tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)
    while tmp_lanelet.adj_right is not None:
        if tmp_lanelet.adj_right_same_direction:
            n_right += 1
            tmp_lanelet_id = tmp_lanelet.adj_right
            lanelets_id_adj_right.append(tmp_lanelet_id)
            tmp_lanelet = ln.find_lanelet_by_id(tmp_lanelet_id)     
    if include_ego:
        lanelets_id_adj = lanelets_id_adj_left[::-1] + [lanelet_id] + lanelets_id_adj_right
    else:
        lanelets_id_adj =  lanelets_id_adj_left[::-1] + lanelets_id_adj_right
    return lanelets_id_adj, n_left, n_right

def find_target_frenet_axis(lanelet_id_matrix, lanelet_id_target, ln:LaneletNetwork):
    ''' 寻找穿过目标车道frenet s轴。为便于lattice使用，延长至下一个lanelet
    '''
    # 判断在第几条车道
    n_lane = np.where(lanelet_id_matrix == lanelet_id_target)[0]
    assert n_lane.shape[0]>0, 'lanelet_id_target do not in lanelet_id_matrix!'
    lanelets_frenet_axis = lanelet_id_matrix[n_lane[0], :]

    lanelets_frenet_axis_ = []
    for lanelet_id in lanelets_frenet_axis:
        # 如果lanelet_id_matrix矩阵，在第i行存在-1...直接跳过(无法处理)
        if lanelet_id == -1:
            print('error!! 无法生成非法目标车道的frenet轴线 ')
            break
        lanelets_frenet_axis_.append(lanelet_id)
    
    # 增长frenet参考线
    end_lanelet_id = lanelets_frenet_axis_[-1]
    end_lanelet = ln.find_lanelet_by_id(end_lanelet_id)
    # end_lanelet如果有子节点，选用子节点进行延展
    extend_lanelet_id_list = end_lanelet.successor
    if len(extend_lanelet_id_list) > 0:
        extend_lanelet_id = extend_lanelet_id_list[0]
    lanelets_frenet_axis_.append(extend_lanelet_id)

    lanelets_frenet_axis = np.array(lanelets_frenet_axis_)
            
    print('frenet lanelet id axis :', lanelets_frenet_axis)
    cv = []
    for lanelet_id in lanelets_frenet_axis:
        lanelet = ln.find_lanelet_by_id(lanelet_id)
        cv.append(lanelet.center_vertices)
    cv = np.concatenate(cv, axis=0)

    return cv


def lanelet_network2grid(ln : LaneletNetwork, route):
    '''明阳需求。将lanelet_network转成网格地图。
    v2: 
    param: 
        ln: commonroad lanelet_network
        route: lanelet_id list.
    return: 
        grid: m(车道数)*n(纵向lanelet数目)矩阵. 值域 [0,1,-1]。-1不可通行，0可通行，1自车所在lanelet。
        lanelet_id： 对应的车道的lanelet_id
    '''

    end_lanelets, _, _ = find_adj_lanelets(ln, route[-1])

    # 找与route中相邻的lanelet的集合
    adj_route =[]    # adj_route可能存在重复元素。因为route中存在相邻车道，被反复计算

    for route_lanelet in route:
        adj_route_, _, _ = find_adj_lanelets(ln, route_lanelet)
        adj_route += adj_route_         # list相加。直接拼接。

    #找straight_route_id。并求出lanelet_id_matrix的矩阵车道方向。
    #straight_route_id： 从route[0]出发。找一条直线，直到end_lanelets或其相邻车道
    straight_route_id = []      
    current_lanelet_id = route[0]
    current_lanelet  =ln.find_lanelet_by_id(route[0])
    straight_route_id.append(current_lanelet_id)
    # left_n_max。车道max数目。
    _, left_n_max, right_n_max = find_adj_lanelets(ln, route[0])
    while current_lanelet.successor is not None:
        if current_lanelet_id in end_lanelets:
            break
        current_lanelets_id = current_lanelet.successor

        assert isinstance(current_lanelets_id, list) 
        # 如果有多个 successor. 按照原有顺序，选择第一个在lanelet route的相邻车道中的
        is_next = False
        for tmp_lanelet_id in current_lanelets_id:

            if tmp_lanelet_id in adj_route:
                current_lanelet_id = tmp_lanelet_id
                is_next = True
                break
        if not is_next:
            print('error. cannot found the next lanelet')
        assert is_next
        current_lanelet = ln.find_lanelet_by_id(current_lanelet_id)
        straight_route_id.append(current_lanelet_id)

        _, left_n, right_n = find_adj_lanelets(ln, current_lanelet_id)
        if left_n>left_n_max:
            left_n_max = left_n
        if right_n>right_n_max:
            right_n_max = right_n
        
    #根据n_left_max, n_right_max 建立list
    n_lane = left_n_max+ right_n_max +1
    lanelet_id_matrix = -1 * np.ones([n_lane, len(straight_route_id)], dtype=int) 
    for i_route, lanelet_id in enumerate(straight_route_id):
        adj_lanelets, n_l, n_r = find_adj_lanelets(ln, lanelet_id)
        lanelet_id_matrix[left_n_max- n_l : left_n_max + n_r +1, i_route] = adj_lanelets
        
    return lanelet_id_matrix


def get_detail_cv_of_lanelets(lanelet_ids_frenet_axis, ln: LaneletNetwork):
    ''' 输入 lanelet id 的列表，输出detail cv以及相关的cv参数
    '''
    cv = []
    for lanelet in lanelet_ids_frenet_axis:
        cv.append(ln.find_lanelet_by_id(lanelet).center_vertices)
    
    cv = np.concatenate(cv, axis=0)
    cv, new_direct, s_cv = detail_cv(cv)
    cv = np.array(cv).T

    return cv, new_direct, s_cv,


def state_cr2state_mcts(lanelet_ids_frenet_axis, lanelet_id_matrix, ln:LaneletNetwork, state_cr):
    ''' 将 cr中的state(包含 position, velocity属性)，转化为MCTs需要的状态[车道，位置，速度]
    Params:
        lanelet_ids_frenet_axis: lanelet id列表。代表选择的frenet坐标系
        lanelet_id_matrix: lanelet id矩阵。代表直道场景的lanelet拓扑信息
        ln: cr scenario.LaneletNetwork
        state_cr: cr state
    Returns:
        状态列表: [车道，位置，速度]
    '''
    pos = state_cr.position
    cv, _, s_cv = get_detail_cv_of_lanelets(lanelet_ids_frenet_axis, ln)

    # 寻找为第几车道
    lane_index = -1         # 初始值 -1
    lanelets_id = ln.find_lanelet_by_position([pos])[0]          # lanelets_id 该位置可能在诸多lanelet上
    for lanelet_id_i in lanelets_id:
        i, j = np.where(lanelet_id_matrix == lanelet_id_i)
        if len(i) > 0:          # 如果在某一行。直接赋值
            lane_index = i[0]
    if lane_index == -1:
        # not in the lanelet id matrix. return.
        return [-1, -1, -1]

    # 车位置 与frenet坐标系原点的s方向距离。即车的s坐标
    s = distance_lanelet(cv, s_cv, cv[0, :], pos)
    v = state_cr.velocity

    return [lane_index, s, v]


def get_obstacle_info(lanelet_ids_frenet_axis,  lanelet_id_matrix, ln: LaneletNetwork, obstacles, T):
    '''通过自车位置，标记对应的1。并且返回自车位置相对于左上角lanelet的中心线的frenet坐标系的距离。
    Params: 
        lanelet_ids_frenet_axis:    frenet s轴lanelet id列表
        lanelet_id_matrix: lanelet网格矩阵。每个位置为对应的lanelet_id.
        ln: cr scenario.lanelet_network
        obstacles: cr scenario.obstacles
        T: 仿真步长。乘以0.1即为时间。
        
    Return: 
        obstacle_states: 他车状态矩阵。Nx3矩阵（N为总车数）.表示每辆车的状态。[[所在车道编号，位置，速度]...]

    '''

    n_obstacles = len(obstacles)
    obstacle_states = -1 * np.ones((n_obstacles, 3))

    # 遍历所有障碍物
    for i_ob in range(n_obstacles):
        obstacle = obstacles[i_ob]
        state = obstacle.state_at_time(T)
        if state is None:
            print('obstacle dead.')
            continue
        
        obstacle_states[i_ob, :] = state_cr2state_mcts(lanelet_ids_frenet_axis,lanelet_id_matrix, ln, state)

    # 删除为-1（空的）元素
    obstacle_states_in = obstacle_states[obstacle_states[:, 0] != -1, :]

    return obstacle_states_in



def get_map_info(is_goal, lanelet_id_goal,  lanelet_ids_frenet_axis, lanelet_id_matrix, ln: LaneletNetwork, planning_problem: PlanningProblem, is_interactive=False):
    ''' 获取map	地图信息，表达决策任务.	1x4矩阵：[总车道数，目标车道编号，目标位置, 限速(m/s)]（注：最左侧车道为0号）
    目标位置选择：如果is_goal==True。选择目标区域前端的点，尽早进入 goal region；如果不是，则需要延长至路口内。
    Params: 
        is_goal: 该阶段的最终目标位置是否已经抵达 goal_region.

    return:
        
    '''
    # 获取 goal_pos_end：未延长的目标终点。延长放在之后做
    if is_goal:
        # 如果是直接规划到 goal。goal pos设置为矩形区域"前部分"
        # 如果目标车道的头几个点在
        goal_pos_end  = planning_problem.goal.state_list[0].position.shapes[0].center
        
        # 假设 planning_problem.goal.state_list[0].position.shapes 就是对应一整个lanelet
        goal_lanelet_id = planning_problem.goal.lanelets_of_goal_position[0][0]
        goal_lanelet = ln.find_lanelet_by_id(goal_lanelet_id)
        # goal_pos_end_ = goal_lanelet.center_vertices[0, :]
        goal_pos_end_ = goal_lanelet.center_vertices[-1, :]
        lanelets_of_goal = ln.find_lanelet_by_position([goal_pos_end_])[0]
        if lanelet_id_goal in lanelets_of_goal:
            # 如果假设成立，再进行赋值
            goal_pos_end = goal_pos_end_
        
    else:
        lanelet_goal = ln.find_lanelet_by_id( lanelet_id_goal)
        # ！！！ lanelet中心线最后一个点，居然不是该lanelet的
        goal_pos_end = lanelet_goal.center_vertices[-1, :]
        

    map  = []
    n_lane = lanelet_id_matrix.shape[0]         # 总车道数
    
    # 目标车道编号
    # lanelet_id_goal = lanelet_network.find_lanelet_by_position([goal_pos])[0][0]
    lane_pos_ = np.where(lanelet_id_matrix==lanelet_id_goal)[0]
    assert lane_pos_.shape[0] != 0, 'error. cannot found goal lanelet position'

    lane_pos = lane_pos_[0]

    # 求取frenet s轴对应的加密后的cv
    cv, _, s_cv = get_detail_cv_of_lanelets(lanelet_ids_frenet_axis, ln)
    
    # 目标s位置. 增加5米，直接延长至路口内。
    goal_s = distance_lanelet(cv, s_cv, cv[0, :], goal_pos_end) + 5

    speed_limit = extract_speed_limit_from_traffic_sign(ln)

    map = [n_lane, lane_pos, goal_s, speed_limit]

    return map


def get_frenet_lanelet_axis(lanelet_id_matrix):
    ''' 取lanelet_id_matrix中第一列中第一个可行的 lanelet 的中心线为frenet坐标系参考线
    Return:
        lanelet id.
    '''

    lanelet00id = -1
    lanelet00_line = -1
    for i in range(lanelet_id_matrix.shape[1]):
        if lanelet_id_matrix[i, 0] !=-1:
            lanelet00id =  lanelet_id_matrix[i, 0]
            lanelet00_line = i
            break
    # laneletid第一列不应该全都不可行
    assert lanelet00id != -1

    return lanelet_id_matrix[lanelet00_line,:]


def edit_scenario4test(scenario, ego_init_pos):
    # ------------ 删除车辆，使得每个车道最多一辆车 ------------------------------------
    # scenario.remove_obstacle(obstacle id)
    lanelet_network = scenario.lanelet_network

    lanelet_ids = [14, 17, 20, 23, 26]
    conf_point = []
    for i in range(len(lanelet_ids)):
        conf_point.append(ego_init_pos)

    obstacle_remain = [252, 254, 234, 126]
    obstacle_remove = []
    for i in range(len(scenario.obstacles)):
        obstacle_id = scenario.obstacles[i].obstacle_id
        if obstacle_id in obstacle_remain:
            continue
        obstacle_remove.append(obstacle_id)
    for obstalce_id_remove in obstacle_remove:
        scenario.remove_obstacle(scenario.obstacle_by_id(obstalce_id_remove))
    return scenario

def extract_speed_limit_from_traffic_sign(ln :Scenario.lanelet_network):
    if len(ln.traffic_signs) == 0:
        print('None traffic sign. Cannot extract')
        return None
    position_list = []
    speed_list = []
    for traffic_sign in ln.traffic_signs:
        position = traffic_sign.position
        for traffic_sign_element in traffic_sign.traffic_sign_elements:
            if not traffic_sign_element.traffic_sign_element_id.name == 'MAX_SPEED':
                continue

            position_list.append(position)
            speed_list.append(float(traffic_sign_element.additional_values[0]))
    max_speed = max(speed_list)
    # print('speed limit: ', max_speed)
    return max_speed

def generate_len_map(lanelet_network, lanelet_map, isContinous=True):
    """

    :param scenario: CR scenario
    :param lanelet_map: n by m lanelet id matrix, where n is the number of parallel lanes
    and m is the number of lanelets along the route, e.g.:
    [[-1, -1, 453],
    [-1, -1, 454],
    [210, 212, 239],
    [231, 232, 240]]
    (ps: -1 means unusable lanelet)
    :return: a list of usable length range in each lane
    """

    lm = lanelet_map  # lanelet map
    ln = lanelet_network

    # calculate length of each parallel lanelet set [m by 1]
    len_lanelet = np.zeros(3)
    for m in range(lm.shape[1]):
        for n in range(lm.shape[0]):
            if not lm[n, m] == -1:  # find id of a solid lanelet
                id = lm[n, m]
                lanelet = ln.find_lanelet_by_id(id)
                direction, length_temp = get_lane_feature(lanelet.center_vertices)
                len_lanelet[m] = np.sum(length_temp)
                break
    len_points = np.hstack((np.array(0), np.cumsum(len_lanelet)))

    # generate a list of usable length in each lane [n by <=m]
    len_map_temp = []
    len_map = []
    for n in range(lm.shape[0]):
        for m in range(lm.shape[1]):
            if not lm[n, m] == -1:
                solid_len_range = [len_points[m], len_points[m + 1]]
                len_map_temp.append(solid_len_range)
        len_map.append(len_map_temp)
        len_map_temp = []

    if not isContinous:
        return len_map

    #  concatenate adjacent periods [n by <m]
    for n in range(len(len_map)):
        for m in range(len(len_map[n]) - 1, 0, -1):
            if len_map[n][m][0] == len_map[n][m - 1][1]:
                print(n, m)
                len_map[n][m - 1][1] = len_map[n][m][1]
                del len_map[n][m]

    return len_map


if __name__ == '__main__':
    # 测试MCTS算法

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
    # 提取自车初始状态
    # scenario = edit_scenario4test(scenario, ego_pos_init)

    # ---------------可视化修改后的场景 ------------------------------
    plt.figure(figsize=(25, 10))
    # 画一小段展示一下
    for i in range(10):
        plt.clf()
        # draw_params = {
        #     'time_begin': i,
        #     'scenario':
        #         {'dynamic_obstacle': {'show_label': True, },
        #          'lanelet_network': {'lanelet': {'show_label': False, }, },
        #          },
        # }
        draw_params = {'lanelet': {'draw_start_and_direction': False, 'draw_center_bound': False},
                'dynamic_obstacle': {'show_label': True}}
        draw_object(scenario, draw_params=draw_params)
        draw_object(planning_problem_set)
        plt.gca().set_aspect('equal')
        plt.pause(0.001)
        # plt.show()
    plt.close()
    # ---------------可视化 end ------------------------------

    # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
    ln = scenario.lanelet_network
    lanelet_id_matrix = lanelet_network2grid(ln)
    print('lanelet_id_matrix: ', lanelet_id_matrix)

    # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
    T = 50  # 5*0.1=0.5.返回0.5s时周车的状态。注意下面函数返回的自车状态仍然是初始时刻的。
    grid, ego_d, obstacles = get_obstacle_info(ego_pos_init, lanelet_id_matrix, ln, scenario, T)
    # print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)

    v_ego = planning_problem.initial_state.velocity

    lanelet00_cv_info = detail_cv(ln.find_lanelet_by_id(lanelet_id_matrix[0, 0]).center_vertices)
    lane_ego_n_array, _ = np.where(grid == 1)

    goal_pos  =  planning_problem.goal.state_list[0].position.shapes[0].center
    map = get_map_info(goal_pos, grid, lanelet00_cv_info, lanelet_id_matrix, ln)
    if len(lane_ego_n_array)>0:
        lane_ego_n = lane_ego_n_array[0]
    else:
        print('ego_lane not found. out of lanelet')
        lane_ego_n = -1
    state = [lane_ego_n, ego_d, v_ego]

    # 决策初始时刻目前无法给出，需要串起来后继续[TODO]
    T = 0

    print('自车初始状态矩阵', state)
    print('地图信息', map)
    print('他车矩阵', obstacles)

    # generate a map of usable range of each lane
    lanelet_map = np.array([[-1, -1, 453],
                            [-1, -1, 454],
                            [210, 212, 239],
                            [231, 232, 240]])

    len_map = generate_len_map(scenario, lanelet_map)
