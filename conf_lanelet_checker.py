# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from detail_central_vertices import detail_cv

import os
import numpy as np
import math
import matplotlib.pyplot as plt


class Conf_Lanelet:
    '''冲突lanelet类a
        id: 路口内的冲突lanelet的id列表
        conf_point: 对应的冲突点位置 列表
    '''

    def __init__(self, id=None, conf_point=None):
        self.id = id
        self.conf_point = conf_point


def conf_lanelet_checker(ln, sub_lanelet_id: int, lanelet_state: int, lanelet_route):
    # change incoming id into current lanelet id and add
    # if ego car is in incoming or in intersection
    """
    check conflict lanelets when driving in a incoming lanelet

    :param ln: lanelet network of the scenario
    :param sub_lanelet_id: the lanelet that subjective vehicle is located
    :param lanelet_state: 1:left 2:straight 3:right
    :param lanelet_route: lanelet route
    :return: Conf_lanelet类【.id:路口内的冲突lanelet的id；.conf_point:对应的冲突点位置】
    """

    def check_sub_car_in_incoming():
        """ 返回主车所在的intersection序号和incoming序号 """
        route = lanelet_route
        # 初始化
        idx_intersect = []  # 主车所在的路口的序号
        lanelet_id_in_intersection = []  # 主车即将经过的路口中的lanelet的id
        sub_lanelet_id_incoming = []

        if lanelet_state == 2:
            id_temp = route.index(sub_lanelet_id)
            if id_temp+1 < len(route):  # incoming lanelet is NOT the last one in route
                lanelet_id_in_intersection = route[id_temp+1]

            sub_lanelet_id_incoming = sub_lanelet_id  # 找到主车进入路口的incoming,用于确定路口序号

        elif lanelet_state == 3:
            lanelet_id_in_intersection = sub_lanelet_id  # 此时主车已经在路口内，不用找id_in_intersection了
            sub_lanelet_id_incoming = ln.find_lanelet_by_id(sub_lanelet_id).predecessor[0]  # 找到主车进入路口的incoming,用于确定路口序号

        # 遍历场景中的路口，找到主车所在的路口
        for idx, intersection in enumerate(ln.intersections):
            # 遍历路口中的每个incoming
            incomings = intersection.incomings
            for n, incoming in enumerate(incomings):
                lanelet_id_list = setToArray(incoming.incoming_lanelets)
                if np.isin(sub_lanelet_id_incoming, lanelet_id_list):  # 检查主车所在lanelet对应的incoming id
                    # 记录主车所在的【路口序号】
                    idx_intersect = idx

        return idx_intersect, lanelet_id_in_intersection

    def check_in_intersection_lanelets():
        """ 返回提取路口内所有的lanelet的id列表 """
        interscetionlist = list(lanelet_network.intersections)
        intersection = interscetionlist[id_intersect]  # 主车所在路口
        incomings = intersection.incomings

        laneletlist = list()
        for n in range(len(incomings)):
            incoming = incomings[n]
            if len(incoming.successors_left):
                # print("incoming_id =", n, "left successor exists")
                list_temp = list(incoming.successors_left)
                laneletlist.append(list_temp[0])
            if len(incoming.successors_straight):
                # print("incoming_id =", n, "straight successor exists")
                list_temp = list(incoming.successors_straight)
                laneletlist.append(list_temp[0])
            if len(incoming.successors_right):
                # print("incoming_id =", n, "right successor exists")
                list_temp = list(incoming.successors_right)
                laneletlist.append(list_temp[0])
        return laneletlist

    def check_collision(cv_sub_origin, cv_other_origin):
        """检测两条中线之间是否存在冲突，返回冲突是否存在，以及冲突点位置（若存在） """

        # 加密中线
        cv_sub_detailed_info = detail_cv(cv_sub_origin)
        cv_other_detailed_info = detail_cv(cv_other_origin)

        # 转置，方便后续处理
        cv_sub_detailed = np.array(cv_sub_detailed_info[0])
        cv_sub_detailed = cv_sub_detailed.T
        cv_other_detailed = np.array(cv_other_detailed_info[0])
        cv_other_detailed = cv_other_detailed.T

        # 初始化冲突信息
        isconf = 0
        conf_point = []

        # 检测两条中线间的最近点
        for n in range(len(cv_sub_detailed)):
            for m in range(len(cv_other_detailed)):
                relative_position = cv_sub_detailed[n][:] - cv_other_detailed[m][:]
                dis = math.sqrt(relative_position[0] ** 2 + relative_position[1] ** 2)
                if dis < 0.1:
                    isconf = 1
                    conf_point = cv_other_detailed[m][:]
            if isconf == 1:
                break
        return isconf, conf_point

    def check_conf_lanelets(lanelet_network, laneletid_list: list, sub_lanelet_id_in_intersection: int):
        """ 返回存在冲突的lanelet列表（id,冲突位置） """
        # 创建一个冲突lanelet的类，记录冲突信息
        cl = Conf_Lanelet()

        if sub_lanelet_id_in_intersection:
            lanelet_sub = lanelet_network.find_lanelet_by_id(sub_lanelet_id_in_intersection)  # 主车的lanelet

            # 列出其他lanelet
            other_lanelet = list()
            for n in range(len(laneletid_list)):
                lanelet = lanelet_network.find_lanelet_by_id(laneletid_list[n])
                if not lanelet.lanelet_id == sub_lanelet_id_in_intersection:
                    other_lanelet.append(lanelet)

            cl.id = []
            cl.conf_point = []
            for n in range(len(other_lanelet)):
                [isconf, conf_point] = check_collision(lanelet_sub.center_vertices, other_lanelet[n].center_vertices)
                if isconf:
                    cl.id.append(other_lanelet[n].lanelet_id)
                    cl.conf_point.append(conf_point)
        else:

            cl.id = []
            cl.conf_point = []

        return cl

    # 主程序
    # 检查主车所在的路口和incoming序号
    [id_intersect, sub_lanelet_id_in_intersection] = check_sub_car_in_incoming()
    print("intersection no.", id_intersect)
    # print("incoming_lanelet_id", id_incoming)
    print("lanelet id of subjective car:", sub_lanelet_id_in_intersection)

    lanelet_network = ln
    # 提取路口内的lanelet的id列表
    inter_laneletid_list = check_in_intersection_lanelets()
    print("all lanelets:", inter_laneletid_list)

    # 检查主车在路口内所需通过的lanelet与其他路口内的lanelet的冲突情况
    cl = check_conf_lanelets(lanelet_network, inter_laneletid_list, sub_lanelet_id_in_intersection)
    print("conflict lanelets", cl.id)
    return cl


def potential_conf_lanelet_checkerv2(lanelet_network, cl_info):
    """
    check [potential] conflict lanelets when driving in a incoming lanelet

    :param ln: lanelet network of the scenario
    :param cl_info: 直接冲突lanelet的Conf_lanelet类【.id:路口内的冲突lanelet的id；.conf_point:对应的冲突点位置】
    :return: dict_lanelet_parent. 直接冲突lanelet_id ->父节点ID列表。若没有父节点则为None。【注意】仅有一个父节点时，字典的value为一个值的列表。
    """

    dict_parent_lanelet = {}
    for conf_lanelet_id in cl_info.id:
        conf_lanlet = lanelet_network.find_lanelet_by_id(conf_lanelet_id)
        parents = conf_lanlet.predecessor
        if parents is not None:
            for parent in parents:
                if parent not in dict_parent_lanelet.keys():
                    parent_lanelet = lanelet_network.find_lanelet_by_id(parent)

                    child_lanelet_ids = parent_lanelet.successor
                    dict_parent_lanelet[parent] = child_lanelet_ids

    return dict_parent_lanelet


def setToArray(setInput):
    arrayOutput = np.zeros((len(setInput), 1))
    index = 0
    for every in setInput:
        arrayOutput[index][0] = every
        index += 1
    return arrayOutput


# ————————————————
# 原文链接：https://blog.csdn.net/qq_41221841/article/details/109613783


if __name__ == '__main__':
    import matplotlib.pyplot  as plt
    from commonroad.visualization.draw_dispatch_cr import draw_object

    #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/zxc/Downloads/commonroad-scenarios/scenarios/hand-crafted')
    # 文件名
    id_scenario = 'ZAM_Tjunction-1_282_T-1'

    path_scenario = os.path.join(path_scenario_download, id_scenario + '.xml')
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    # retrieve the first planning problem in the problem set
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # lanelet_network
    lanelet_network = scenario.lanelet_network

    # 主车所在lanelet
    incoming_lanelet_id_sub = 50195
    direction_sub = 1
    cl_info = conf_lanelet_checker(lanelet_network, incoming_lanelet_id_sub, 2, direction_sub)

    plt.clf()
    draw_parameters = {
        'time_begin': 1,
        'scenario':
            {'dynamic_obstacle': {'show_label': True, },
             'lanelet_network': {'lanelet': {'show_label': True, }, },
             },
    }

    draw_object(scenario, draw_params=draw_parameters)
    draw_object(planning_problem_set)
    plt.gca().set_aspect('equal')
    # plt.pause(0.001)
    plt.show()
