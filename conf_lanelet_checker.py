# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from detail_central_vertices import detail_cv

import os
import numpy as np
import math
import matplotlib.pyplot as plt

#  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
path_scenario_download = os.path.abspath('D:\OneDrive - tongji.edu.cn\Desktop\Study/1_Codes/1_CommonRoad\commonroad'
                                         '-scenarios\scenarios\hand-crafted')
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


def conf_lanelet_checker(ln, incoming_lanelet_id: int, direction: int):
    """
    check conflict lanelets when driving in a incoming lanelet

    :param ln: lanelet network of the scenario
    :param incoming_lanelet_id: the lanelet that subjective vehicle is located
    :param direction: 1:left 2:straight 3:right
    :return: Conf_lanelet类【.id:路口内的冲突lanelet的id；.conf_point:对应的冲突点位置】
    """

    class Conf_Lanelet:
        def __init__(self, id=None, conf_point=None):
            self.id = id
            self.conf_point = conf_point

    def check_sub_car_in_incoming():
        """ 返回主车所在的intersection序号和incoming序号 """

        # 初始化
        idx_intersect = []   # 主车所在的路口的序号
        idx_incoming = []  # 主车所在的incoming lanelet的序号
        lanelet_id_in_intersection = []  # 主车即将经过的路口中的lanelet的id

        # 遍历场景中的路口
        for idx, intersection in enumerate(ln.intersections):
            # 遍历路口中的每个incoming
            incomings = intersection.incomings

            for n, incoming in enumerate(incomings):

                laneletid = list(incoming.incoming_lanelets)
                if incoming_lanelet_id == laneletid[0]:  # 检查主车所在lanelet对应的incoming id
                    idx_incoming = n  # 保存主车所在的incoming lanelet的序号

                    # 记录主车即将进入的路口内的lanelet
                    if direction == 1:
                        lanelet_id_in_intersection = list(incoming.successors_left)
                    elif direction == 2:
                        lanelet_id_in_intersection = list(incoming.successors_straight)
                    else:
                        lanelet_id_in_intersection = list(incoming.successors_right)
                # 记录主车所在的【路口序号】
                idx_intersect = idx

        return idx_intersect, idx_incoming, lanelet_id_in_intersection[0]

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
        lanelet_sub = lanelet_network.find_lanelet_by_id(sub_lanelet_id_in_intersection)  # 主车的lanelet

        # 列出其他lanelet
        other_lanelet = list()
        for n in range(len(laneletid_list)):
            lanelet = lanelet_network.find_lanelet_by_id(laneletid_list[n])
            if not lanelet.lanelet_id == sub_lanelet_id_in_intersection:
                other_lanelet.append(lanelet)

        # 创建一个冲突lanelet的类，记录冲突信息
        cl = Conf_Lanelet()
        cl.id = []
        cl.conf_point = []
        for n in range(len(other_lanelet)):
            [isconf, conf_point] = check_collision(lanelet_sub.center_vertices, other_lanelet[n].center_vertices)
            if isconf:
                cl.id.append(other_lanelet[n].lanelet_id)
                cl.conf_point.append(conf_point)
        return cl

    # 主程序
    # 检查主车所在的路口和incoming序号
    [id_intersect, id_incoming, sub_lanelet_id_in_intersection] = check_sub_car_in_incoming()
    print("intersection no.", id_intersect)
    print("incoming_lanelet_id", id_incoming)
    print("lanelet id of subjective car:", sub_lanelet_id_in_intersection)

    # 提取路口内的lanelet的id列表
    inter_laneletid_list = check_in_intersection_lanelets()
    print("all lanelets:", inter_laneletid_list)

    # 检查主车在路口内所需通过的lanelet与其他路口内的lanelet的冲突情况
    cl = check_conf_lanelets(lanelet_network, inter_laneletid_list, sub_lanelet_id_in_intersection)
    print("conflict lanelets", cl.id)
    return cl


if __name__ == '__main__':
    cl_info = conf_lanelet_checker(lanelet_network, incoming_lanelet_id_sub, direction_sub)
