# -*- coding: UTF-8 -*-
from functools import reduce
import numpy as np
import os
import math
import matplotlib.pyplot as plt
# from utils import plot_lanelet
from commonroad.common.file_reader import CommonRoadFileReader

import time


def detail_cv(cv):  # 将原车道中心线上的点加密为0.1m间隔的点
    """

    :param cv:一个lanelet的原始中线点序列（lanelet.center_vertices）
    :return: new_cv：密集化后的中线点序列
            new_direct：对应点处的方向
            new_add_len：对应点处，从当前lanelet起点开始，沿着中线的行驶距离）
    """
    [direct, add_length] = get_lane_feature(cv)
    dist_interval = 0.1
    new_cv = [[], []]
    new_direct = []
    new_add_len = [0]
    temp_length = dist_interval
    for k in range(0, len(cv) - 1):
        new_cv[0].append(cv[k][0])
        new_cv[1].append(cv[k][1])
        new_add_len.append(temp_length)
        new_direct.append(direct[k])
        while temp_length < add_length[k + 1]:
            temp_length += dist_interval
            new_cv[0].append(new_cv[0][-1] + dist_interval * math.cos(direct[k]))
            new_cv[1].append(new_cv[1][-1] + dist_interval * math.sin(direct[k]))
            new_add_len.append(temp_length)
            new_direct.append(direct[k])
    return [new_cv, new_direct, new_add_len]

# 根据车道中心线坐标计算行驶方向和线长度序列
def get_lane_feature(cv):  # cv: central vertice
    cv = np.array(cv)

    x_prior = cv.T[0][:-1]
    y_prior = cv.T[1][:-1]
    x_post = cv.T[0][1:]
    y_post = cv.T[1][1:]
    # 根据前后中心点坐标计算【行驶方向】
    dx = x_post - x_prior
    dy = y_post - y_prior

    direction = list(map(lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx)))

    length = np.sqrt(dx ** 2 + dy ** 2)
    length = length.tolist()
    for i in range(len(length) - 1):
        length[i + 1] += length[i]
    length.insert(0, 0)
    return direction, length

# def calc_cross_point(self, la, lb):
#     # skip start and end position
#     nmd = []
#     for n in range(1, len(la.x()) - 1):
#         for m in range(1, len(lb.x()) - 1):
#             d = (la.x()[n] - lb.x()[m]) ** 2 + (la.y()[n] - lb.y()[m]) ** 2
#             nmd.append((n, m, d))
#     res = reduce(lambda p1, p2: p1[2] < p2[2] and p1 or p2, nmd)
#     if res[2] < 0.2:
#         n, m = res[0], res[1]
#         la_offset = la.add_length[n]
#         lb_offset = lb.add_length[m]
#         la.add_cross(la_offset, lb_offset, lb, (la.x()[n], la.y()[n]))
#         lb.add_cross(lb_offset, la_offset, la, (la.x()[n], la.y()[n]))


if __name__ == '__main__':
        #  下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    folder_scenarios = os.path.abspath(
        '../competition_scenarios_new/interactive/')
    # folder_scenarios = os.path.abspath('/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/hand-crafted')
    # 文件名
    name_scenario = "DEU_Frankfurt-4_3_I-1"  # 交叉口测试场景
    # name_scenario = "DEU_Frankfurt-95_2_I-1"  # 直道测试场景
    interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)
    scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    # lanelets
    lanelets = scenario.lanelet_network.lanelets
    plt.figure(1)
    for lanelet in lanelets:
        cv_original = lanelet.center_vertices
        # t1 = time.time()
        cv_info = detail_cv(cv_original)
        # t2 = time.time()
        # print('total time\n',t1-t2)
        cv_new = cv_info[0]
        # plot_lanelet(lanelet)
        plt.plot(cv_new[0][:], cv_new[1][:], 'b*')
        plt.show()
