# -*- coding: UTF-8 -*-
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import str_len


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

def plot_lanelet(lanelet):
    lv = lanelet.left_vertices
    rv = lanelet.right_vertices
    plt.plot(lv[:,0], lv[:,1], 'k-')
    plt.plot(rv[:,0], rv[:,1],  'k-')
    start_line = np.array([lv[0,:], rv[0,:]])
    end_line = np.array([lv[-1,:], rv[-1,:]])
    plt.plot(start_line[:,0], start_line[:,1], 'r--')
    plt.plot(end_line[:,0], end_line[:,1], 'r--')
    
    mid_id = int((len(lv))/2)
    center = (lv[mid_id, :] + rv[mid_id, :])/2
    arror_end_id = min(mid_id+1, len(lv))
    arror_end = (lv[arror_end_id, :] + rv[arror_end_id, :])/2
    plt.plot(center[0], center[1], 'r*')
    # plt.arrow(center[0], center[1], arror_end[0] - center[0], arror_end[1]- center[1])
    plt.annotate(str(lanelet.lanelet_id), (center[0], center[1]) )
    plt.annotate('', xytext=(center[0], center[1]), xy=(arror_end[0], arror_end[1]), arrowprops=dict(arrowstyle="->"))
    # plt.show()


if __name__=='__main__':
    # 写一个plot道路结构的函数
    for lanelet in lanelet_network.lanelets:
        plot_lanelet(lanelet)
    plt.show()
        
