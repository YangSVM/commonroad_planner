from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os

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