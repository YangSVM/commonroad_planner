from commonroad.common.file_reader import CommonRoadFileReader

import os
import numpy as np
import matplotlib.pyplot as plt


class Srd_map():

    def __init__(self):
        self.cv_current = None
        self.cv_left = None
        self.cv_right = None

    def generate_srd_map(self, ego_id, ln):
        # retrieve lanelets
        current_lanelet = ln.find_lanelet_by_id(ego_id)
        current_cv = np.array(current_lanelet.center_vertices)
        self.cv_current = current_cv
        if not current_lanelet.successor == []:
            current_lanelet_successor = ln.find_lanelet_by_id(current_lanelet.successor[0])
            current_cv_successor = np.array(current_lanelet_successor.center_vertices)
            self.cv_current = np.concatenate((self.cv_current, current_cv_successor), axis=0)

        if current_lanelet.adj_left_same_direction:
            left_lanelet = ln.find_lanelet_by_id(current_lanelet.adj_left)
            left_cv = np.array(left_lanelet.center_vertices)
            self.cv_left = left_cv
            if not left_lanelet.successor == []:
                left_lanelet_successor = ln.find_lanelet_by_id(left_lanelet.successor[0])
                left_cv_successor = np.array(left_lanelet_successor.center_vertices)
                self.cv_left = np.concatenate((self.cv_left, left_cv_successor), axis=0)

        if current_lanelet.adj_right_same_direction:
            right_lanelet = ln.find_lanelet_by_id(current_lanelet.adj_right)
            right_cv = np.array(right_lanelet.center_vertices)
            self.cv_right = right_cv
            if not right_lanelet.successor == []:
                right_lanelet_successor = ln.find_lanelet_by_id(right_lanelet.successor[0])
                right_cv_successor = np.array(right_lanelet_successor.center_vertices)
                self.cv_right = np.concatenate((self.cv_right, right_cv_successor), axis=0)


if __name__ == '__main__':
    # 下载 common road scenarios包。https://gitlab.lrz.de/tum-cps/commonroad-scenarios。修改为下载地址
    path_scenario_download = os.path.abspath('/home/zxc/Downloads/commonroad-scenarios/scenarios/hand-crafted')
    # 文件名
    id_scenario = 'ZAM_Zip-1_67_T-1'

    path_scenario = os.path.join(path_scenario_download, id_scenario + '.xml')
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()

    # id of ego car
    ego_id = 26

    # generate a map
    sdr_map = Srd_map()
    sdr_map.generate_srd_map(ego_id, scenario.lanelet_network)

    # plot lanelet network and map
    # plot_lanelet(scenario.lanelet_network.find_lanelet_by_id(ego_id))
    plot_lanelet_network(scenario.lanelet_network)
    plt.plot(sdr_map.cv_current[:, 0], sdr_map.cv_current[:, 1])
    if sdr_map.cv_left is not None:
        plt.plot(sdr_map.cv_left[:, 0], sdr_map.cv_left[:, 1])
    if sdr_map.cv_right is not None:
        plt.plot(sdr_map.cv_right[:, 0], sdr_map.cv_right[:, 1])

    plt.show()
