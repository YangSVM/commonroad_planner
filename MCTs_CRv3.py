from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_dispatch_cr import draw_object
import os
from numpy.lib.function_base import gradient
from detail_central_vertices import detail_cv
from intersection_planner import distance_lanelet
import numpy as np
import matplotlib.pyplot as plt
from grid_lanelet import lanelet_network2grid
from grid_lanelet import ego_pos2tree
from grid_lanelet import get_map_info
from grid_lanelet import edit_scenario4test
from MCTs_v3 import NaughtsAndCrossesState
from MCTs_v3 import mcts


class MCTs_CRv3():
    def __init__(self, scenario, planning_problem, lanelet_route, ego_vehicle) -> None:
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.lanelet_route = lanelet_route
        self.ego_vehicle = ego_vehicle



    def planner(self, T):
        planning_problem  = self.planning_problem
        scenario =self.scenario

        # 原有场景车辆太多。删除部分车辆
        ego_pos_init = planning_problem.initial_state.position
        # 提取自车初始状态
        # scenario = edit_scenario4test(scenario, ego_pos_init)
        # ---------------可视化修改后的场景 ------------------------------
        # plt.figure(figsize=(25, 10))
        # # 画一小段展示一下
        # for  i in range(10):
        #     plt.clf()
        #     draw_parameters = {
        #         'time_begin':i, 
        #         'scenario':
        #         { 'dynamic_obstacle': { 'show_label': True, },
        #             'lanelet_network':{'lanelet':{'show_label': False,  },} ,
        #         },
        #     }
        #     draw_object(scenario, draw_params=draw_parameters)
        #     draw_object(planning_problem_set)
        #     plt.gca().set_aspect('equal')
        #     plt.pause(0.001)
        #     # plt.show()
        # plt.close()
        # ---------------可视化 end ------------------------------
        # 提供初始状态。位于哪个lanelet，距离lanelet 末端位置
        lanelet_network  = scenario.lanelet_network
        lanelet_id_matrix  = lanelet_network2grid(lanelet_network)
        print('lanelet_id_matrix: ', lanelet_id_matrix)
        # 在每次规划过程中，可能需要反复调用这个函数得到目前车辆所在的lanelet，以及相对距离
        T = 50           # 5*0.1=0.5.返回0.5s时周车的状态。注意下面函数返回的自车状态仍然是初始时刻的。
        grid, ego_d, obstacles =ego_pos2tree(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, T)
        # print('车辆所在车道标记矩阵：',grid,'自车frenet距离', ego_d)
        v_ego = planning_problem.initial_state.velocity
        lanelet00_cv_info = detail_cv(lanelet_network.find_lanelet_by_id(lanelet_id_matrix[0, 0]).center_vertices)
        lane_ego_n_array, _ = np.where(grid == 1)
        map = get_map_info(planning_problem, grid, lanelet00_cv_info, lanelet_id_matrix, lanelet_network)
        if len(lane_ego_n_array)>0:
            lane_ego_n = lane_ego_n_array[0]
        else:
            print('ego_lane not found. out of lanelet')
            lane_ego_n = -1
        state = [lane_ego_n, ego_d, v_ego]
        # 决策初始时刻目前无法给出，需要串起来后继续[TODO]
        T= 0        
        print('自车初始状态矩阵', state)
        print('地图信息', map)
        print('他车矩阵', obstacles)

        initialState = NaughtsAndCrossesState(state,map,obstacles)
        searcher = mcts(iterationLimit=5000) #改变循环次数或者时间
        action = searcher.search(initialState=initialState) #一整个类都是其状态
        #print(action.act)

if __name__=='__main__':
    print('error')