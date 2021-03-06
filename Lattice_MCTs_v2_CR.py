from typing import Iterable
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from detail_central_vertices import detail_cv
from route_planner import route_planner
from intersection_planner import distance_lanelet

from grid_lanelet import edit_scenario4test
from grid_lanelet import lanelet_network2grid
from grid_lanelet import get_obstacle_info
# from MCTs_v2 import NaughtsAndCrossesState
# from MCTs_v2 import mcts

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
from commonroad.visualization.mp_renderer import MPRenderer

from Lattice_v2 import Obstacle
from Lattice_v2 import CalcRefLine
from Lattice_v2 import TrajPoint
from Lattice_v2 import SampleBasis
from Lattice_v2 import LocalPlanner
from Lattice_v2 import Dist

from generate_srd_map import Srd_map

class From_decision():
    def __init__(self, act, v_init, T):
        self.act = act
        self.v_init = v_init
        self.T = T
        self.path_point = []
        self.d_end = 0
        self.s_end = 0
        self.v_end = 0
        self.acc_end = 0

    def decode(self, path_point_l:None, path_point_e, path_point_r:None, s_init):
        v_init = self.v_init
        T = self.T
        if self.act == 1:
            self.path_point = path_point_l
            self.d_end = 0
            self.s_end = s_init + v_init*T
            self.v_end = v_init
        if self.act == 2:
            self.path_point = path_point_r
            self.d_end = 0
            self.s_end = s_init + v_init*T
            self.v_end = v_init
        if self.act == 3:
            self.path_point = path_point_e
            self.d_end = 0
            self.s_end = s_init + v_init*T + 0.5*T*T
            self.v_end = v_init + T
            self.acc_end = 1
        if self.act == 4:
            self.path_point = path_point_e
            self.d_end = 0
            self.s_end = s_init + v_init*T
            self.v_end = v_init
            self.acc_end = 0
        if self.act == 5:
            self.path_point = path_point_e
            self.d_end = 0
            self.s_end = s_init + v_init*T - 0.5*T*T
            self.v_end = v_init - T
            self.acc_end = -1


if __name__ == '__main__':
    M_PI = 3.141593
    delta_t = 0.1 * 1                           # fixed time between two consecutive trajectory points, sec
    # v_tgt = 2.0                               # fixed target speed, m/s
    T = 30                                       # decision-making horizon
    sight_range = 10                            # ????????????????????????????????????
    # -------------- ??????????????????common road??????????????? -----------------
    #  ?????? common road scenarios??????https://gitlab.lrz.de/tum-cps/commonroad-scenarios????????????????????????
    path_scenario_download = os.path.abspath('/home/thor/commonroad-interactive-scenarios/commonroad-scenarios/scenarios/NGSIM/US101')
    # ?????????
    id_scenario = 'USA_US101-16_1_T-1'
    path_scenario = os.path.join(path_scenario_download, id_scenario + '.xml')
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    # -------------- ???????????? -----------------
    # ?????????????????????????????????????????????
    # initial state of ego car [0,0]
    ego_pos_init = planning_problem.initial_state.position
    scenario = edit_scenario4test(scenario, ego_pos_init)
    # from commonroad scenario, get initial state
    ego_heading_init = planning_problem.initial_state.orientation
    v_ego = planning_problem.initial_state.velocity
    tp_list = [ego_pos_init[0], ego_pos_init[1], v_ego, 0, ego_heading_init, 0]
    traj_point = TrajPoint(tp_list)
    
    # ?????????????????????????????????lanelet?????????lanelet ????????????
    lanelet_network  = scenario.lanelet_network
    lanelet_id_matrix  = lanelet_network2grid(lanelet_network)
    grid, ego_d, obstacle_states =get_obstacle_info(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, T)
    # print('?????????????????????????????????',grid,'??????frenet??????', ego_d)
    ego_lane_init = np.nonzero(grid)[0][0]
    #print('????????????????????? ', v_ego)
    #print('????????????', obstacle_states)
    b = [ego_lane_init, ego_d, v_ego]
    ini = obstacle_states
    initialState = NaughtsAndCrossesState(b, ini)
    searcher = mcts(iterationLimit=5000)  # ??????????????????
    action = searcher.search(initialState=initialState)  # ???????????????????????????
    s_init = ego_d   # ??????????????????????????????lanelet???????????????frenet????????????????????

    # print(ego_pos_init)
    lanelet_ego_id = scenario.lanelet_network.find_lanelet_by_position([ego_pos_init])[0][0]
    if not lanelet_ego_id:
        print("ego car not in lanelet!")
    else:
        lanelet_ego = scenario.lanelet_network.find_lanelet_by_id(lanelet_ego_id)
    # Commonroad Map, find path_points
    sdr_map = Srd_map()
    sdr_map.generate_srd_map(lanelet_ego_id, scenario.lanelet_network)
    
    cts_points = []
    for i,j in zip(sdr_map.cv_current[:, 0],sdr_map.cv_current[:, 1]):
        cts_points.append([i,j])
    
    if sdr_map.cv_left is not None:
        cts_points_l = []
        for i,j in zip(sdr_map.cv_left[:, 0],sdr_map.cv_left[:, 1]):
            cts_points_l.append([i,j])
        cvs_l,a,b = detail_cv(cts_points_l)
        path_point_l = CalcRefLine(cvs_l)
    else:
        path_point_l = None
    if sdr_map.cv_right is not None:
        cts_points_r = []
        for i,j in zip(sdr_map.cv_right[:, 0],sdr_map.cv_right[:, 1]):
            cts_points_r.append([i,j])
        cvs_r,a,b = detail_cv(cts_points_r)
        path_point_r = CalcRefLine(cvs_r)
    else:
        path_point_r = None
    
    cvs,a,b = detail_cv(cts_points)
    # plt.plot(cvs[0],cvs[1])
    # plt.show()

    path_point_e = CalcRefLine(cvs)
    
    traj_point.MatchPath(path_point_e)
    #plot path_point and ego_car position
    path_x = []
    path_y = []
    for i in range(len(path_point_e)):
        path_x.append(path_point_e[i].rx)
        path_y.append(path_point_e[i].ry)
    # plt.plot(path_x,path_y)
    # plt.scatter(ego_pos_init[0],ego_pos_init[1],s=10, color='g')
    # plt.show()

    act = action.act
    print(act)
    from_decision = From_decision(act, v_ego, T/10)
    from_decision.decode(path_point_l, path_point_e, path_point_r, s_init)
    

    # sampling parameters
    theta_thr = M_PI/6                          # delta theta threhold, deviation from matched path
    # ttcs = [3, 4, 5]                            # static ascending time-to-collision, sec
    # dist_samp = [v_tgt * ttc for ttc in ttcs]   # sampling distances, m
    # dist_prvw = dist_samp[0]                    # preview distance, equal to minimal sampling distance
    # d_end_samp = [0]                            # sampling d_cond_end[0], probably useless
    samp_basis = SampleBasis(traj_point, theta_thr, from_decision, s_init)
    obstacle_list = []
    for j in range(0, len(scenario.obstacles)):
        obs_x = scenario.obstacles[j].state_at_time(0).position[0]
        obs_y = scenario.obstacles[j].state_at_time(0).position[1]
        obs_veh = scenario.obstacles[j].state_at_time(0).velocity
        obs_leg = scenario.obstacles[j].obstacle_shape.length
        obs_wid = scenario.obstacles[j].obstacle_shape.width
        obs_ori = scenario.obstacles[j].state_at_time(0).orientation
        obs_acc = scenario.obstacles[j].state_at_time(0).acceleration
        obstacle_list.append(Obstacle([obs_x, obs_y, obs_veh, obs_leg, obs_wid, obs_ori]))
    for obstacle in obstacle_list:
            obstacle.MatchPath(path_point_e)
    
    local_planner = LocalPlanner(traj_point, path_point_e, obstacle_list, samp_basis, from_decision)
    print("Status: ", local_planner.status, "If stop: ", local_planner.to_stop)
    # plt.plot(local_planner.traj_x,local_planner.traj_y,'r')
    # plt.title("Traj")
    # plt.show()
    traj_points_opt = local_planner.LocalPlanning(traj_point, path_point_e, obstacle_list, samp_basis)
    
    state_list = []
    # v_list = []
    # a_list = []
    # i = 0
    for t in range(0,60):
        
        # print(path_point_e[-1].rs - ego_d)
        if (ego_d - path_point_e[-1].rs) > -5:
            break

        if t % T == 0:
            grid, ego_d, obstacle_states =get_obstacle_info(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, t)
            # print('?????????????????????????????????',grid,'??????frenet??????', ego_d)
            # print(np.nonzero(grid)[0][0])
            if np.nonzero(grid)[0][0]:
                ego_lane_init = np.nonzero(grid)[0][0]
            else:
                print("cannot find ego car")
            #print('????????????????????? ', v_ego)
            #print('????????????', obstacle_states)
            b = [ego_lane_init, ego_d, v_ego]
            ini = obstacle_states
            initialState = NaughtsAndCrossesState(b, ini)
            searcher = mcts(iterationLimit=5000)  # ??????????????????
            action = searcher.search(initialState=initialState)  # ???????????????????????????
            s_init = ego_d   # ??????????????????????????????lanelet???????????????frenet????????????????????

            # if t % 1 == 0:
            
            # if not traj_points_opt:
            #     print("no path find")
            #     # break
            # else:
            #     traj_point = traj_points_opt[1]
            #     print("position: " , traj_point.x,traj_point.y)

            act = action.act
            print("act: ",act)
            time1  = time.time()
            # global: delta_t, v_tgt
            # from commonroad: path_data, tp_list, obstacles
            # sampling: input theta_thr, ttcs; ouput theta_samp, dist_samp, d_end_samp
            # to commonroad: traj_points (list) or False
            
            # find lanelet of ego car by position
            lanelet_ego_id = scenario.lanelet_network.find_lanelet_by_position([ego_pos_init])[0]
            for lanelet_ego_id_num in lanelet_ego_id:
                lanelet_ego = scenario.lanelet_network.find_lanelet_by_id(lanelet_ego_id_num)
            # Commonroad Map, find path_points
            sdr_map = Srd_map()
            sdr_map.generate_srd_map(lanelet_ego_id_num, scenario.lanelet_network)

            cts_points_e = []
            for i,j in zip(sdr_map.cv_current[:, 0],sdr_map.cv_current[:, 1]):
                cts_points_e.append([i,j])
        
            if sdr_map.cv_left is not None:
                cts_points_l = []
                for i,j in zip(sdr_map.cv_left[:, 0],sdr_map.cv_left[:, 1]):
                    cts_points_l.append([i,j])
                cvs_l,a,b = detail_cv(cts_points_l)
                path_point_l = CalcRefLine(cvs_l)
                # path_point_r = CalcRefLine(cts_points_l)
            else:
                path_point_l = None
            if sdr_map.cv_right is not None:
                cts_points_r = []
                for i,j in zip(sdr_map.cv_right[:, 0],sdr_map.cv_right[:, 1]):
                    cts_points_r.append([i,j])
                cvs_r,a,b = detail_cv(cts_points_r)
                path_point_r = CalcRefLine(cvs_r)
                # path_point_r = CalcRefLine(cts_points_r)
            else:
                path_point_r = None

            cvs,a,b = detail_cv(cts_points_e)
            path_point_e = CalcRefLine(cvs)  
            # path_point_e = CalcRefLine(cts_points_e)


            from_decision = From_decision(act, v_ego, T/10)
            from_decision.decode(path_point_l, path_point_e, path_point_r, s_init)
            path_points = from_decision.path_point

            if path_points == path_point_l:
                print("left lane change!")
            
            # get path point [rx,ry]
            path_x = []
            path_y = []
            for i in range(len(path_points)):
                path_x.append(path_points[i].rx)
                path_y.append(path_points[i].ry)
            plt.figure()
            plt.plot(path_x,path_y)
            # plt.show()
            # plt.pause(0.1)
            # print(path_points[0].rx, path_points[0].ry)
            d_end = from_decision.d_end
            s_end = from_decision.s_end
            v_end = from_decision.v_end
            v_tgt = v_end

            # return state of obstacles per time step
            obstacle_list = []
            for j in range(0, len(scenario.obstacles)):
                obs_x = scenario.obstacles[j].state_at_time(t).position[0]
                obs_y = scenario.obstacles[j].state_at_time(t).position[1]
                obs_veh = scenario.obstacles[j].state_at_time(t).velocity
                obs_leg = scenario.obstacles[j].obstacle_shape.length
                obs_wid = scenario.obstacles[j].obstacle_shape.width
                obs_ori = scenario.obstacles[j].state_at_time(t).orientation
                obs_acc = scenario.obstacles[j].state_at_time(t).acceleration
                obstacle_list.append(Obstacle([obs_x, obs_y, obs_veh, obs_leg, obs_wid, obs_ori]))
            for obstacle in obstacle_list:
                obstacle.MatchPath(path_points)
        
            # traj_point = traj_points_opt[1]
                
            traj_point.MatchPath(path_points)
            theta_thr = M_PI/6   # calibration!!
            # ttcs_ = [2, 3, 4, 5, 6, 7, 8]

            grid, ego_d, obstacle_states =get_obstacle_info(ego_pos_init, lanelet_id_matrix, lanelet_network, scenario, t)
            s_init = ego_d
            
            samp_basis = SampleBasis(traj_point, theta_thr, from_decision,s_init)
            local_planner = LocalPlanner(traj_point, path_points, obstacle_list, samp_basis, from_decision)
            print("Status: ", local_planner.status, "If stop: ", local_planner.to_stop)
            traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacle_list, samp_basis)
            # traj_samp = local_planner.traj_point_samp
            # for i in range(len(traj_samp)):
            #     plt.plot(traj_samp[i][0],traj_samp[i][1],'r')
            #     plt.title("Traj")
            # plt.show()


            traj_points=[]
            for tp_opt in traj_points_opt:
                traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])
            tx=[x[0] for x in traj_points ]
            ty=[y[1] for y in traj_points ]
            # plt.plot(tx,ty,'r')
            # plt.show()

            # print(len(traj_points)) delta(t)=0.1s

            # ego_state = State()
            # ego_state.position = [traj_point.x, traj_point.y]
            # ego_state.velocity = traj_point.v
            # ego_state.orientation = traj_point.theta
            # ego_state.time_step = t

            # ego_pos_init = np.array([traj_point.x, traj_point.y])
            # # plt.plot(path_x,path_y)
            # # plt.scatter(ego_pos_init[0],ego_pos_init[1],s=10, color='g')
            # # plt.show()
            # v_ego = traj_point.v
            # state_list.append(ego_state)

            for j in range(0, len(traj_points)):
                ego_state = State()
                ego_state.position = [traj_points[j][0], traj_points[j][1]]
                ego_state.velocity = traj_points[j][2]
                ego_state.orientation = traj_points[j][4]
                ego_state.time_step = t + j
                state_list.append(ego_state)
    
            ego_pos_init = np.array([traj_points[-1][0], traj_points[-1][1]])
            # plt.plot(path_x,path_y)
            plt.scatter(ego_pos_init[0],ego_pos_init[1],s=10, color='r')
            plt.show()
            v_ego = traj_points[-1][2]
            traj_point = traj_points_opt[-1]


    # create the planned trajectory starting at time step 1
    ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])
    # create the prediction using the planned trajectory and the shape of the ego vehicle

    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
    ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                                    shape=ego_vehicle_shape)

    # the ego vehicle can be visualized by converting it into a DynamicObstacle
    ego_vehicle_type = ObstacleType.CAR
    state_init = planning_problem.initial_state
    ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                    obstacle_shape=ego_vehicle_shape, initial_state=state_init,
                                    prediction=ego_vehicle_prediction)

    plt.figure(1)
    for i in range(0, 60):
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i})
        ego_vehicle.draw(rnd, draw_params={'time_begin': i, 'dynamic_obstacle': {
            'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
                'facecolor': 'r'}}}}}})
        planning_problem_set.draw(rnd)
        rnd.render()
        plt.pause(0.01)
    plt.show()

