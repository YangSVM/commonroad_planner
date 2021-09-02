from typing import Iterable
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os
import time
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from detail_central_vertices import detail_cv

from grid_lanelet import edit_scenario4test
from grid_lanelet import lanelet_network2grid
from grid_lanelet import get_obstacle_info

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
from commonroad.visualization.mp_renderer import MPRenderer

from Lattice_v3 import Obstacle
from Lattice_v3 import CalcRefLine
from Lattice_v3 import TrajPoint
from Lattice_v3 import SampleBasis
from Lattice_v3 import LocalPlanner
from Lattice_v3 import Dist
from Lattice_v3 import CartesianToFrenet
from Lattice_v3 import PolyTraj

from generate_srd_map import Srd_map
from CR_tools.utility import smooth_cv

class Lattice_CRv3():
    def __init__(self, scenario, ego_vehicle):
        self.scenario = scenario
        # self.planning_problem = planning_problem
        # self.current_time_step = current_time_step
        self.ego_vehicle = ego_vehicle
        self.ego_state = ego_vehicle.current_state
    
    def is_require_decision(self,action,path_points):

        # path_points = action.frenet_cv
        ## real-time position
        # get ego planning init traj point
        ego_pos = self.ego_state.position
        
        ## decision target position
        ego_state_decision = action.ego_state_init
        tp_list_decison = [ego_state_decision[0], 
                            ego_state_decision[1], 
                            ego_state_decision[2],
                            ego_state_decision[3],
                            ego_state_decision[4],
                            0]
        traj_point_decision = TrajPoint(tp_list_decison)
        traj_point_decision.MatchPath(path_points)
        s_cond_decision_init, d_cond_decision_init = CartesianToFrenet(traj_point_decision.matched_point, traj_point_decision)
        s_decision_end = s_cond_decision_init[0] + action.delta_s

        total_t = action.T
        poly_traj = PolyTraj(s_cond_decision_init, d_cond_decision_init, total_t)
        s_cond_end = np.array([s_decision_end, action.v_end, action.a_end])
        poly_traj.GenLongTraj(s_cond_end)  
        d_end = 0     
        d_cond_end = np.array([d_end, 0, 0])
        poly_traj.GenLatTraj(d_cond_end)
        delta_t = 0.1
        target_pos = [poly_traj.GenCombinedTraj(path_points, delta_t)[-1].x, 
                            poly_traj.GenCombinedTraj(path_points, delta_t)[-1].y]
        if (Dist(ego_pos[0],ego_pos[1],target_pos[0],target_pos[1])< (8)):
            is_new_action_needed = True
        else:
            is_new_action_needed = False

        return is_new_action_needed
    
    def get_reference_line(self,frenet_cv):
        new_cv = smooth_cv(frenet_cv)
        cts_points = []
        for i,j in zip(new_cv[:, 0],new_cv[:, 1]):
            cts_points.append([i,j])
        ref_line = np.array(cts_points)
        path_point = CalcRefLine(ref_line)
        return path_point
    
    def plot_reference_line(self,path_points):
        path_x = []
        path_y = []
        path_theta = []
        path_kappa = []
        for i in range(len(path_points)):
            path_x.append(path_points[i].rx)
            path_y.append(path_points[i].ry)
            path_theta.append(path_points[i].rtheta)
            path_kappa.append(path_points[i].rkappa)
        plt.figure()
        num = [i for i in range(len(path_points))]
        plt.plot(num,path_theta)
        # plt.plot(num,path_kappa)
        # plt.plot(path_x,path_y)
        plt.show()

    def plot_traj_point(self,traj_points_opt):
        traj_points=[]
        for tp_opt in traj_points_opt:
            traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])
        tx=[x[0] for x in traj_points ]
        ty=[y[1] for y in traj_points ]
        tv=[v[2] for v in traj_points ]
        t=[i for i in range(len(traj_points))] 
        plt.plot(t,tv,'b')
        # plt.plot(tx,ty,'r')
        plt.show()

    def planner(self, action, semantic_action):
        # Exception handling while mcts tell lattice v_end should be zero
        # and the initial vecocity is zero

        # get ego planning init traj point
        ego_pos = self.ego_state.position
        ego_v = self.ego_state.velocity
        ego_heading = self.ego_state.orientation
        ego_acc = self.ego_state.acceleration
        if action.v_end == 0 and ego_v == 0:
            tp_list_init = [0, 0, 0, 0, 0, 0]
            tp_opt = TrajPoint(tp_list_init)
            tp_opt.a = 0
            tp_opt.v = 0
            tp_opt.x = ego_pos[0]
            tp_opt.y = ego_pos[1]
            tp_opt.theta = ego_heading
            traj_points = []
            traj_points.append([tp_opt.x, tp_opt.y, tp_opt.v, tp_opt.a, tp_opt.theta, tp_opt.kappa])
            print('Need to stop for a while!')
            is_new_action_needed = 1

            next_state = State()
            next_state.position = np.array([traj_points[0][0], traj_points[0][1]])
            next_state.velocity = traj_points[0][2]
            next_state.acceleration = traj_points[0][3]
            next_state.orientation = traj_points[0][4]
            return [next_state], is_new_action_needed
        t=0
        M_PI = 3.141593
        path_points = self.get_reference_line(action.frenet_cv)
        # plot reference line
        # self.plot_reference_line(path_points)
        is_new_action_needed = self.is_require_decision(action, path_points)
        if is_new_action_needed:
            print('current action finished, new action needed!')
        
        tp_list = [ego_pos[0], ego_pos[1], ego_v, ego_acc, ego_heading, 0]
        traj_point = TrajPoint(tp_list)
        traj_point.MatchPath(path_points)
        # get obstacle state
        obstacle_list = []
        for j in range(0, len(self.scenario.obstacles)):
            state = self.scenario.obstacles[j].state_at_time(0)  # zxc:scenario是实时的，所有T都改成0
            # 当前时刻这辆车可能没有
            if state is None:
                continue            
            obs_x = self.scenario.obstacles[j].state_at_time(t).position[0]
            obs_y = self.scenario.obstacles[j].state_at_time(t).position[1]
            obs_veh = self.scenario.obstacles[j].state_at_time(t).velocity
            obs_leg = self.scenario.obstacles[j].obstacle_shape.length
            obs_wid = self.scenario.obstacles[j].obstacle_shape.width
            obs_ori = self.scenario.obstacles[j].state_at_time(t).orientation
            obs_acc = self.scenario.obstacles[j].state_at_time(t).acceleration
            obstacle_list.append(Obstacle([obs_x, obs_y, obs_veh, obs_leg, obs_wid, obs_ori]))
        for obstacle in obstacle_list:
            obstacle.MatchPath(path_points)
        # decision initial ego state
        ego_state_decision = action.ego_state_init
        tp_list_decison = [ego_state_decision[0], 
                            ego_state_decision[1], 
                            ego_state_decision[2],
                            ego_state_decision[3],
                            ego_state_decision[4],
                            0]
        traj_point_decision = TrajPoint(tp_list_decison)
        traj_point_decision.MatchPath(path_points)
        s_cond_decision_init, _ = CartesianToFrenet(traj_point_decision.matched_point, traj_point_decision)
        s_decision_end = s_cond_decision_init[0] + action.delta_s
        # calibration init theta
        theta_thr = M_PI/6   
        # sample basis
        samp_basis = SampleBasis(traj_point, action, s_decision_end)
        # global variable
        delta_t = 0.1 * 1
        sight_range = 20
        # planner
        local_planner = LocalPlanner(traj_point, path_points, obstacle_list, samp_basis)
        # print("Status: ", local_planner.status, "If stop: ", local_planner.to_stop)
        traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacle_list, samp_basis)

        # 
        traj_points=[]
        if traj_points_opt != False:
            for tp_opt in traj_points_opt:
                traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])

            plt.clf()
            draw_parameters = {
                'time_begin': 1,
                'scenario':
                    {'dynamic_obstacle': {'show_label': True, },
                     'lanelet_network': {'lanelet': {'show_label': True, }, },
                     },
            }
            draw_object(self.scenario, draw_params=draw_parameters)
            trajectory = np.array(traj_points)
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'r*', zorder=30)
            plt.plot(self.ego_state.position[0], self.ego_state.position[1], 'b*', zorder=30)
            plt.plot(action.frenet_cv[:, 0], action.frenet_cv[:, 1], 'b', zorder=30)
            plt.axis([trajectory[0, 0] - 200., trajectory[0, 0] + 200.,
                      trajectory[0, 1] - 100., trajectory[0, 1] + 100.])

            plt.pause(0.01)
            # plt.show()

            horizon = 10
            if semantic_action in {1, 2}:
                horizon = 20
            n_points = min(len(traj_points), horizon)
            next_states = []
            for i_point in range(n_points):
                next_state = State()
                next_state.position = np.array([traj_points[i_point][0], traj_points[i_point][1]])
                next_state.velocity = traj_points[i_point][2]
                # next_state.acceleration = traj_points[1][3]
                next_state.acceleration = action.a_end
                next_state.orientation = traj_points[i_point][4]
                next_states.append(next_state)
            return next_states, is_new_action_needed
        # In the car following scenerio if lattice has no solution,
        # replanning by v_end=v_init along the road reference line
        else:
            # tp_list_init = [0,0,0,0,0,0]
            # tp_opt = TrajPoint(tp_list_init)
            # tp_opt.a = 0
            # tp_opt.v = ego_v + tp_opt.a * delta_t
            # tp_opt.x = ego_pos[0] + (ego_v * delta_t + 0.5 * tp_opt.a * delta_t * delta_t) * math.cos(ego_heading)
            # tp_opt.y = ego_pos[1] + (ego_v * delta_t + 0.5 * tp_opt.a * delta_t * delta_t) * math.sin(ego_heading)
            # tp_opt.theta = ego_heading
            # traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])
            action_follow = copy.deepcopy(action)
            action_follow.v_end = action_follow.ego_state_init[2]
            action_follow.a_end = 0
            action_follow.T = 3
            s_decision_end_follow = s_cond_decision_init[0] + action_follow.T * action_follow.v_end
            # sample basis
            samp_basis = SampleBasis(traj_point, action_follow, s_decision_end_follow)
            # global variable
            delta_t = 0.1 * 1
            sight_range = 20
            # planner
            local_planner = LocalPlanner(traj_point, path_points, obstacle_list, samp_basis)
            # print("Status: ", local_planner.status, "If stop: ", local_planner.to_stop)
            traj_points_opt_follow = local_planner.LocalPlanning(traj_point, path_points, obstacle_list, samp_basis)
            print('lattice has no solution, keep lane following, new action needed!')
            is_new_action_needed = 1

            if traj_points_opt_follow != False:
                for tp_opt in traj_points_opt_follow:
                    traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])

                plt.clf()
                draw_parameters = {
                    'time_begin': 1,
                    'scenario':
                        {'dynamic_obstacle': {'show_label': True, },
                         'lanelet_network': {'lanelet': {'show_label': True, }, },
                         },
                }
                draw_object(self.scenario, draw_params=draw_parameters)
                trajectory = np.array(traj_points)
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'r*', zorder=30)
                plt.plot(self.ego_state.position[0], self.ego_state.position[1], 'b*', zorder=30)
                plt.plot(action.frenet_cv[:, 0], action.frenet_cv[:, 1], 'b', zorder=30)
                plt.axis([trajectory[0, 0] - 200., trajectory[0, 0] + 200.,
                          trajectory[0, 1] - 100., trajectory[0, 1] + 100.])

                plt.pause(0.01)
                # plt.show()

                horizon = 5
                if semantic_action in {1, 2}:
                    horizon = 10
                n_points = min(len(traj_points), horizon)
                next_states = []
                for i_point in range(n_points):
                    next_state = State()
                    next_state.position = np.array([traj_points[i_point][0], traj_points[i_point][1]])
                    next_state.velocity = traj_points[i_point][2]
                    # next_state.acceleration = traj_points[1][3]
                    next_state.acceleration = action.a_end
                    next_state.orientation = traj_points[i_point][4]
                    next_states.append(next_state)
                return next_states, is_new_action_needed
            else:
                tp_list_init = [0,0,0,0,0,0]
                tp_opt = TrajPoint(tp_list_init)
                tp_opt.a = 0
                tp_opt.v = ego_v + tp_opt.a * delta_t
                tp_opt.x = ego_pos[0] + (ego_v * delta_t + 0.5 * tp_opt.a * delta_t * delta_t) * math.cos(ego_heading)
                tp_opt.y = ego_pos[1] + (ego_v * delta_t + 0.5 * tp_opt.a * delta_t * delta_t) * math.sin(ego_heading)
                tp_opt.theta = ego_heading
                traj_points.append([tp_opt.x,tp_opt.y,tp_opt.v,tp_opt.a,tp_opt.theta,tp_opt.kappa])
                print('lattice has no solution, new action needed!')
                is_new_action_needed = 1

                next_state = State()
                next_state.position = np.array([traj_points[0][0], traj_points[0][1]])
                next_state.velocity = traj_points[0][2]
                # next_state.acceleration = traj_points[1][3]
                next_state.acceleration = action.a_end
                next_state.orientation = traj_points[0][4]
                return [next_state], is_new_action_needed
        # plot trajectory points
        # self.plot_traj_point(traj_points_opt)
