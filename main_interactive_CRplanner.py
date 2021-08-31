# this main function as the body of the interactive planner
# it takes the current state of the CR scenario
# and outputs the next state of the ego vehicle
import copy
from time import sleep

from commonroad.planning.planning_problem import PlanningProblem

from CR_tools.utility import distance_lanelet, brake
from networkx.generators import ego
import os

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
# from commonroad_dc.costs.evaluation import CostFunctionEvaluator
from commonroad_dc.feasibility.solution_checker import valid_solution
from sumocr.visualization.video import create_video
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.interface.sumo_simulation import SumoSimulation
from simulation.utility import save_solution
from simulation.simulations import load_sumo_configuration

from route_planner import route_planner
from Lattice_CRv3 import Lattice_CRv3
from intersection_planner import front_vehicle_info_extraction, IntersectionPlanner
from MCTs_CR import MCTs_CR
from CR_tools.utility import distance_lanelet, brake
from commonroad.common.solution import PlanningProblemSolution, Solution, CommonRoadSolutionWriter, VehicleType, \
    VehicleModel, CostFunction

# attributes for saving the simualted scenarios
author = 'Desmond'
affiliation = 'Tongji & Tsinghua'
source = ''
tags = {Tag.URBAN}


class InteractiveCRPlanner:
    lanelet_ego = None
    lanelet_state = None

    def __init__(self):
        # get current scenario info. from CR
        self.lanelet_ego = None  # the lanelet which ego car is located in
        self.lanelet_state = None  # straight-going /incoming /in-intersection
        self.lanelet_route = None
        # initialize the last action info
        self.is_new_action_needed = True
        self.last_action = []
        self.last_semantic_action = []

        self.next_states_queue = []
        # goal infomation. [MCTs目标是否为goal_region, frenet中线(略)，中线距离(略)，目标位置]
        self.goal_info = None

    def check_state(self):
        """check if ego car is straight-going /incoming /in-intersection"""
        lanelet_id_ego = self.lanelet_ego
        ln = self.scenario.lanelet_network
        # find current lanelet
        potential_ego_lanelet_id_list = \
            self.scenario.lanelet_network.find_lanelet_by_position([self.ego_state.position])[0]
        for idx in potential_ego_lanelet_id_list:
            if idx in self.lanelet_route:
                lanelet_id_ego = idx
        self.lanelet_ego = lanelet_id_ego
        print('current lanelet id:', self.lanelet_ego)

        for idx_inter, intersection in enumerate(ln.intersections):
            incomings = intersection.incomings

            for idx_inc, incoming in enumerate(incomings):
                incoming_lanelets = list(incoming.incoming_lanelets)
                in_intersection_lanelets = list(incoming.successors_straight) + \
                                           list(incoming.successors_right) + list(incoming.successors_left)

                for laneletid in incoming_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 2  # incoming

                for laneletid in in_intersection_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 3  # in-intersection

        if self.lanelet_state is None:
            self.lanelet_state = 1  # straighting-going        

    def check_state_again(self, current_scenario, ego_vehicle):
        # # 路口规划器，交一部分由MCTS进行决策
        if self.lanelet_state == 3:
            ip = IntersectionPlanner(current_scenario, self.lanelet_route, ego_vehicle, self.lanelet_state)
            dis_ego2cp, _ = ip.desicion_making()
            if len(dis_ego2cp) == 0 or dis_ego2cp > 150:
                self.lanelet_state = 4
                if not self.last_state == 4:  # 如果已经在4 不需要新的action
                    self.is_new_action_needed = 1  # 必须进入MCTS
        return

    def generate_route(self, scenario, planning_problem):
        """

        :param planning_problem:
        :return: lanelet_route
        """
        route = route_planner(scenario, planning_problem)
        if route:
            self.lanelet_route = route.list_ids_lanelets
        # add_successor = scenario.lanelet_network.find_lanelet_by_id(lanelet_route[-1]).successor
        # if add_successor:
        #     lanelet_route.append(add_successor[0])

        return self.lanelet_route

    def check_goal_state(self, position):
        # 没有经过
        goal_info = self.goal_info
        if goal_info is None:
            return False

        is_goal = False
        is_goal4mcts = goal_info[0]
        # 必须是mcts的终点是问题终点
        if is_goal4mcts:
            cv, cv_s, s_goal = goal_info[1:]
            s_ego = distance_lanelet(cv, cv_s, cv[0, :], position)
            # 自车s距离已经超过终点距离
            if s_ego >= s_goal:
                is_goal = True
        return is_goal

    def initialize(self, folder_scenarios, name_scenario):

        self.vehicle_type = VehicleType.FORD_ESCORT
        self.vehicle_model = VehicleModel.KS
        self.cost_function = CostFunction.TR1
        self.vehicle = VehicleDynamics.KS(self.vehicle_type)
        self.dt = 0.1

        interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

        conf = load_sumo_configuration(interactive_scenario_path)
        scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
        self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_file).open()

        #
        scenario_wrapper = ScenarioWrapper()
        scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
        scenario_wrapper.initial_scenario = self.scenario

        # self.num_of_steps = conf.simulation_steps
        self.num_of_steps = 40
        sumo_sim = SumoSimulation()

        # initialize simulation
        sumo_sim.initialize(conf, scenario_wrapper, None)

        self.t_record = 0

        return sumo_sim

    def process(self, sumo_sim):

        # generate ego vehicle
        ego_vehicles = sumo_sim.ego_vehicles

        for step in range(self.num_of_steps):
            if step == 126:
                print('debug')
                pass

            print("process:", step, "/", self.num_of_steps)
            current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
            planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[0]
            ego_vehicle = list(ego_vehicles.values())[0]

            # initial positions do not match, stupid!!!
            planning_problem.initial_state.position = copy.deepcopy(ego_vehicle.current_state.position)
            planning_problem.initial_state.orientation = copy.deepcopy(ego_vehicle.current_state.orientation)
            planning_problem.initial_state.velocity = copy.deepcopy(ego_vehicle.current_state.velocity)
            # ====== plug in your motion planner here
            # ====== paste in simulations

            # force to get a new action every 1 sceonds
            self.t_record += 0.1
            if self.t_record > 1 and self.last_semantic_action not in {1, 2}:
                self.is_new_action_needed = True
                print('force to get a new action during straight-going')
                self.t_record = 0

            # generate a CR planner
            next_state = self.planning(current_scenario,
                                       planning_problem,
                                       ego_vehicle,
                                       sumo_sim.current_time_step)

            print('velocity:', next_state.velocity)
            print('position:', next_state.position)
            # ====== paste in simulations
            # ====== end of motion planner
            next_state.time_step = 1
            next_state.steering_angle = 0.0
            trajectory_ego = [next_state]
            ego_vehicle.set_planned_trajectory(trajectory_ego)

            sumo_sim.simulate_step()

        # retrieve the simulated scenario in CR format
        simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

        # stop the simulation
        sumo_sim.stop()

        # match pp_id
        ego_vehicles = {list(self.planning_problem_set.planning_problem_dict.keys())[0]:
                            ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

        for pp_id, planning_problem in self.planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            simulated_scenario.add_objects(obstacle_ego)

        return simulated_scenario, ego_vehicles

    def planning(self, current_scenario,
                 planning_problem: PlanningProblem,
                 ego_vehicle,
                 current_time_step):

        """body of our planner"""
        #  get last action
        self.scenario = current_scenario
        self.ego_state = ego_vehicle.current_state

        action = self.last_action
        semantic_action = self.last_semantic_action

        # planning problem start from current vehicle states
        # generate a global lanelet route from initial position to goal region
        self.generate_route(current_scenario, planning_problem)

        # check for goal info
        is_goal = self.check_goal_state(ego_vehicle.current_state.position)
        if is_goal and self.is_new_action_needed:
            print('goal reached! braking!')
            # 直接刹车
            # next_state = brake(ego_vehicle.current_state, self.goal_info[1], self.goal_info[2])
            action = brake(self.scenario, ego_vehicle)
            self.is_new_action_needed = False
            # update the last action info
            self.last_action = action

            # lattice planning
            lattice_planner = Lattice_CRv3(self.scenario, ego_vehicle)
            next_state, _ = lattice_planner.planner(action)
            return next_state

        # check state 1:straight-going /2:incoming /3:in-intersection
        self.last_state = copy.deepcopy(self.lanelet_state)
        self.check_state()
        if self.lanelet_state == 3:
            self.check_state_again(current_scenario, ego_vehicle)
        print("current state:", self.lanelet_state)
        # self.lanelet_state = 1
        # send to sub planner according to current lanelet state
        if self.lanelet_state in {1, 2, 4}:

            if len(self.next_states_queue)>0:
                print('use next_states_buffer')
                next_state = self.next_states_queue.pop(0)
                return next_state
            # === insert straight-going planner here
            if self.is_new_action_needed:
                mcts_planner = MCTs_CR(current_scenario, planning_problem, self.lanelet_route, ego_vehicle)
                semantic_action, action, self.goal_info = mcts_planner.planner(current_time_step)
                self.is_new_action_needed = False
                # if semantic_action in {3, 4, 5}:
                #     action.T = action.T / 4
                #     action.v_end = action.ego_state_init[2] + (action.v_end - action.ego_state_init[2]) / 4
                #     action.delta_s = action.T * action.v_end
            else:
                # update action
                action.T -= 0.1

            # get front car info.
            front_veh_info = front_vehicle_info_extraction(self.scenario,
                                                           self.ego_state.position,
                                                           self.lanelet_route)
            if semantic_action in {3, 4, 5}:
                # too close to front car, start to car-following
                ttc = front_veh_info['dhw'] / (self.ego_state.velocity - front_veh_info['v'])
                if 0 < ttc < 5:
                    print('ttc', ttc)
                    print('too close to front car, start to car-following')
                    action_temp = copy.deepcopy(action)
                    action_temp.delta_s = front_veh_info['dhw']
                    action_temp.v_end = front_veh_info['v'] - 3
                    action_temp.T = action_temp.delta_s / (action_temp.v_end + self.ego_state.velocity) * 2
                    action = action_temp

            # print('init position:', action.ego_state_init)
            # print('frenet_cv:', action.frenet_cv[0, :], 'to', action.frenet_cv[-1:])
            print('delta_s:', action.delta_s)
            print('T_duration:', action.T)
            print('v_end:', action.v_end)
            lattice_planner = Lattice_CRv3(current_scenario, ego_vehicle)
            self.next_states_queue, self.is_new_action_needed = lattice_planner.planner(action)
            next_state = self.next_states_queue.pop(0)
            # === end of straight-going planner

        # if self.lanelet_state == 2 or self.lanelet_state == 3:
        if self.lanelet_state == 3:
            # === insert intersection planner here
            self.is_new_action_needed = 1
            semantic_action = 9
            ip = IntersectionPlanner(current_scenario, self.lanelet_route, ego_vehicle, self.lanelet_state)
            next_state = ip.planning(current_time_step)
            # === end of intersection planner

        # update the last action info
        self.last_action = action
        self.last_semantic_action = semantic_action

        return next_state


def motion_planner_interactive(scenario_path: str):
    main_planner = InteractiveCRPlanner()
    paths = scenario_path.split('/')
    name_scenario = paths[-1]
    # folder_scenarios = os.path.join(*paths[:-1])
    folder_scenarios = "/commonroad/scenarios"
    sumo_sim = main_planner.initialize(folder_scenarios, name_scenario)

    simulated_scenario, ego_vehicles = main_planner.process(sumo_sim)

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)
    print('Feasible? {}'.format(feasible))
    if not feasible:
        # if not feasible. reconstruct the inputs
        ego_vehicle.driven_trajectory.trajectory.state_list = reconstructed_inputs.state_list

    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in main_planner.planning_problem_set.planning_problem_dict
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=main_planner.vehicle_type,
                                           vehicle_model=main_planner.vehicle_model,
                                           cost_function=main_planner.cost_function,
                                           trajectory=ego_vehicle.driven_trajectory.trajectory))

    solution = Solution(simulated_scenario.scenario_id, pps)

    return solution


if __name__ == '__main__':

    # 曹雷
    # folder_scenarios = os.path.abspath(
    #     '/home/thor/commonroad-interactive-scenarios/competition_scenarios_new/interactive')
    # 奕彬
    folder_scenarios = os.path.abspath(
        '/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/scenarios_cr_competition/competition_scenarios_new/interactive')
    # 晓聪
    # folder_scenarios = os.path.abspath(
    #     '/home/zxc/Downloads/competition_scenarios_new/interactive')
    # name_scenario = "DEU_Frankfurt-24_7_I-1"
    name_scenario = "DEU_Frankfurt-7_6_I-1"

    main_planner = InteractiveCRPlanner()

    sumo_sim = main_planner.initialize(folder_scenarios, name_scenario)

    simulated_scenario, ego_vehicles = main_planner.process(sumo_sim)

    # path for outputting results
    # output_path = '/home/zxc/Videos/CR_outputs/'
    output_path = '/home/thicv/codes/commonroad/CR_outputs'

    # video
    output_folder_path = os.path.join(output_path, 'videos/')
    # solution
    path_solutions = os.path.join(output_path, 'solutions/')
    # simulated scenarios
    path_scenarios_simulated = os.path.join(output_path, 'simulated_scenarios/')

    # create mp4 animation
    create_video(simulated_scenario,
                 output_folder_path,
                 main_planner.planning_problem_set,
                 ego_vehicles,
                 True,
                 "_planner")

    # write simulated scenario to file
    fw = CommonRoadFileWriter(simulated_scenario, main_planner.planning_problem_set, author, affiliation, source, tags)
    fw.write_to_file(f"{path_scenarios_simulated}{name_scenario}_planner.xml", OverwriteExistingFile.ALWAYS)

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)
    print('Feasible? {}'.format(feasible))
    if not feasible:
        # if not feasible. reconstruct the inputs
        ego_vehicle.driven_trajectory.trajectory.state_list = reconstructed_inputs.state_list

    # saves trajectory to solution file
    save_solution(simulated_scenario, main_planner.planning_problem_set, ego_vehicles,
                  main_planner.vehicle_type,
                  main_planner.vehicle_model,
                  main_planner.cost_function,
                  path_solutions, overwrite=True)

    solution = CommonRoadSolutionReader.open(os.path.join(path_solutions,
                                                          f"solution_KS1:TR1:{name_scenario}:2020a.xml"))
    res = valid_solution(main_planner.scenario, main_planner.planning_problem_set, solution)
    print(res)
    # ce = CostFunctionEvaluator.init_from_solution(solution)
    # cost_result = ce.evaluate_solution(scenario, planning_problem_set, solution)
    # print(cost_result)
