# this main function as the body of the interactive planner
# it takes the current state of the CR scenario
# and outputs the next state of the ego vehicle

from networkx.generators import ego
from MCTs_v3a import output
import os
import matplotlib as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from route_planner import route_planner
from intersection_planner import IntersectionPlanner
from Lattice_CRv3 import Lattice_CRv3
# from simulation.simulations import create_video_for_simulation
# from bezier import biz_planner
from MCTs_CRv3 import MCTs_CRv3
from sumocr.visualization.video import create_video
from commonroad.scenario.scenario import Tag
from simulation.utility import save_solution
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
# from commonroad_dc.costs.evaluation import CostFunctionEvaluator
from commonroad_dc.feasibility.solution_checker import valid_solution

# attributes for saving the simualted scenarios
author = 'Desmond'
affiliation = 'Tongji & Tsinghua'
source = ''
tags = {Tag.URBAN}


class InteractiveCRPlanner:
    lanelet_ego = None
    lanelet_state = None

    def __init__(self, current_scenario, state_current_ego):
        # get current scenario info. from CR
        self.scenario = current_scenario
        self.ego_state = state_current_ego
        self.lanelet_ego = None  # the lanelet which ego car is located in
        self.lanelet_state = None  # straight-going /incoming /in-intersection

    def check_state(self):
        """check if ego car is straight-going /incoming /in-intersection"""

        ln = self.scenario.lanelet_network
        # find current lanelet
        self.lanelet_ego = ln.find_lanelet_by_position([self.ego_state.position])[0][0]

        for idx_inter, intersection in enumerate(ln.intersections):
            incomings = intersection.incomings

            for idx_inc, incoming in enumerate(incomings):
                incoming_lanelets = list(incoming.incoming_lanelets)
                in_intersection_lanelets = list(incoming.successors_straight)

                for laneletid in incoming_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 2  # incoming

                for laneletid in in_intersection_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 3  # in-intersection

        if self.lanelet_state is None:
            self.lanelet_state = 1  # straighting-going

    def generate_route(self, scenario, planning_problem):
        """

        :param planning_problem:
        :return: lanelet_route
        """
        route = route_planner(scenario, planning_problem)
        lanelet_route = route.list_ids_lanelets
        add_successor = scenario.lanelet_network.find_lanelet_by_id(lanelet_route[-1]).successor
        if add_successor:
            lanelet_route.append(add_successor[0])

        return lanelet_route

    def planning(self, current_scenario,
                 planning_problem,
                 ego_vehicle,
                 current_time_step,
                 last_action,
                 is_new_action_needed):

        """body of our planner"""
        #  get last action
        action = last_action

        # generate a global lanelet route from initial position to goal region
        lanelet_route = self.generate_route(current_scenario, planning_problem)

        # check state 1:straight-going /2:incoming /3:in-intersection
        self.check_state()
        print("current state:", self.lanelet_state)
        # self.lanelet_state = 1
        # send to sub planner according to current lanelet state
        # if self.lanelet_state == 1:
        if self.lanelet_state == 2 or self.lanelet_state == 1:

            # === insert straight-going planner here
            if is_new_action_needed:
                mcts_planner = MCTs_CRv3(current_scenario, planning_problem, lanelet_route, ego_vehicle)
                semantic_action, action = mcts_planner.planner(current_time_step)
            # else:
                # update action

                # for straight-going
                # action.ego_state_init[0] = ego_vehicle.current_state.position[0]
                # action.ego_state_init[1] = ego_vehicle.current_state.position[1]

                # for lane=changing
                # action.T -= 0.1

            # next_state, is_new_action_needed = biz_planner(current_scenario, action)
            lattice_planner = Lattice_CRv3(current_scenario, ego_vehicle)
            next_state, is_new_action_needed = lattice_planner.planner(action)
            # === end of straight-going planner

        # if self.lanelet_state == 2 or self.lanelet_state == 3:
        if self.lanelet_state == 3:
            # === insert intersection planner here
            is_new_action_needed = 1
            ip = IntersectionPlanner(current_scenario, lanelet_route, ego_vehicle, self.lanelet_state)
            next_state = ip.planning(current_time_step)
            # === end of intersection planner

        return next_state, action, is_new_action_needed


if __name__ == '__main__':
    from simulation.simulations import load_sumo_configuration
    from sumocr.maps.sumo_scenario import ScenarioWrapper
    from sumocr.interface.sumo_simulation import SumoSimulation

    # 曹雷
    # folder_scenarios = os.path.abspath(
    #     '/home/thor/commonroad-interactive-scenarios/competition_scenarios_new/interactive')
    # 奕彬
    folder_scenarios = os.path.abspath(
        '/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/scenarios_cr_competition/competition_scenarios_new/interactive/')
    # 晓聪
    # folder_scenarios = os.path.abspath(
    #     '/home/zxc/Downloads/competition_scenarios_new/interactive')

    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_model = VehicleModel.KS
    cost_function = CostFunction.TR1
    vehicle = VehicleDynamics.KS(vehicle_type)
    dt = 0.1
    # name_scenario = "DEU_Frankfurt-4_2_I-1"  # 交叉口测试场景
    name_scenario = "DEU_Frankfurt-4_3_I-1"  # 交叉口测试场景 2
    # name_scenario = "DEU_Frankfurt-95_9_I-1"  # 直道测试场景
    interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    #
    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    # num_of_steps = conf.simulation_steps
    num_of_steps = 300
    sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)

    #
    ego_vehicles = sumo_sim.ego_vehicles
    is_new_action_needed = True
    last_action = []
    t_record = 0
    for step in range(num_of_steps):
        print("process:", step, "/", num_of_steps)
        current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        ego_vehicle = list(ego_vehicles.values())[0]

        # ====== plug in your motion planner here
        # ====== paste in simulations

        # force to get a new action every 3 sceonds
        # t_record += 0.1
        # if t_record > 3:
        #     is_new_action_needed = True
        #     t_record = 0

        # generate a CR planner
        main_planner = InteractiveCRPlanner(current_scenario, ego_vehicle.current_state)
        next_state, last_action, is_new_action_needed = main_planner.planning(current_scenario,
                                                                              planning_problem,
                                                                              ego_vehicle,
                                                                              sumo_sim.current_time_step,
                                                                              last_action,
                                                                              is_new_action_needed)
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
                 planning_problem_set,
                 ego_vehicles,
                 True,
                 "_planner")

    pp_id = list(planning_problem_set.planning_problem_dict.keys())[0]
    ego_ori_key = list(ego_vehicles.keys())[0]
    ego_vehicles[pp_id] = ego_vehicle
    ego_vehicles.pop(ego_ori_key)
    # write simulated scenario to file
    fw = CommonRoadFileWriter(simulated_scenario, planning_problem_set, author, affiliation, source, tags)
    fw.write_to_file(f"{path_scenarios_simulated}{name_scenario}_planner.xml", OverwriteExistingFile.ALWAYS)

    # get trajectory
    trajectory = ego_vehicle.driven_trajectory.trajectory
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory, vehicle, dt)
    print('Feasible? {}'.format(feasible))

    # saves trajectory to solution file
    save_solution(simulated_scenario, planning_problem_set, ego_vehicles,
                  vehicle_type,
                  vehicle_model,
                  cost_function,
                  path_solutions, overwrite=True)

    solution = CommonRoadSolutionReader.open(os.path.join(path_solutions,
                                                          f"solution_KS1:TR1:{name_scenario}:2020a.xml"))
    valid_solution(scenario, planning_problem_set, solution)
    # ce = CostFunctionEvaluator.init_from_solution(solution)
    # cost_result = ce.evaluate_solution(scenario, planning_problem_set, solution)
    # print(cost_result)