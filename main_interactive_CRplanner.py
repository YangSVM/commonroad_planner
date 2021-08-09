# this main function as the body of the interactive planner
# it takes the current state of the CR scenario
# and outputs the next state of the ego vehicle

import os
from commonroad.common.file_reader import CommonRoadFileReader
from route_planner import route_planner
from intersection_planner import IntersectionPlanner
from Lattice_CRv3 import Lattice_CRv3
from bezier import biz_planner
# sys.path.append('/home/thicv/codes/commonroad/commonroad-interactive-scenarios')
from MCTs_CRv3 import MCTs_CRv3


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
        if self.lanelet_state == 1:

            # === insert straight-going planner here
            if is_new_action_needed:
                mcts_planner = MCTs_CRv3(current_scenario, planning_problem, lanelet_route, ego_vehicle)
                semantic_action, action = mcts_planner.planner(current_time_step)
            # next_state, is_new_action_needed = biz_planner(current_scenario, action)
            lattice_planner = Lattice_CRv3(current_scenario, ego_vehicle)
            next_state, is_new_action_needed = lattice_planner.planner(action)
            # === end of straight-going planner

        if self.lanelet_state == 2 or self.lanelet_state == 3:
            # === insert intersection planner here
            is_new_action_needed = 1
            ip = IntersectionPlanner(current_scenario, lanelet_route, ego_vehicle, self.lanelet_state)
            next_state, ev = ip.planner(current_time_step)
            # === end of intersection planner

        return next_state, action, is_new_action_needed


if __name__ == '__main__':
    from simulation.simulations import load_sumo_configuration
    from sumocr.maps.sumo_scenario import ScenarioWrapper
    from sumocr.interface.sumo_simulation import SumoSimulation

    folder_scenarios = os.path.abspath(
        '/home/zxc/Downloads/competition_scenarios_new/interactive/')
    # folder_scenarios = os.path.abspath(
    #     '/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/scenarios_cr_competition/competition_scenarios_new/interactive/')
    # name_scenario = "DEU_Frankfurt-4_2_I-1"  # 交叉口测试场景
    name_scenario = "DEU_Frankfurt-95_2_I-1"  # 直道测试场景
    interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    #
    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    num_of_steps = conf.simulation_steps

    sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)

    #
    current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
    ego_vehicles = sumo_sim.ego_vehicles
    ego_vehicle = list(ego_vehicles.values())[0]

    # f=open('variables.pkl', 'rb')
    # current_scenario,planning_problem, lanelet_route, ego_vehicle  = pickle.load(f)
    # f.close()

    # ====== plug in your motion planner here
    # ====== paste in simulations

    # generate a CR planner
    main_planner = InteractiveCRPlanner(current_scenario, ego_vehicle.current_state)
    last_action = []
    is_new_action_needed = True
    next_state, last_action, is_new_action_needed = main_planner.planning(current_scenario,
                                                                          planning_problem,
                                                                          ego_vehicle,
                                                                          sumo_sim.current_time_step,
                                                                          last_action,
                                                                          is_new_action_needed)

    # ====== paste in simulations
    # ====== end of motion planner
    next_state.time_step = 1
    trajectory_ego = [next_state]
    ego_vehicle.set_planned_trajectory(trajectory_ego)
    sumo_sim.simulate_step()
