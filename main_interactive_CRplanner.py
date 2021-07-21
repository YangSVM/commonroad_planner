# this main function as the body of the interactive planner
# it takes the current state of the CR scenario
# and outputs the next state of the ego vehicle
import os
from commonroad.common.file_reader import CommonRoadFileReader
from route_planner import route_planner
from simulation.simulations import load_sumo_configuration
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.interface.sumo_simulation import SumoSimulation
from intersection_planner import IntersectionPlanner
import pickle


class InteractiveCRPlanner:
    lanelet_ego = None
    lanelet_state = None

    def __init__(self, current_scenario, state_current_ego):
        # get current scenario info. from CR
        self.scenario = current_scenario
        self.ego_state = state_current_ego
        self.lanelet_ego = None   # the lanelet which ego car is located in
        self.lanelet_state = None  # straight-going /incoming /in-intersection

    def check_state(self):
        """check if ego car is straight-going /incoming /in-intersection"""

        ln = self.scenario.lanelet_network
        # find current lanelet
        self.lanelet_ego = ln.find_lanelet_by_position([self.ego_state.position])

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


if __name__ == '__main__':
    folder_scenarios = os.path.abspath(
        '/home/zxc/Downloads/competition_scenarios_new/interactive/')
    name_scenario = "DEU_Frankfurt-7_11_I-1"
    interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # generate a global lanelet route from initial position to goal region
    route = route_planner(scenario, planning_problem)
    lanelet_route = route.list_ids_lanelets
    add_successor = scenario.lanelet_network.find_lanelet_by_id(lanelet_route[-1]).successor
    if add_successor:
        lanelet_route.append(add_successor[0])

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    num_of_steps = conf.simulation_steps

    sumo_sim = SumoSimulation()
    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)
    sumo_sim.simulate_step()

    current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
    ego_vehicles = sumo_sim.ego_vehicles
    ego_vehicle = list(ego_vehicles.values())[0]

    # save variables for debugging
    # f = open('variables.pkl', 'wb')
    # pickle.dump([current_scenario, planning_problem,
    # lanelet_route, ego_vehicle], f)
    # f.close()

    # get variables for bugging
    # f = open('store.pkl', 'rb')
    # obj = pickle.load(f)
    # f.close()

    # generate a CR planner
    planner = InteractiveCRPlanner(current_scenario, ego_vehicle.current_state)

    # check state 1:straight-going /2:incoming /3:in-intersection
    planner.check_state()

    # send to sub planner according to current lanelet state
    if planner.lanelet_state == 1:

        # === insert straight-going planner here
        next_state = MCTs_CRv3(current_scenario, planning_problem, lanelet_route, ego_vehicle)
        # === end of straight-going planner

    elif planner.lanelet_state == 2 or planner.lanelet_state == 3:

        # === insert intersection planner here
        ip = IntersectionPlanner(current_scenario, planning_problem, lanelet_route, ego_vehicle)
        next_state = ip.planner()
        # === end of intersection planner
