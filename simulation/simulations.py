"""
SUMO simulation specific helper methods
"""

__author__ = "Peter Kocsis, Edmond Irani Liu"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Edmond Irani Liu"
__email__ = "edmond.irani@tum.de"
__status__ = "Integration"

import copy
import os
import pickle
from enum import unique, Enum
from math import sin, cos
from typing import Tuple, Dict, Optional

import numpy as np
from sumocr.sumo_config.default import DefaultConfig
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.visualization.video import create_video
from sumocr.sumo_docker.interface.docker_interface import SumoInterface

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.solution import Solution
from commonroad.common.file_reader import CommonRoadFileReader

from main_interactive_CRplanner import InteractiveCRPlanner
import matplotlib.pyplot as plt
from commonroad.visualization.draw_dispatch_cr import draw_object

@unique
class SimulationOption(Enum):
    WITHOUT_EGO = "_without_ego"
    MOTION_PLANNER = "_planner"
    SOLUTION = "_solution"


def simulate_scenario(mode: SimulationOption,
                      conf: DefaultConfig,
                      scenario_wrapper: ScenarioWrapper,
                      scenario_path: str,
                      num_of_steps: int = None,
                      planning_problem_set: PlanningProblemSet = None,
                      solution: Solution = None,
                      use_sumo_manager: bool = False) -> Tuple[Scenario, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with specified mode

    :param mode: 0 = without ego, 1 = with plugged in planner, 2 = with solution trajectory
    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param scenario_path: path to the interactive scenario folder
    :param num_of_steps: number of steps to simulate
    :param planning_problem_set: planning problem set of the scenario
    :param solution: solution to the planning problem
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """

    if num_of_steps is None:
        num_of_steps = conf.simulation_steps

    sumo_interface = None
    if use_sumo_manager:
        sumo_interface = SumoInterface(use_docker=True)
        sumo_sim = sumo_interface.start_simulator()

        sumo_sim.send_sumo_scenario(conf.scenario_name,
                                    scenario_path)
    else:
        sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)

    if mode is SimulationOption.WITHOUT_EGO:
        # simulation without ego vehicle
        for step in range(num_of_steps):
            # set to dummy simulation
            sumo_sim.dummy_ego_simulation = True
            sumo_sim.simulate_step()

    elif mode is SimulationOption.MOTION_PLANNER:
        # simulation with plugged in planner

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles
            for step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles

                # retrieve the CommonRoad scenario at the current time step, e.g. as an input for a predicition module
                current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
                for idx, ego_vehicle in enumerate(ego_vehicles.values()):
                    # retrieve the current state of the ego vehicle
                    state_current_ego = ego_vehicle.current_state

                    # ====== plug in your motion planner here
                    # example motion planner which decelerates to full stop
                    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
                    main_planner = InteractiveCRPlanner(current_scenario, ego_vehicle.current_state)
                    last_action = []
                    is_new_action_needed = True
                    next_state, last_action, is_new_action_needed = main_planner.planning(current_scenario,
                                                                                        planning_problem,
                                                                                        ego_vehicle,
                                                                                        sumo_sim.current_time_step,
                                                                                        last_action,
                                                                                        is_new_action_needed)
                    # plt.clf()
                    # draw_parameters = {
                    #     'time_begin': sumo_sim.current_time_step,
                    #     'scenario':
                    #         {'dynamic_obstacle': {'show_label': True, },
                    #         'lanelet_network': {'lanelet': {'show_label': False, }, },
                    #         },
                    # }

                    # draw_object(current_scenario, draw_params=draw_parameters)
                    # draw_object(planning_problem_set)
                    # plt.gca().set_aspect('equal')
                    # plt.pause(0.001)
                    # next_state = copy.deepcopy(state_current_ego)
                    # next_state.steering_angle = 0.0
                    # a = -5.0
                    # dt = 0.1
                    # if next_state.velocity > 0:
                    #     v = next_state.velocity
                    #     x, y = next_state.position
                    #     o = next_state.orientation

                    #     next_state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
                    #     next_state.velocity += a * dt
                    # ====== end of motion planner

                    # update the ego vehicle with new trajectory with only 1 state for the current step
                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using sumo_docker
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        run_simulation()

    elif mode is SimulationOption.SOLUTION:
        # simulation with given solution trajectory

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles

            for time_step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles
                for idx_ego, ego_vehicle in enumerate(ego_vehicles.values()):
                    # update the ego vehicles with solution trajectories
                    trajectory_solution = solution.planning_problem_solutions[idx_ego].trajectory
                    next_state = copy.deepcopy(trajectory_solution.state_list[time_step])

                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using SUMO Manager
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        check_trajectories(solution, planning_problem_set, conf)
        run_simulation()

    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()
    if use_sumo_manager:
        sumo_interface.stop_simulator()

    ego_vechicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                         ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    return simulated_scenario, ego_vechicles



def simulate_without_ego(interactive_scenario_path: str,
                         output_folder_path: str = None,
                         create_video: bool = False,
                         use_sumo_manager: bool = False) -> Tuple[Scenario, PlanningProblemSet]:
    """
    Simulates an interactive scenario without ego vehicle

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :return: Tuple of the simulated scenario and the planning problem set
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    # simulation without ego vehicle
    simulated_scenario_without_ego, _ = simulate_scenario(SimulationOption.WITHOUT_EGO, conf,
                                                          scenario_wrapper,
                                                          interactive_scenario_path,
                                                          num_of_steps=conf.simulation_steps,
                                                          planning_problem_set=planning_problem_set,
                                                          solution=None,
                                                          use_sumo_manager=use_sumo_manager)
    simulated_scenario_without_ego.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(simulated_scenario_without_ego, output_folder_path, planning_problem_set,
                                    {}, SimulationOption.WITHOUT_EGO.value)

    return simulated_scenario_without_ego, planning_problem_set


def simulate_with_solution(interactive_scenario_path: str,
                           output_folder_path: str = None,
                           solution: Solution = None,
                           create_video: bool = False,
                           use_sumo_manager: bool = False,
                           create_ego_obstacle: bool = False) -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a given solution

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param solution: solution to the planning problem
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles as the ego vehicles
    :return: Tuple of the simulated scenario and the planning problem set
    """
    if not isinstance(solution, Solution):
        raise Exception("Solution to the planning problem is not given.")

    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    scenario_with_solution, ego_vehicles = simulate_scenario(SimulationOption.SOLUTION, conf,
                                                                       scenario_wrapper,
                                                                       interactive_scenario_path,
                                                                       num_of_steps=conf.simulation_steps,
                                                                       planning_problem_set=planning_problem_set,
                                                                       solution=solution,
                                                                       use_sumo_manager=use_sumo_manager)
    scenario_with_solution.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_solution, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.SOLUTION.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_solution.add_objects(obstacle_ego)

    return scenario_with_solution, planning_problem_set, ego_vehicles


def simulate_with_planner(interactive_scenario_path: str,
                          output_folder_path: str = None,
                          create_video: bool = False,
                          use_sumo_manager: bool = False,
                          create_ego_obstacle: bool = False) \
        -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles from the planned trajectories as the ego vehicles
    :return: Tuple of the simulated scenario, planning problem set, and list of ego vehicles
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    scenario_with_planner, ego_vehicles = simulate_scenario(SimulationOption.MOTION_PLANNER, conf,
                                                                      scenario_wrapper,
                                                                      interactive_scenario_path,
                                                                      num_of_steps=conf.simulation_steps,
                                                                      planning_problem_set=planning_problem_set,
                                                                      use_sumo_manager=use_sumo_manager)
    scenario_with_planner.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_planner, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.MOTION_PLANNER.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_planner.add_objects(obstacle_ego)

    return scenario_with_planner, planning_problem_set, ego_vehicles


def load_sumo_configuration(interactive_scenario_path: str) -> DefaultConfig:
    with open(os.path.join(interactive_scenario_path, "simulation_config.p"), "rb") as input_file:
        conf = pickle.load(input_file)

    return conf


def check_trajectories(solution: Solution, pps: PlanningProblemSet, config: DefaultConfig):
    assert len(set(solution.planning_problem_ids) - set(pps.planning_problem_dict.keys())) == 0, \
        f"Provided solution trajectories with IDs {solution.planning_problem_ids} don't match " \
        f"planning problem IDs{list(pps.planning_problem_dict.keys())}"

    for s in solution.planning_problem_solutions:
        if s.trajectory.final_state.time_step < config.simulation_steps:
            raise ValueError(f"The simulation requires {config.simulation_steps}"
                             f"states, but the solution only provides"
                             f"{s.trajectory.final_state.time_step} time steps!")


def create_video_for_simulation(scenario_with_planner: Scenario, output_folder_path: str,
                                planning_problem_set: PlanningProblemSet,
                                ego_vehicles: Optional[Dict[int, EgoVehicle]],
                                suffix: str, follow_ego: bool = True):
    """Creates the mp4 animation for the simulation result."""
    if not output_folder_path:
        print("Output folder not specified, skipping mp4 generation.")
        return

    # create list of planning problems and trajectories
    list_planning_problems = []

    # create mp4 animation
    create_video(scenario_with_planner,
                 output_folder_path,
                 planning_problem_set=planning_problem_set,
                 trajectory_pred=ego_vehicles,
                 follow_ego=follow_ego,
                 suffix=suffix)
