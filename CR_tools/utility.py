from typing import Dict
import numpy as np
from IPython import display
from commonroad.common.solution import PlanningProblemSolution, Solution, CommonRoadSolutionWriter, VehicleType, \
    VehicleModel, CostFunction
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from sumocr.interface.ego_vehicle import EgoVehicle


def visualize_scenario_with_trajectory(scenario: Scenario,
                                       planning_problem_set: PlanningProblemSet,
                                       ego_vehicles: Dict[int, EgoVehicle] = None,
                                       discrete_time_step: bool = False,
                                       num_time_steps: int = None) -> None:
    if ego_vehicles is not None:
        ego_vehicles = [e.get_dynamic_obstacle() for _, e in ego_vehicles.items()]
    if not num_time_steps:
        if ego_vehicles:
            num_time_steps = ego_vehicles[0].prediction.final_time_step
        else:
            num_time_steps = 50

    # visualize scenario
    for i in range(0, num_time_steps):
        if not discrete_time_step:
            display.clear_output(wait=True)
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i})
        planning_problem_set.draw(rnd)
        if ego_vehicles:
            rnd.draw_list(ego_vehicles,
                          draw_params={'time_begin': i,
                                       'dynamic_obstacle': {'vehicle_shape': {"occupancy": {"shape": {"rectangle": {
                                              "facecolor": "green"}}}}}})
        rnd.render(show=True)


def save_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, ego_vehicles: Dict[int, EgoVehicle],
                  vehicle_type: VehicleType,
                  vehicle_model: VehicleModel,
                  cost_function: CostFunction,
                  output_path: str = './', overwrite: bool = False):
    """Saves the given trajectory as a solution to the planning problem"""

    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in planning_problem_set.planning_problem_dict
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=vehicle_type,
                                           vehicle_model=vehicle_model,
                                           cost_function=cost_function,
                                           trajectory=ego_vehicle.driven_trajectory.trajectory))

    solution = Solution(scenario.scenario_id, pps)

    # write solution
    csw = CommonRoadSolutionWriter(solution)
    csw.write_to_file(output_path=output_path, overwrite=overwrite)
    print("Trajectory saved to solution file.")


def distance_lanelet(center_line, s, p1, p2):
    """ 计算沿着道路中心线的路程. p2 - p1（正数说明p2在道路后方）
         直线的时候，保证是直线距离；曲线的时候，近似正确
    Args:
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    Return:

    """
    # 规范化格式。必须是numpy数组。并且m*2维，m是点的数量
    if type(center_line) is not np.ndarray:
        center_line = np.array(center_line)
    if center_line.shape[1] != 2:
        center_line = center_line.T
    if center_line.shape[0] == 2:
        print('distance_lanelet warning! may wrong size of center line input. check the input style ')

    d1 = np.linalg.norm(center_line - p1, axis=1)
    i1 = np.argmin(d1)
    d2 = np.linalg.norm(center_line - p2, axis=1)
    i2 = np.argmin(d2)

    return s[i2] - s[i1]