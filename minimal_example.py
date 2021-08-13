""""
Minimal example to simulate interactive scenarios
"""
__author__ = "Edmond Irani Liu"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.5"
__maintainer__ = "Edmond Irani Liu"
__email__ = "edmond.irani@tum.de"
__status__ = "Integration"


import os, sys
sys.path.append(os.path.join(os.getcwd(), "./"))
import matplotlib as mpl

mpl.use('TkAgg')

from CR_tools.simulations import simulate_without_ego, simulate_with_solution, simulate_with_planner
from CR_tools.utility import save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction


def main():

    # folder_scenarios = os.path.join(os.path.dirname(__file__), "interactive_scenarios")
    folder_scenarios = '/home/thicv/codes/commonroad/commonroad-scenarios/scenarios/scenarios_cr_competition/competition_scenarios_new/interactive'
    # name_scenario = "DEU_Frankfurt-95_2_I-1"
    name_scenario =  'DEU_Frankfurt-4_2_I-1'
    path_scenario = os.path.join(folder_scenarios, name_scenario)


    # for simulation with a given solution trajectory
    # name_solution = "solution_KS1:TR1:DEU_Frankfurt-34_11_I-1:2020a"

    path_solutions = os.path.join(os.path.dirname(__file__), "solutions")
    # solution = CommonRoadSolutionReader.open(os.path.join(path_solutions, name_solution + ".xml"))

    # path to store output GIFs
    path_videos = "../outputs/videos/"

    # path to store simulated scenarios
    path_scenarios_simulated = "../outputs/simulated_scenarios/"

    vehicle_type = VehicleType.FORD_ESCORT
    vehicle_model = VehicleModel.KS
    cost_function = CostFunction.TR1

    # simulation with plugged-in motion planner
    scenario_with_planner, pps, ego_vehicles = simulate_with_planner(interactive_scenario_path=path_scenario,
                                                                        output_folder_path=path_videos,
                                                                        create_video=True)
    if scenario_with_planner:
        # write simulated scenario to file
        fw = CommonRoadFileWriter(scenario_with_planner, pps)
        fw.write_to_file(f"{path_scenarios_simulated}{name_scenario}_planner.xml", OverwriteExistingFile.ALWAYS)

        # saves trajectory to solution file
        save_solution(scenario_with_planner, pps, ego_vehicles,
                        vehicle_type,
                        vehicle_model,
                        cost_function,
                        path_solutions, overwrite=True)



if __name__ == '__main__':
    main()
