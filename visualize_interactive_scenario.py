import os
from commonroad.common.file_reader import CommonRoadFileReader
from simulation.simulations import load_sumo_configuration
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.interface.sumo_simulation import SumoSimulation
from utils import plot_lanelet_network

folder_scenarios = "/home/zxc/Downloads/competition_scenarios_new/interactive/"
name_scenario = "DEU_Frankfurt-7_11_I-1"
interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

conf = load_sumo_configuration(interactive_scenario_path)
scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

ln = scenario.lanelet_network
plot_lanelet_network(ln)
