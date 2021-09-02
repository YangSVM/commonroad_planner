import os.path

from commonroad.common.solution import Solution, CommonRoadSolutionWriter

# Load planner module
from iterator import scenario_iterator_interactive, scenario_iterator_non_interactive, _search_interactive_scenarios
from main_interactive_CRplanner import motion_planner_interactive


def save_solution(solution: Solution, path: str) -> None:
    """
    Save the given solution to the given path.
    """
    return CommonRoadSolutionWriter(solution).write_to_file(
        output_path=path,
        overwrite=True,
        pretty=True
    )


# Run Main Process
if __name__ == "__main__":
    scenario_dir = "/commonroad/scenarios"
    solution_dir = "/commonroad/solutions"

    # solve all non-interactive scenarios
    # for scenario, planning_problem_set in scenario_iterator_non_interactive(scenario_dir):
    #     print(f"Processing scenario {str(scenario.scenario_id)} ...")
    #     solution = motion_planner(scenario)
    #     save_solution(solution, solution_dir)

    # solve the second half of interactive scenarios
    interactive_paths = _search_interactive_scenarios(scenario_dir)
    n_interactive_scenarios = len(interactive_paths)
    last_half_scenario_path = interactive_paths[int(n_interactive_scenarios/2):]

    for scenario_path in last_half_scenario_path:
        print(f"Processing scenario {os.path.basename(scenario_path)} ...")
        try:
            solution = motion_planner_interactive(scenario_path)
            save_solution(solution, solution_dir)
        except:  
            print('-'*20,'cannot solve this scenario', scenario_path,'-'*20)
        else:
            print('-'*20,scenario_path, 'solved already','-'*20)
