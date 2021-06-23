from commonroad_route_planner.route_planner import RoutePlanner


def route_planner(scenario, planning_problem):
    # ========== route planning =========== #
    # instantiate a route planner with the scenario and the planning problem
    rp = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    # plan routes, and save the routes in a route candidate holder
    candidate_holder = rp.plan_routes()
    route = candidate_holder.retrieve_best_route_by_orientation()
    return route
