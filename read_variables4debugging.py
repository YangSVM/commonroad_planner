# get variables ( after sumo_sim.simulate_step() ) for debugging
import pickle


# === beginning of codes in main
# # save variables for bugging
#     f = open('variables.pkl', 'wb')
#     pickle.dump([current_scenario, planning_problem, lanelet_route, ego_vehicle], f)
#     f.close()
# === end of codes in main

f = open('variables.pkl', 'rb')
obj = pickle.load(f)
f.close()