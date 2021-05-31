from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os

# 当前路径
path_py = os.path.abspath('.')
# 文件名
id_scenario = 'USA_Peach-2_1_T-1'

path_scenario =os.path.join(path_py, id_scenario + '.xml')
# read in scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
# retrieve the first planning problem in the problem set
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# dynamic obstacle
obstacles = scenario.obstacles
n_obstacle = len(obstacles)
for i in range(n_obstacle):
    state = obstacles[i].state_at_time(5)                # 5是整数。代表第5个仿真步长
    position = state.position
    orientation = state.orientation
    vel = state.velocity
    a = state.acceleration
    print(state, position, orientation, vel, a)


# lanelets
lanelets = scenario.lanelet_network.lanelets
n_lanelet = len(lanelets)
for i in range(n_lanelet):
    lanelet = lanelets[i]
    # 属性。   
    print(lanelet.left_vertices, lanelet.lanelet_id, lanelet.predecessor, lanelet.successor, lanelet.adj_left, lanelet.adj_right, lanelet.center_vertices, lanelet.distance) 


# 终止状态
goal, initial_state = planning_problem.goal, planning_problem.initial_state

# 自车状态
# 在非交互式场景中。没有相关函数。自车状态完全由自己定义。目标是在其他车辆状态、地图已知情况下，给出符合模型约束的轨迹，所以没有相关接口
# 交互式场景中。如在simulation.py 的simulate_scenario函数中，state_current_ego = ego_vehicle.current_state
