# 该版本写于0711，2,1版本进一步考虑了地图信息输入（总车道数？目标车道和位置？），和多辆障碍车的输入。
# 0827修改：增加针对第一步的校验，当无解时进入行为5（lattice会做跟车）
# 外部输入信息：详细接口定义，见接口说明文档

from __future__ import division
import time
import math
import random
from copy import deepcopy
import numpy

import time
PLANNING_HORIZON = 5


class checker():
    def __init__(self, b, tar, obstacles, mapInfo):
        self.board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 表示一个具体的棋面
        self.currentPlayer = 1
        self.laststep = [2, 0]
        self.map = tar[0]
        self.target = [tar[1], tar[2]]
        self.speedLimit = tar[3]
        self.start = [0, 0, 0, 0, 0, 0]
        self.termi = [0,0,0,0,0,0]
        for i in range (tar[0]):
            self.termi[i] = mapInfo[i][0][1]
            self.start[i] = mapInfo[i][0][0]
        self.reward = 100
        self.edge = [75, 120, 300]
        self.lastlane = [b[0], 0]
        self.laststate = [b[1], b[2]]
        self.T = 0
        self.Tstep = PLANNING_HORIZON
        self.num = len(obstacles)
        self.lane0 = numpy.zeros(shape=(30, 2))  # 建立6条车道的存储空间
        self.lane1 = numpy.zeros(shape=(30, 2))
        self.lane2 = numpy.zeros(shape=(30, 2))
        self.lane3 = numpy.zeros(shape=(30, 2))
        self.lane4 = numpy.zeros(shape=(30, 2))
        self.lane5 = numpy.zeros(shape=(30, 2))
        self.point = numpy.zeros(shape=6).astype(int)
        for i in range(self.num):
            if obstacles[i][0] == 0:
                self.lane0[self.point[0]][0] = obstacles[i][1]
                self.lane0[self.point[0]][1] = obstacles[i][2]
                self.point[0] = self.point[0] + 1
            if obstacles[i][0] == 1:
                self.lane1[self.point[1]][0] = obstacles[i][1]
                self.lane1[self.point[1]][1] = obstacles[i][2]
                self.point[1] = self.point[1] + 1
            if obstacles[i][0] == 2:
                self.lane2[self.point[2]][0] = obstacles[i][1]
                self.lane2[self.point[2]][1] = obstacles[i][2]
                self.point[2] = self.point[2] + 1
            if obstacles[i][0] == 3:
                self.lane3[self.point[3]][0] = obstacles[i][1]
                self.lane3[self.point[3]][1] = obstacles[i][2]
                self.point[3] = self.point[3] + 1
            if obstacles[i][0] == 4:
                self.lane4[self.point[4]][0] = obstacles[i][1]
                self.lane4[self.point[4]][1] = obstacles[i][2]
                self.point[4] = self.point[4] + 1
            if obstacles[i][0] == 5:
                self.lane5[self.point[5]][0] = obstacles[i][1]
                self.lane5[self.point[5]][1] = obstacles[i][2]
                self.point[5] = self.point[5] + 1

    def positions(self, t, vehicle_id, num):
        vehicleP = 0
        vehicleV = 0
        if vehicle_id == 0:
            vehicleP = self.lane0[num][0] + self.lane0[num][1] * t
            vehicleV = self.lane0[num][1]
        if vehicle_id == 1:
            vehicleP = self.lane1[num][0] + self.lane1[num][1] * t
            vehicleV = self.lane1[num][1]
        if vehicle_id == 2:
            vehicleP = self.lane2[num][0] + self.lane2[num][1] * t
            vehicleV = self.lane2[num][1]
        if vehicle_id == 3:
            vehicleP = self.lane3[num][0] + self.lane3[num][1] * t
            vehicleV = self.lane3[num][1]
        if vehicle_id == 4:
            vehicleP = self.lane4[num][0] + self.lane4[num][1] * t
            vehicleV = self.lane4[num][1]
        if vehicle_id == 5:
            vehicleP = self.lane5[num][0] + self.lane5[num][1] * t
            vehicleV = self.lane5[num][1]
        return (vehicleP, vehicleV)

    def checkPossibleActions(self):  # 输入：node.state
        possibleActions = []
        Flag = 0
        flag1 = 0
        if self.lastlane[0] <= 0:  # 不能在最上
            flag1 = 1
        else:
            lane = self.lastlane[0] - 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[0]:  # 并道后在车前
                    if (self.laststate[0] + self.Tstep * self.laststate[1]) < \
                            5 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag1 = 1
                    if self.laststate[0] <= 3 + self.positions(self.T, self.lastlane[0] - 1, i)[0]:  # 不能并道之前在车后
                        flag1 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 5:
                        flag1 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] + 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag1 = 1
        lane0 = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane0]:  # 换道前，不能距离原车道终点太近
            flag1 = 1
        if self.laststate[0] <= self.start[lane0-1]:  # 换道前，不能还没到上方车道的起始
            flag1 = 1
        if flag1 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=1))

        # 在考虑向下并线
        flag2 = 0
        if self.lastlane[0] >= self.map - 1:  # 不能在最下
            flag2 = 1
        else:
            lane = self.lastlane[0] + 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                    if (self.laststate[0] + self.Tstep * self.laststate[1]) < \
                            5 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag2 = 1
                    if self.laststate[0] <= 3 + self.positions(self.T, self.lastlane[0] + 1, i)[0]:  # 不能并道前在车后
                        flag2 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 5:
                        flag2 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] - 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag2 = 1
        lane0 = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane0]:  # 换道前，不能距离原车道终点太近
            flag2 = 1
        if self.laststate[0] <= self.start[lane0+1]:  # 换道前，不能还没到上方车道的起始
            flag2 = 1
        if flag2 == 0:  # 满足所有条件时，可以向下换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=2))

        # 下面考虑直行--加速（+1 m/s2）
        flag3 = 0
        if self.laststate[1] >= self.speedLimit:
            flag3 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag3 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag3 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag3 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[
                        1] + 0.5 * self.Tstep * self.Tstep)) <= 0.05 * (
                            self.laststate[1] * self.laststate[1] - aa * aa):
                        flag3 = 1  # 不能小于安全车距
        if flag3 == 0:  # 满足所有条件时，可以加速
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=3))

        # 下面考虑直行--匀速（+0 m/s2）
        flag4 = 0
        if self.laststate[1] > 100000:
            flag4 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1]
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag4 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag4 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag4 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 0.05 * (
                            self.laststate[1] * self.laststate[1] - aa * aa):
                        flag4 = 1  # 不能小于安全车距
        if flag4 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=4))

        # 下面考虑直行--减速（-1 m/s2）
        flag5 = 0
        if self.laststate[1] < 3:
            flag5 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag5 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag5 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag5 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[
                        1] - 0.5 * self.Tstep * self.Tstep)) <= 0.05 * (
                            (self.laststate[1] - self.Tstep) * (self.laststate[1] - self.Tstep) - aa * aa):
                        flag5 = 1  # 不能小于安全车距
        if flag5 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=5))

        if possibleActions == []:
            Flag = 1
        #print('Flag:',Flag)
        return Flag  # 输出为所有可能落子位置的集合

def findFrontVechicle(position, lane):
    FrontVehicle = 0
    state_FrontVehicle = [0,0]
    count = 0
    for iVeh in range(30):
        if lane[iVeh][0] < position:
            count = iVeh
            FrontVehicle = count
            break
    if count == 0:
        state_FrontVehicle = [1000000, 100000]
    else:
        state_FrontVehicle = lane[FrontVehicle-1]
    #print('FrontVehicle',state_FrontVehicle)
    return state_FrontVehicle

def findTargetLane(lane00,lane01,lane02,lane03,lane04,lane05,laneNum):
    if laneNum == 0:
        TargetLane = deepcopy(lane00)
    if laneNum == 1:
        TargetLane = deepcopy(lane01)
    if laneNum == 2:
        TargetLane = deepcopy(lane02)
    if laneNum == 3:
        TargetLane = deepcopy(lane03)
    if laneNum == 4:
        TargetLane = deepcopy(lane04)
    if laneNum == 5:
        TargetLane = deepcopy(lane05)
    return TargetLane

def output(state,action,speedLimit,obstacles):
    state_out = [0,0,0]
    state_out[0] = state[0]
    state_out[1] = state[1]
    state_out[2] = state[2]
    number = len(obstacles)
    lane00 = numpy.zeros(shape=(30, 2))  # 建立6条车道的存储空间
    lane01 = numpy.zeros(shape=(30, 2))
    lane02 = numpy.zeros(shape=(30, 2))
    lane03 = numpy.zeros(shape=(30, 2))
    lane04 = numpy.zeros(shape=(30, 2))
    lane05 = numpy.zeros(shape=(30, 2))
    pointer = numpy.zeros(shape=6).astype(int)
    for i in range(number):
        if obstacles[i][0] == 0:
            lane00[pointer[0]][0] = obstacles[i][1]
            lane00[pointer[0]][1] = obstacles[i][2]
            pointer[0] = pointer[0] + 1
        if obstacles[i][0] == 1:
            lane01[pointer[1]][0] = obstacles[i][1]
            lane01[pointer[1]][1] = obstacles[i][2]
            pointer[1] = pointer[1] + 1
        if obstacles[i][0] == 2:
            lane02[pointer[2]][0] = obstacles[i][1]
            lane02[pointer[2]][1] = obstacles[i][2]
            pointer[2] = pointer[2] + 1
        if obstacles[i][0] == 3:
            lane03[pointer[3]][0] = obstacles[i][1]
            lane03[pointer[3]][1] = obstacles[i][2]
            pointer[3] = pointer[3] + 1
        if obstacles[i][0] == 4:
            lane04[pointer[4]][0] = obstacles[i][1]
            lane04[pointer[4]][1] = obstacles[i][2]
            pointer[4] = pointer[4] + 1
        if obstacles[i][0] == 5:
            lane05[pointer[5]][0] = obstacles[i][1]
            lane05[pointer[5]][1] = obstacles[i][2]
            pointer[5] = pointer[5] + 1
    lane00 = sorted(lane00, key=lambda s: s[0], reverse=True)  # 基于位置，将各车道车辆从前之后排序，更新列表
    lane01 = sorted(lane01, key=lambda s: s[0], reverse=True)
    lane02 = sorted(lane02, key=lambda s: s[0], reverse=True)
    lane03 = sorted(lane03, key=lambda s: s[0], reverse=True)
    lane04 = sorted(lane04, key=lambda s: s[0], reverse=True)
    lane05 = sorted(lane05, key=lambda s: s[0], reverse=True)
    t = PLANNING_HORIZON                                       # 调节每步时间（用于目标位置输出）（第1处，共2处）
    if action == 1:
        state_out[0] = state[0] - 1
        state_out[1] = t * state[2]
        state_out[2] = state[2]
        targetLane0 = findTargetLane(lane00, lane01, lane02, lane03, lane04, lane05, state[0])
        frontVehicle0 = findFrontVechicle(state[1], targetLane0)
        targetLane1 = findTargetLane(lane00, lane01, lane02, lane03, lane04, lane05, state[0]-1)
        frontVehicle1 = findFrontVechicle(state[1], targetLane1)
        if speedLimit - state[2] >= 1 * t:
            front0_ss = frontVehicle0[0] + frontVehicle0[1] * t
            front1_ss = frontVehicle1[0] + frontVehicle1[1] * t
            ego_ss = state[1] + t * state[2] + 0.5 * 1 * t * t
            if front0_ss > ego_ss - 3 * state[2]:                    # 参数：要求原车道前车在t时间后，跑到安全位置
                if (front1_ss - ego_ss) >= 5:                       # 参数：要求行为结束后，距离目标车道前车5m
                    state_out[1] = t * state[2] + 0.5 * 1 * t * t
                    state_out[2] = state[2] + 1 * t
                if (front1_ss - ego_ss) < 5:
                    acc2 = (front1_ss - 5 - t * state[2] - state[1]) / (0.5 * t * t)
                    state_out[1] = t * state[2] + 0.5 * acc2 * t * t
                    state_out[2] = state[2] + acc2 * t
    if action == 2:
        state_out[0] = state[0] + 1
        state_out[1] = t * state[2]
        state_out[2] = state[2]
        targetLane0 = findTargetLane(lane00, lane01, lane02, lane03, lane04, lane05, state[0])
        frontVehicle0 = findFrontVechicle(state[1], targetLane0)
        targetLane1 = findTargetLane(lane00, lane01, lane02, lane03, lane04, lane05, state[0] + 1)
        frontVehicle1 = findFrontVechicle(state[1], targetLane1)
        if speedLimit - state[2] >= 1 * t:
            front0_ss = frontVehicle0[0] + frontVehicle0[1] * t
            front1_ss = frontVehicle1[0] + frontVehicle1[1] * t
            ego_ss = state[1] + t * state[2] + 0.5 * 1 * t * t
            if front0_ss > ego_ss - 3*state[2]:  # 参数：要求原车道前车在t时间后，跑到安全位置
                if (front1_ss - ego_ss) >= 5:  # 参数：要求行为结束后，距离目标车道前车5m
                    state_out[1] = t * state[2] + 0.5 * 1 * t * t
                    state_out[2] = state[2] + 1 * t
                if (front1_ss - ego_ss) < 5:
                    acc2 = (front1_ss - 5 - t * state[2] - state[1]) / (0.5 * t * t)
                    state_out[1] = t * state[2] + 0.5 * acc2 * t * t
                    state_out[2] = state[2] + acc2 * t
    if action == 3:
        targetLane = findTargetLane(lane00,lane01,lane02,lane03,lane04,lane05,state[0])
        frontVehicle = findFrontVechicle(state[1],targetLane)
        if speedLimit - state[2] >= 1 * t:
            state_out[1] = t * state[2] + 0.5 * t * t
            state_out[2] = state[2] + t
            if speedLimit - state[2] >= 2 * t:                    # 考虑能否用2加速
                ego_ss = state[1] + t * state[2] + 0.5 * 2 * t * t
                front_ss = frontVehicle[0] + frontVehicle[1] * t
                if (front_ss - ego_ss) >= 5:                       # 参数：要求行为结束后距离前车5m
                    state_out[1] = t * state[2] + 0.5 * 2 * t * t
                    state_out[2] = state[2] + 2 * t
                if (front_ss - ego_ss) < 5:
                    acc2 = (front_ss - 5 - t * state[2] - state[1]) / (0.5 * t * t)
                    state_out[1] = t * state[2] + 0.5 * acc2 * t * t
                    state_out[2] = state[2] + acc2 * t
                if speedLimit - state[2] >= 3 * t:                 # 考虑能否用3加速
                    ego_ss = state[1] + t * state[2] + 0.5 * 3 * t * t
                    if (front_ss - ego_ss) >= 8:                   # 参数：要求行为结束后距离前车8m
                        state_out[1] = t * state[2] + 0.5 * 3 * t * t
                        state_out[2] = state[2] + 3 * t
                    if (front_ss - ego_ss) < 8:
                        acc3 = (front_ss - 8 - t * state[2] - state[1]) / (0.5 * t * t)
                        state_out[1] = t * state[2] + 0.5 * acc3 * t * t
                        state_out[2] = state[2] + acc3 * t
        else:
            accel = (speedLimit - state[2]) / t
            state_out[1] = t * state[2] + 0.5 * accel * t * t
            state_out[2] = state[2] + t * accel
    if action == 4:
        state_out[1] = t * state[2]
    if action == 5:
        if state[2] - t > 0:
            state_out[1] = t * state[2] - 0.5 * t * t
            state_out[2] = state[2] - t
        else:
            dccel = state[2] / t
            state_out[1] = t * state[2] - 0.5 * dccel * t * t
            state_out[2] = state[2] - t * dccel
    if action == 6:
        state_out = [0,0,0]
    return state_out

def randomPolicy(state):
    while not state.isTerminal():  # 从expand生成棋面，rollout直到棋局结束
        a = state.lastlane[0]
        if a > state.target[0]:
            action = Action(player=state.currentPlayer, x=1, y=1, act=1)  # 基于rollout过程当前棋面，随机产生行为（落子位置）
            state = state.takeAction(action)
        if a < state.target[0]:
            action = Action(player=state.currentPlayer, x=1, y=1, act=2)  # 基于rollout过程当前棋面，随机产生行为（落子位置）
            state = state.takeAction(action)
        else:
            action = Action(player=state.currentPlayer, x=1, y=1, act=7)  # 基于rollout过程当前棋面，随机产生行为（落子位置）
            state = state.takeAction(action)
    return state.reward  # 这个state是rollout出来的终局。此处，返回值包含+1，-1，和False（0）


def transform(node):
    while not node.state.isTerminal():
        bestChild = mcts.getBestChild(mcts, node, 0)
        action = (action for action, node in node.children.items() if
                  node is bestChild).__next__()  # Children需要改为行为，而非lanelet
        node = bestChild
        print(action)
        print(node.state.laststate)
        print(node.state.lastlane)
        if len(node.children.items()) == 0:
            return 0


class treeNode():  # 一个节点是一个棋面，包含以下所有信息
    def __init__(self, state, parent):
        self.state = state  # 很重要，通过state连接到圈叉游戏的类（包括棋面、玩家和一堆函数）
        self.isTerminal = state.isTerminal()  # 通过这种方法，将isTerminal从圈叉的类中连接到了treeNode类中
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):  # 当输出该棋面内容时，按照下方的格式输出内容
        s = []
        s.append("totalReward: %s" % (self.totalReward))
        s.append("numVisits: %d" % (self.numVisits))
        s.append("isTerminal: %s" % (self.isTerminal))
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)  # 从初始棋盘问题中提取根节点，其没有父节点

        # if self.limitType == 'time':
        #     timeLimit = time.time() + self.timeLimit / 1000
        #     while time.time() < timeLimit:
        #         self.executeRound()  # 在循环次数内，不断重复该循环。没有参数输入，节点状态在循环内更新
        # else:
        #     for i in range(self.searchLimit):
        #         # print(i)
        #         self.executeRound()
        timeLimit = time.time() + 2
        for i in range(self.searchLimit):
            #print(i)
            if time.time() < timeLimit:
                self.executeRound()

        self.node1 = self.root
        self.output = transform(self.node1)

        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)  # 每轮的节点选择都是从根节点（初始棋面）开始的。输出node：一个expand后的新棋面。
        # print(node.state.board)
        # print(node.state.laststep)
        # print(node.state.reward)
        reward = self.rollout(node.state)  # 基于该新棋面，rollout至结尾。当前对应了随机策略。输出rollout结束时的结果（+1、-1、0）。
        self.backpropogate(node, reward)  # 基于新棋面，更新路径上节点的value。

    def selectNode(self, node):
        while not node.isTerminal:  # 在抵达终局之前，不断执行下述循环。随着大循环进行，步数整体上升（原因：return在expand中）
            if node.isFullyExpanded:  # 如果所有可能子节点均被拓展过，则选择最好的子节点
                node = self.getBestChild(node, self.explorationConstant)  # 以最佳节点为中心，开始一轮新拓展（如果新点也完全拓展，则基于它再找）
            else:
                return self.expand(node)  # 每一次返回一个新棋面（节点）。expand后立即退出循环。
        return node  # 可见，总是输出一个expand后的新棋面

    def expand(self, node):
        actions = node.state.getPossibleActions()  # 所有可能位置，同一棋面下一样
        for action in actions:  # 这个循环只负责遍历，实际上一轮只输出一个action，原因是return的位置
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)  # 更新一个新棋面（状态，和父节点）
                node.children[action] = newNode  # 给当前节点增加一个子节点
                if len(actions) == len(node.children):  # 完全拓展判断
                    node.isFullyExpanded = True
                return newNode  # expand一个新节点后，就直接退出循环了。返回这个新棋面（节点）

    def backpropogate(self, node, reward):  # 基于父节点关系，反向更新路径上的信息。输入当前节点和基于它的rollout结果。
        while node is not None:
            node.numVisits += 1  # 更新当前节点的总访问次数
            node.totalReward = 0.5 * reward + 0.5 * node.totalReward  # 更细当前节点的reward
            node = node.parent  # 指向当前节点父节点（root是初始棋面）

    def getBestChild(self, node, explorationValue):  # 拓展过程，寻找最佳节点（当eV为0，则是纯贪心）
        bestValue = float("-inf")  # 初始化评价值和最佳节点
        bestNodes = []  # 最佳节点可能有多个，都放进去最后随机选择
        for child in node.children.values():  # 每一个child都对应一个棋面
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)  # 这个就是UCT算法了，不过这里还有棋手的问题
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)  # 如果有多个相同的，则随机选择一个子节点（棋面）


class NaughtsAndCrossesState():  # 连接到treeNode的state中
    def __init__(self, b, tar, obstacles, mapInfo):
        self.board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # 表示一个具体的棋面
        self.currentPlayer = 1
        self.laststep = [2, 0]
        self.map = tar[0]
        self.target = [tar[1], tar[2]]
        self.speedLimit = tar[3]
        self.start = [0, 0, 0, 0, 0, 0]
        self.termi = [0,0,0,0,0,0]
        for i in range (tar[0]):
            self.termi[i] = mapInfo[i][0][1]
            self.start[i] = mapInfo[i][0][0]
        self.reward = 100
        self.edge = [75, 120, 300]
        self.lastlane = [b[0], 0]
        self.laststate = [b[1], b[2]]
        self.T = 0
        self.Tstep = PLANNING_HORIZON                           # 调节每步时间（用于MCTS内部计算）（第2处，共2处）

        self.num = len(obstacles)
        self.lane0 = numpy.zeros(shape=(30, 2))  # 建立5条车道的存储空间
        self.lane1 = numpy.zeros(shape=(30, 2))
        self.lane2 = numpy.zeros(shape=(30, 2))
        self.lane3 = numpy.zeros(shape=(30, 2))
        self.lane4 = numpy.zeros(shape=(30, 2))
        self.lane5 = numpy.zeros(shape=(30, 2))
        self.point = numpy.zeros(shape=6).astype(int)
        for i in range(self.num):
            if obstacles[i][0] == 0:
                self.lane0[self.point[0]][0] = obstacles[i][1]
                self.lane0[self.point[0]][1] = obstacles[i][2]
                self.point[0] = self.point[0] + 1
            if obstacles[i][0] == 1:
                self.lane1[self.point[1]][0] = obstacles[i][1]
                self.lane1[self.point[1]][1] = obstacles[i][2]
                self.point[1] = self.point[1] + 1
            if obstacles[i][0] == 2:
                self.lane2[self.point[2]][0] = obstacles[i][1]
                self.lane2[self.point[2]][1] = obstacles[i][2]
                self.point[2] = self.point[2] + 1
            if obstacles[i][0] == 3:
                self.lane3[self.point[3]][0] = obstacles[i][1]
                self.lane3[self.point[3]][1] = obstacles[i][2]
                self.point[3] = self.point[3] + 1
            if obstacles[i][0] == 4:
                self.lane4[self.point[4]][0] = obstacles[i][1]
                self.lane4[self.point[4]][1] = obstacles[i][2]
                self.point[4] = self.point[4] + 1
            if obstacles[i][0] == 5:
                self.lane5[self.point[5]][0] = obstacles[i][1]
                self.lane5[self.point[5]][1] = obstacles[i][2]
                self.point[5] = self.point[5] + 1

    def getCurrentPlayer(self):  # 输入：node.state
        return self.currentPlayer

    def positions(self, t, vehicle_id, num):
        vehicleP = 0
        vehicleV = 0
        if vehicle_id == 0:
            vehicleP = self.lane0[num][0] + self.lane0[num][1] * t
            vehicleV = self.lane0[num][1]
        if vehicle_id == 1:
            vehicleP = self.lane1[num][0] + self.lane1[num][1] * t
            vehicleV = self.lane1[num][1]
        if vehicle_id == 2:
            vehicleP = self.lane2[num][0] + self.lane2[num][1] * t
            vehicleV = self.lane2[num][1]
        if vehicle_id == 3:
            vehicleP = self.lane3[num][0] + self.lane3[num][1] * t
            vehicleV = self.lane3[num][1]
        if vehicle_id == 4:
            vehicleP = self.lane4[num][0] + self.lane4[num][1] * t
            vehicleV = self.lane4[num][1]
        if vehicle_id == 5:
            vehicleP = self.lane5[num][0] + self.lane5[num][1] * t
            vehicleV = self.lane5[num][1]
        return (vehicleP, vehicleV)

    def getPossibleActions(self):  # 输入：node.state
        possibleActions = []
        flag1 = 0
        if self.lastlane[0] <= 0:  # 不能在最上
            flag1 = 1
        else:
            lane = self.lastlane[0] - 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[0]:  # 并道后在车前
                    if (self.laststate[0] + self.Tstep * self.laststate[1]) < \
                            5 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag1 = 1
                    if self.laststate[0] <= 3 + self.positions(self.T, self.lastlane[0] - 1, i)[0]:  # 不能并道之前在车后
                        flag1 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 5:
                        flag1 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] + 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag1 = 1
        lane0 = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane0]:  # 换道前，不能距离原车道的终点太近
            flag1 = 1
        if self.laststate[0] <= self.start[lane0-1]:  # 换道前，不能还没到上方车道的起始
            flag1 = 1
        if flag1 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=1))

        # 在考虑向下并线
        flag2 = 0
        if self.lastlane[0] >= self.map - 1:  # 不能在最下
            flag2 = 1
        else:
            lane = self.lastlane[0] + 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                    if (self.laststate[0] + self.Tstep * self.laststate[1]) < \
                            5 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag2 = 1
                    if self.laststate[0] <= 3 + self.positions(self.T, self.lastlane[0] + 1, i)[0]:  # 不能并道前在车后
                        flag2 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 5:
                        flag2 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] - 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag2 = 1
        lane0 = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane0]:  # 换道前，不能距离原车道终点太近
            flag2 = 1
        if self.laststate[0] <= self.start[lane0+1]:  # 换道前，不能还没到下方车道的起始
            flag2 = 1
        if flag2 == 0:  # 满足所有条件时，可以向下换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=2))

        # 下面考虑直行--加速（+1 m/s2）
        flag3 = 0
        if self.laststate[1] >= self.speedLimit:
            flag3 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag3 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag3 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag3 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[
                        1] + 0.5 * self.Tstep * self.Tstep)) <= 0.05 * (
                            self.laststate[1] * self.laststate[1] - aa * aa):
                        flag3 = 1  # 不能小于安全车距
        if flag3 == 0:  # 满足所有条件时，可以加速
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=3))

        # 下面考虑直行--匀速（+0 m/s2）
        flag4 = 0
        if self.laststate[1] > 100000:
            flag4 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1]
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag4 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag4 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag4 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 0.05 * (
                            self.laststate[1] * self.laststate[1] - aa * aa):
                        flag4 = 1  # 不能小于安全车距
        if flag4 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=4))

        # 下面考虑直行--减速（-1 m/s2）
        flag5 = 0
        if self.laststate[1] <= 0:
            flag5 = 1
        else:
            st = self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep
            lane = self.lastlane[0]
            if self.lastlane[0] != self.target[0]:  # 不能超过纵向距离约束（非目标lanelet）
                if st >= self.target[1]:
                    flag5 = 1
            if st >= self.termi[lane]:  # 本步结束后，不能超过所在车道终点
                flag5 = 1
            for i in range(self.point[lane]):
                if (self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep) >= \
                        self.positions(self.T + self.Tstep, self.lastlane[0], i)[0]:  # 行为完成后在障碍车之前
                    if self.laststate[0] <= self.positions(self.T, self.lastlane[0], i)[0]:  # 行为开始前，不能在障碍车之后
                        flag5 = 1
                else:  # 行为完成后，在障碍车后方
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0], i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0], i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[
                        1] - 0.5 * self.Tstep * self.Tstep)) <= 0.05 * (
                            (self.laststate[1] - self.Tstep) * (self.laststate[1] - self.Tstep) - aa * aa):
                        flag5 = 1  # 不能小于安全车距
        if flag5 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=5))

        if possibleActions == []:
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=6))
        # print(possibleActions)
        return possibleActions  # 输出为所有可能落子位置的集合

    def takeAction(self, action):  # node.state.takeAction(action)，这一步只更新棋面和棋手
        newState = deepcopy(self)

        newState.board[action.x][action.y] = 1
        newState.laststep = [action.x, action.y]
        newState.currentPlayer = self.currentPlayer

        if action.act == 1:
            newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[1]
            newState.laststate[1] = self.laststate[1]
            newState.lastlane[0] = self.lastlane[0] - 1
        if action.act == 2:
            newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[1]
            newState.laststate[1] = self.laststate[1]
            newState.lastlane[0] = self.lastlane[0] + 1
        if action.act == 3:
            if self.speedLimit - self.laststate[1] >= 1 * self.Tstep:
                newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[
                    1] + 0.5 * self.Tstep * self.Tstep
                newState.laststate[1] = self.laststate[1] + 1 * self.Tstep
                newState.lastlane[0] = self.lastlane[0]
            else:
                accel = (self.speedLimit - self.laststate[1]) / self.Tstep
                newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[
                    1] + 0.5 * accel * self.Tstep * self.Tstep
                newState.laststate[1] = self.laststate[1] + accel * self.Tstep
                newState.lastlane[0] = self.lastlane[0]
        if action.act == 4:
            newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[1]
            newState.laststate[1] = self.laststate[1]
            newState.lastlane[0] = self.lastlane[0]
        if action.act == 5:
            newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep
            newState.laststate[1] = self.laststate[1] - 1 * self.Tstep
            newState.lastlane[0] = self.lastlane[0]
        if action.act == 6:
            newState.laststate[0] = self.laststate[0]
            newState.laststate[1] = 0
            newState.lastlane[0] = self.lastlane[0]
        if action.act == 7:
            newState.laststate[0] = self.laststate[0] + self.Tstep * 10
            newState.laststate[1] = 20
            newState.lastlane[0] = self.lastlane[0]

        newState.T = self.T + self.Tstep
        newState.reward = self.reward - 10 - 1000 * (newState.lastlane[0] - self.target[0]) * (
                    newState.lastlane[0] - self.target[0]) - 0.5 * (self.target[1] + 200 - newState.laststate[0])
        if action.act == 1:
            newState.reward = newState.reward - 100
        if action.act == 2:
            newState.reward = newState.reward - 100
        if action.act == 3:
            newState.reward = newState.reward + 1000
        if action.act == 5:
            newState.reward = newState.reward - 25 * (self.target[1] + 200 - newState.laststate[0])
        if action.act == 6:
            newState.reward = newState.reward - 100000000
        return newState

    def isTerminal(self):  # 因为treeNode中定义了连接，输入node.isTerminal
        tag = False
        if self.lastlane[0] == self.target[0]:
            if self.laststate[0] > self.target[1]:
                tag = True
        return tag  # 最终输出0或1.要根据自己的要求修改逻辑


class Action():
    def __init__(self, player, x, y, act):
        self.player = player
        self.x = x
        self.y = y
        self.act = act

    def __str__(self):
        return str((self.act))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player, self.act))


if __name__ == "__main__":
    # start = time.time()
    state = [1, 616.2000000000716, 35.49808537142544]
    map = [3, 1, 1549.8999999996904, 43.333333333333336]
    # obstacles = [[0, 100, 25], [0, 150, 20], [1, 180, 10], [1, 120, 20], [2, 80, 15], [3, 80, 10], [4, 80, 15]]
    obstacles =  [[1.00000000e+00, 1.22230000e+03, 3.84660840e+01],
 [1.00000000e+00, 1.39900000e+02, 2.34867287e+01],
 [1.00000000e+00, 7.78000000e+01, 1.68350461e+01],
 [0.00000000e+00, 2.69000000e+01, 8.33863084e+00],
 [1.00000000e+00, 6.90000000e+00, 3.64456328e+00],
 [1.00000000e+00, 9.46500000e+02, 3.06818203e+01],
 [2.00000000e+00, 1.06750000e+03, 3.52290504e+01],
 [2.00000000e+00, 9.82700000e+02, 3.41876130e+01],
 [1.00000000e+00, 1.10400000e+03, 3.96644613e+01],
 [1.00000000e+00, 8.63100000e+02, 3.03480878e+01],
 [1.00000000e+00, 9.81300000e+02, 3.75637993e+01],
 [1.00000000e+00, 8.14400000e+02, 3.15314354e+01],
 [1.00000000e+00, 7.72600000e+02, 3.15792138e+01],
 [0.00000000e+00, 4.49900000e+02, 3.54025066e+01],
 [1.00000000e+00, 3.55700000e+02, 3.29370758e+01],
 [2.00000000e+00, 1.13970000e+03, 3.66337925e+01],
 [1.00000000e+00, 2.86800000e+02, 3.14619373e+01],
 [0.00000000e+00, 1.99800000e+02, 2.65687692e+01]]
    mapInfo =  [[[0.0, 467537.88854708296]], [[0.0, 467537.88854708296]], [[42628.51147501361, 467537.88854708296]]]
    actionChecker = checker(state, map, obstacles, mapInfo)
    flag = actionChecker.checkPossibleActions()
    if flag == 0:
        initialState = NaughtsAndCrossesState(state, map, obstacles, mapInfo)
        searcher = mcts(iterationLimit=5000)  # 改变循环次数或者时间
        action = searcher.search(initialState=initialState)  # 一整个类都是其状态
        out = output(state, action.act, map[3], obstacles)
    if flag == 1:
        print('第一步mcts无解，进入跟车')
        out = output(state, 5, map[3], obstacles)
    print(out)  # 包括三个信息：[车道，纵向距离的增量，纵向车速]
    print('state',state)
    print('map', map)
    print('obstacles',obstacles)
    print('mapInfo', mapInfo)

    # print(action.act)
    # end = time.time()
    # print(end - start)