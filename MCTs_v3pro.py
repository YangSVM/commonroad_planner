# 该版本写于0711，2,1版本进一步考虑了地图信息输入（总车道数？目标车道和位置？），和多辆障碍车的输入。
# 外部输入信息：详细接口定义，见接口说明文档

from __future__ import division
import copy
import time
import math
import random
from copy import deepcopy
import numpy
import time

def output(state,action,speedLimit):
    state_out = copy.deepcopy(state) 
    t = 6
    print('speedLimit',speedLimit)
    if action == 1:
        state_out[0] = state[0] - 1
        state_out[1] = t * state[2]
        state_out[2] = state[2]
    if action == 2:
        state_out[0] = state[0] + 1
        state_out[1] = t * state[2]
        state_out[2] = state[2]
    if action == 3:
        if speedLimit - state[2] >= 1 * t:
            state_out[1] = t * state[2] + 0.5 * t * t
            state_out[2] = state[2] + t
        else:
            accel = (speedLimit - state[2]) / t
            state_out[1] = t * state[2] + 0.5 * accel * t * t
            state_out[2] = state[2] + t * accel
    if action == 4:
        state_out[1] = t * state[2]
    if action == 5:
        state_out[1] = t * state[2] - 0.5 * t * t
        state_out[2] = state[2] - t
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

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()  # 在循环次数内，不断重复该循环。没有参数输入，节点状态在循环内更新
        else:
            for i in range(self.searchLimit):
                # print(i)
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
        self.termi = [0,0,0,0,0,0]
        for i in range (tar[0]):
            self.termi[i] = mapInfo[i][0][1]
        self.reward = 100
        self.edge = [75, 120, 300]
        self.lastlane = [b[0], 0]
        self.laststate = [b[1], b[2]]
        self.T = 0
        self.Tstep = 6

        self.num = len(obstacles)
        self.lane0 = numpy.zeros(shape=(15, 2))  # 建立5条车道的存储空间
        self.lane1 = numpy.zeros(shape=(15, 2))
        self.lane2 = numpy.zeros(shape=(15, 2))
        self.lane3 = numpy.zeros(shape=(15, 2))
        self.lane4 = numpy.zeros(shape=(15, 2))
        self.point = numpy.zeros(shape=5).astype(int)
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
                            8 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag1 = 1
                    if self.laststate[0] <= 5 + self.positions(self.T, self.lastlane[0] - 1, i)[0]:  # 不能并道之前在车后
                        flag1 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] - 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 10:
                        flag1 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] + 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag1 = 1
        lane = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane]:  # 换道前，不能距离原车道终点太近
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
                            8 + self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0]:  # 并道后在车前
                        flag2 = 1
                    if self.laststate[0] <= 5 + self.positions(self.T, self.lastlane[0] + 1, i)[0]:  # 不能并道前在车后
                        flag2 = 1
                else:  # 并道后在车后
                    aa = self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[1]
                    if (self.positions(self.T + self.Tstep, self.lastlane[0] + 1, i)[0] - (
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 10:
                        flag2 = 1  # 不能小于安全车距
        if self.lastlane[0] != self.target[0] - 1:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.target[1]:
                flag2 = 1
        lane = self.lastlane[0]
        if (self.laststate[0] + 0.5 * self.Tstep * self.laststate[1]) >= self.termi[lane]:  # 换道前，不能距离原车道终点太近
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
                        1] + 0.5 * self.Tstep * self.Tstep)) <= 0.1 * (
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
                            self.laststate[0] + self.Tstep * self.laststate[1])) <= 0.1 * (
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
                        1] - 0.5 * self.Tstep * self.Tstep)) <= 0.1 * (
                            self.laststate[1] * self.laststate[1] - aa * aa):
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
                    newState.lastlane[0] - self.target[0]) - 0.5 * (self.target[1] - newState.laststate[0])
        if action.act == 1:
            newState.reward = newState.reward - 100
        if action.act == 2:
            newState.reward = newState.reward - 100
        if action.act == 6:
            newState.reward = newState.reward - 100000
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
    state = [1, 101.89999999999849, 31.068532257370617]
    map = [3, 0, 1194.600000000009, 32]
    #obstacles = [[0, 100, 25], [0, 150, 20], [1, 180, 10], [1, 120, 20], [2, 80, 15], [3, 80, 10], [4, 80, 15]]
    obstacles =  [[  2.,          4.9,        29.88358094],
 [  0.,         99.8,        31.28789092],
 [  1.,        166.3,        28.0214067 ],
 [  0.,         37.6,        34.11789059],
 [  0.,        153.7,        29.02353197],
 [  2.,         98.5,        29.58975459],
 [  2.,         61.9,        29.365209  ]]
    mapInfo = [[[0.0, 46560.93503873471]], [[0.0, 46560.93503873471]], [[0.0, 792.1928512579925]]]
    initialState = NaughtsAndCrossesState(state, map, obstacles, mapInfo)
    searcher = mcts(iterationLimit=5000)  # 改变循环次数或者时间
    action = searcher.search(initialState=initialState)  # 一整个类都是其状态
    out = output(state, action.act, map[3])
    print(out)  # 包括三个信息：[车道，纵向距离的增量，纵向车速]

    # print(action.act)
    # end = time.time()
    # print(end - start)