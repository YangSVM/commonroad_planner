from __future__ import division

import time
import math
import random
from copy import deepcopy

def randomPolicy(state):
    while not state.isTerminal():    #从expand生成棋面，rollout直到棋局结束
        a = state.lastlane[0]
        if a > 0:
            action = Action(player=state.currentPlayer, x=1, y=1, act=1)   #基于rollout过程当前棋面，随机产生行为（落子位置）
            state = state.takeAction(action)
        else:
            action = Action(player=state.currentPlayer, x=1, y=1, act=7)  # 基于rollout过程当前棋面，随机产生行为（落子位置）
            state = state.takeAction(action)
    return state.reward     #这个state是rollout出来的终局。此处，返回值包含+1，-1，和False（0）

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

class vehicles():
    def __init__(self, Vmatrix):
        self.state0 = Vmatrix[0]
        self.state1 = Vmatrix[1]
        self.state2 = Vmatrix[2]

    def positions(self, t, vehicle_id):
        vehicleP = 0
        vehicleV = 0
        if vehicle_id == 0:
            vehicleP = self.state0[0] + self.state0[1] * t
            vehicleV = self.state0[1]
        if vehicle_id == 1:
            vehicleP = self.state1[0] + self.state1[1] * t
            vehicleV = self.state1[1]
        if vehicle_id == 2:
            vehicleP = self.state2[0] + self.state2[1] * t
            vehicleV = self.state2[1]
        return (vehicleP, vehicleV)


class treeNode():               #一个节点是一个棋面，包含以下所有信息
    def __init__(self, state, parent):
        self.state = state                    # 很重要，通过state连接到圈叉游戏的类（包括棋面、玩家和一堆函数）
        self.isTerminal = state.isTerminal()  # 通过这种方法，将isTerminal从圈叉的类中连接到了treeNode类中
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):    #当输出该棋面内容时，按照下方的格式输出内容
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

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
        self.root = treeNode(initialState, None) #从初始棋盘问题中提取根节点，其没有父节点

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound() #在循环次数内，不断重复该循环。没有参数输入，节点状态在循环内更新
        else:
            for i in range(self.searchLimit):
                #print(i)
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
        node = self.selectNode(self.root)   #每轮的节点选择都是从根节点（初始棋面）开始的。输出node：一个expand后的新棋面。
        # print(node.state.board)
        # print(node.state.laststep)
        reward = self.rollout(node.state)   #基于该新棋面，rollout至结尾。当前对应了随机策略。输出rollout结束时的结果（+1、-1、0）。
        self.backpropogate(node, reward)    #基于新棋面，更新路径上节点的value。

    def selectNode(self, node):
        while not node.isTerminal:   #在抵达终局之前，不断执行下述循环。随着大循环进行，步数整体上升（原因：return在expand中）
            if node.isFullyExpanded: #如果所有可能子节点均被拓展过，则选择最好的子节点
                node = self.getBestChild(node, self.explorationConstant)  #以最佳节点为中心，开始一轮新拓展（如果新点也完全拓展，则基于它再找）
            else:
                return self.expand(node) #每一次返回一个新棋面（节点）。expand后立即退出循环。
        return node  #可见，总是输出一个expand后的新棋面

    def expand(self, node):
        actions = node.state.getPossibleActions() #所有可能位置，同一棋面下一样
        for action in actions:   #这个循环只负责遍历，实际上一轮只输出一个action，原因是return的位置
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)  #更新一个新棋面（状态，和父节点）
                node.children[action] = newNode  #给当前节点增加一个子节点
                if len(actions) == len(node.children):   #完全拓展判断
                    node.isFullyExpanded = True
                return newNode    #expand一个新节点后，就直接退出循环了。返回这个新棋面（节点）

    def backpropogate(self, node, reward):   #基于父节点关系，反向更新路径上的信息。输入当前节点和基于它的rollout结果。
        while node is not None:
            node.numVisits += 1              #更新当前节点的总访问次数
            node.totalReward = 0.5 * (reward + node.totalReward)     #更细当前节点的reward
            node = node.parent               #指向当前节点父节点（root是初始棋面）

    def getBestChild(self, node, explorationValue):  #拓展过程，寻找最佳节点（当eV为0，则是纯贪心）
        bestValue = float("-inf") #初始化评价值和最佳节点
        bestNodes = []            #最佳节点可能有多个，都放进去最后随机选择
        for child in node.children.values():   #每一个child都对应一个棋面
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)     #这个就是UCT算法了，不过这里还有棋手的问题
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)   #如果有多个相同的，则随机选择一个子节点（棋面）

class NaughtsAndCrossesState():                            # 连接到treeNode的state中
    def __init__(self,b):
        self.board = [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]]     # 表示一个具体的棋面
        self.currentPlayer = 1
        self.laststep = [2, 0]
        self.target = [0,3]

        self.reward = 0
        self.edge = [75, 120, 200]
        self.lastlane = [b[0], 0]
        self.laststate = [b[1], b[2]]
        self.T = 0
        self.Tstep = 3

    def getCurrentPlayer(self):                            # 输入：node.state
        return self.currentPlayer

    def getPossibleActions(self):                          # 输入：node.state
        possibleActions = []
        flag1 = 0
        if self.lastlane[0] <= 0:  # 不能在最上
            flag1 = 1
        else:
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] - 1)[0]:  # 并道后在车前
                if self.laststate[0] <= vehicles.positions(ini, self.T, self.lastlane[0] - 1)[0]:  # 不能并道之前在车后
                    flag1 = 1
            else:  # 并道后在车后
                aa = vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] - 1)[1]
                if (vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] - 1)[0] - (self.laststate[0] + self.Tstep * self.laststate[1])) <= 0.1 * (
                        self.laststate[1] * self.laststate[1] - aa * aa):
                    flag1 = 1  # 不能小于安全车距
        if self.lastlane[0] >= 2:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.edge[2]:
                flag1 = 1
        if flag1 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=1))

        # 在考虑向下并线
        flag2 = 0
        if self.lastlane[0] >= 2:  # 不能在最下
            flag2 = 1
        else:
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] + 1)[0]:  # 并道后在车前
                if self.laststate[0] <= vehicles.positions(ini, self.T, self.lastlane[0] + 1)[0]:  # 不能并道前在车后
                    flag2 = 1
            else:  # 并道后在车后
                aa = vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] + 1)[1]
                if (vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0] + 1)[0] - (self.laststate[0] + self.Tstep * self.laststate[1])) <= 0.1 * (
                        self.laststate[1] * self.laststate[1] - aa * aa):
                    flag2 = 1  # 不能小于安全车距
        if self.lastlane[0] >= 0:  # 不能超过纵向距离约束（非目标lanelet）
            if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.edge[2]:
                flag2 = 1
        if flag2 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=2))

        # 下面考虑直行--加速（+1 m/s2）
        flag3 = 0
        if self.laststate[1] > 17:
            flag3 = 1
        else:
            if self.lastlane[0] >= 1:  # 不能超过纵向距离约束（非目标lanelet）
                if (self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep) >= self.edge[2]:
                    flag3 = 1
            aa = vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0])[1]
            if (self.laststate[0] + 3) >= aa:
                if (vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0])[0] - (
                        self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep)) <= 0.1 * (
                        self.laststate[1] * self.laststate[1] - aa * aa):
                    flag3 = 1  # 不能小于安全车距
        if flag3 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=3))

        # 下面考虑直行--匀速（+0 m/s2）
        flag4 = 0
        if self.laststate[1] > 100000:
            flag4 = 1
        else:
            if self.lastlane[0] >= 1:  # 不能超过纵向距离约束（非目标lanelet）
                if (self.laststate[0] + self.Tstep * self.laststate[1]) >= self.edge[2]:
                    flag4 = 1
            aa = vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0])[1]
            if self.laststate[0] >= aa:
                if (vehicles.positions(ini, self.T + self.Tstep, self.lastlane[0])[0] - (
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
            if self.lastlane[0] >= 1:  # 不能超过纵向距离约束（非目标lanelet）
                if (self.laststate[0] + self.Tstep * self.laststate[1] - 0.5 * self.Tstep * self.Tstep) >= self.edge[2]:
                    flag5 = 1
        if flag5 == 0:  # 满足所有条件时，可以向上换道
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=5))

        if possibleActions == []:
            possibleActions.append(Action(player=self.currentPlayer, x=1, y=1, act=6))
        #print(possibleActions)
        return possibleActions                           # 输出为所有可能落子位置的集合

    def takeAction(self, action):                        # node.state.takeAction(action)，这一步只更新棋面和棋手
        newState = deepcopy(self)

        newState.board[action.x][action.y] = 1
        newState.laststep=[action.x,action.y]
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
            newState.laststate[0] = self.laststate[0] + self.Tstep * self.laststate[1] + 0.5 * self.Tstep * self.Tstep
            newState.laststate[1] = self.laststate[1] + 1 * self.Tstep
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
            newState.laststate[1] = 10
            newState.lastlane[0] = self.lastlane[0]

        newState.T = self.T + self.Tstep
        newState.reward = self.reward - 1 - 0.05 * newState.lastlane[0] -0.05 * (300-newState.laststate[0])
        if action.act == 6:
            newState.reward = newState.reward - 100
        return newState

    def isTerminal(self):                                # 因为treeNode中定义了连接，输入node.isTerminal
        tag = False
        if self.lastlane[0] == 0:
            if self.laststate[0] > self.edge[2]:
                tag = True
        return tag                                       # 最终输出0或1.要根据自己的要求修改逻辑


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

if __name__=="__main__":
    a = [[100,5],[100,5],[50,10]]
    ini = vehicles(a)
    b = [2,20,15]
    initialState = NaughtsAndCrossesState(b)
    searcher = mcts(iterationLimit=5000) #改变循环次数或者时间
    action = searcher.search(initialState=initialState) #一整个类都是其状态
    print(action.act)
