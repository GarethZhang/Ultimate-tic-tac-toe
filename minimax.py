import numpy as np
import random
import time
from uttt_api import *

# # example code
# # initialize an empty uttt board with no states
# init_state = UtttState()
#
# # print out board state on a 9x9 board
# init_state.print_state()
#
# # print out one of successors of current state
# init_state.successors()[80].print_state()
#
# # check if a ttt board is won by a player
# print(init_state.goal_state())
#
# # print out state_string
# init_state.state_string()
#
# # Access small UTTT board indexed from 1 to 9
# print(init_state.boards[1].goal_state())

# define global variables
timeout = 10
num_states = 0
states = []

# compute score of current node using minimax
def miniMax(utttState, searchDepth, alpha, beta, endTime):
    global num_states
    num_states += 1
    if (endTime - time.time()) < 0.5:
        return (alpha, beta)
    # print("State: " + str(num_states))
    # if UTTT has been solved
    if utttState.goal_state() == 0:
        print("Goal state: MAX")
        return (np.inf, beta)
    elif utttState.goal_state() == X:
        print("Goal state: MIN")
        return (alpha, -np.inf)

    # apply minimax
    ChildList = utttState.successors()
    currPlayer = utttState.action[2]
    if currPlayer == 0:
        # if minimax has expanded beyond searchDepth
        if searchDepth == 0:
            # print("Player0: " + str(utttState.heuristic))
            return (utttState.heuristic, beta)
        for x in ChildList:
            alpha = max(alpha, miniMax(x, searchDepth - 1, alpha, beta, endTime)[1])
            # beta = min(beta, miniMax(x, searchDepth - 1, alpha, beta)[1])
            if beta <= alpha:
                break
        return (alpha, beta)
    else:
        # if minimax has expanded beyond searchDepth
        if searchDepth == 0:
            return (alpha, utttState.heuristic)
        for x in ChildList:
            # alpha = max(alpha, miniMax(x, searchDepth - 1, alpha, beta)[0])
            beta = min(beta, miniMax(x, searchDepth - 1, alpha, beta, endTime)[0])
            if beta <= alpha:
                break
        return (alpha, beta)

def monteCarlo(utttState, searchWidth, searchDepth, endTime):
    score = []
    storeState = copy.deepcopy(utttState)
    for i in range(searchWidth):
        utttState = copy.deepcopy(storeState)
        for j in range(searchDepth):
            utttState = random.choice(utttState.successors())
        score.append(utttState.heuristic)
    return np.average(score)

def getMove(utttState, searchDepth=10, timeout=10):
    nextState = None
    cost = -np.inf
    startTime = time.time()
    endTime = startTime + timeout
    for x in utttState.successors():
        print(x.action)
        alpha = miniMax(x, searchDepth, -np.inf, np.inf, endTime)[0]
        print("Possible move cost: " + str(alpha))
        if cost < alpha:
            nextState = x
            cost = alpha
    return nextState.action

def getMCMove(utttState, searchWidth=10, searchDepth=10, timeout=10):
    # set timeout bound
    startTime = time.time()
    endTime = startTime + timeout

    # initialize variables to store statistics
    plays = {}
    wins = {}

    # Childnode of current state
    childSize = len(utttState.successors())
    meanPayout = [1] * childSize
    plays = [1] * childSize
    storeState = copy.deepcopy(utttState)
    for i in range(searchWidth):
        utttState = copy.deepcopy(storeState)
        index = int(np.random.randint(0, childSize-1, 1))
        utttState = utttState.successors()[index]
        for j in range(searchDepth):
            utttState = random.choice(utttState.successors())
            if (endTime - time.time()) < 0.5:
                break
        plays[index] += 1
        meanPayout[index] += utttState.heuristic
    meanPayout = np.int64(np.array(meanPayout) / np.array(plays))
    UCB = meanPayout + np.sqrt(2 * np.log(plays) / np.sum(plays))
    nextState = copy.deepcopy(storeState.successors()[np.argmax(UCB)])
    return nextState.action

def initRandomBoard(randomDepth):
    randomState = UtttState()
    initBoard = int(np.random.randint(1, 9, 1))
    initMove = int(np.random.randint(1, 9, 1))
    randomState = UtttState(parent=randomState, action=(initBoard, initMove, 0))
    print("step " + str(0) + " with Player1's move " + str(initMove) + " at board " + str(initBoard))
    step = 1
    for i in range(randomDepth-1):
        if randomState.goal_state() != -1:
            print("Goal state determined->Winner: " + str(randomState.goal_state()))
            return -1
        if step % 2 == 0:
            randomState = copy.deepcopy(random.choice(randomState.successors()))
            print("step " + str(step) + " with Player1's move " + str(randomState.action[1]) + " at board " + str(randomState.action[0]))
        else:
            randomState = copy.deepcopy(random.choice(randomState.successors()))
            # randomState = UtttState(parent=randomState, action=(prevMove, num, 1))
            print("step " + str(step) + " with Player1's move " + str(randomState.action[1]) + " at board " + str(randomState.action[0]))
        step += 1
    return randomState

if __name__ == "__main__":
    randomS = initRandomBoard(40)
    randomS.print_state()
    print("Possible number of moves: " + str(len(randomS.successors())))
    action = getMove(randomS, searchDepth=10, timeout=timeout)
    # action = getMCMove(randomS, searchWidth=100, searchDepth=10, timeout=timeout)
    print("Selected move: " + str(action))
    # MCMethod = MonteCarlo(randomS, time=10)
    # print("Selected Move: " + str(MCMethod.get_play()))

    # for i in randomS.successors():
    #     i.print_state()
    # randomS2 = initRandomBoard(10)
    # randomS3 = copy.deepcopy(randomS)
    # plays = {}
    # state = []
    # state.append(randomS)
    # plays[(0, randomS)] = 1
    # # print(randomS.parent.currentPlayer)
    # # print(all(plays.get((0, S)) for S in state))
    # print(randomS.uniqueID)
    # print(randomS3.uniqueID)
    # print((0, randomS) in plays)