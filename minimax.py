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
    randomS = initRandomBoard(20)
    randomS.print_state()
    print("Possible number of moves: " + str(len(randomS.successors())))
    action = getMove(randomS, searchDepth=10, timeout=timeout)
    print("Selected move: " + str(action))