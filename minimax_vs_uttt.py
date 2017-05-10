import numpy as np
import random
import datetime
import time
from uttt_api import *

# define global variables
timeout = 10
num_states = 0
states = []

# compute score of current node using minimax
def miniMax(utttState, searchDepth, alpha, beta):
    global num_states
    num_states += 1
    # if UTTT has been solved
    if utttState.goal_state() == 0:
        print("Goal state: MAX")
        return (np.inf, beta)
    elif utttState.goal_state() == X:
        print("Goal state: MIN")
        return (alpha, -np.inf)

    # apply minimax
    ChildList = utttState.successors()
    lastPlayer = utttState.action[2]
    if lastPlayer == 0:
        # if minimax has expanded beyond searchDepth
        if searchDepth == 0:
            return (utttState.heuristic, beta)
        for x in ChildList:
            alpha = max(alpha, miniMax(x, searchDepth - 1, alpha, beta)[1])
            if beta <= alpha:
                break
        return (alpha, beta)
    else:
        # if minimax has expanded beyond searchDepth
        if searchDepth == 0:
            return (alpha, utttState.heuristic)
        for x in ChildList:
            beta = min(beta, miniMax(x, searchDepth - 1, alpha, beta)[0])
            if beta <= alpha:
                break
        return (alpha, beta)

def getMove(utttState, searchDepth=10, timeout=10):
    nextState = None
    alphaScore = -np.inf
    betaScore = np.inf
    startTime = datetime.datetime.utcnow()
    for x in utttState.successors():
        if datetime.datetime.utcnow() - startTime > datetime.timedelta(seconds=timeout):
            break
        # print(x.action)
        if x.currentPlayer==0:
            alpha = miniMax(x, searchDepth, -np.inf, np.inf)[0]
            # print("Possible move cost: " + str(alpha))
            if alphaScore < alpha:
                nextState = x
                alphaScore = alpha
        else:
            beta = miniMax(x, searchDepth, -np.inf, np.inf)[1]
            # print("Possible move cost: " + str(beta))
            if betaScore > beta:
                nextState = x
                betaScore = beta
    return nextState.action

def initRandomBoard(randomDepth):
    randomState = UtttState()
    for i in range(randomDepth):
        randomState = copy.deepcopy(random.choice(randomState.successors()))
    return randomState

if __name__ == "__main__":
    # initialize a randomBoard given a randomDepth
    randomS = initRandomBoard(21)

    # print out the board state
    randomS.print_state()
    print("Possible number of moves: " + str(len(randomS.successors())))

    # init game board for two algorithms to compete
    initState = UtttState()
    state = copy.deepcopy(initState)
    step = 0
    while state.goal_state() == -1 and state.successors() != []:
        if step % 2 == 0:
            MCMethod = MonteCarlo(state, time=10)
            state = UtttState(parent=state, action=MCMethod.get_play())
            state.print_state()
            step += 1
        else:
            state = UtttState(parent=state, action=getMove(state, searchDepth=2, timeout=10))
            state.print_state()
            step += 1
        print("Step: " + str(step))
    state.print_state()
    print("Winner: " + str(state.goal_state()))