import copy
import numpy as np

class StateSpace:
    '''Abstract class for defining State spaces for search routines'''
    
    def __init__(self, parent, action):
        '''Problem specific state space objects must always include the data items
           a) self.parent === the state from which this state was generated (by
              applying "action". If it is the initial state a good
              convention is to supply the state None
           b) self.action === the name of the action used to generate
              this state from parent. If it is the initial state a good
              convention is to supply the action name "START"
        '''
        self.parent = parent
        self.action = action

    def successors(self):
        '''This method when invoked on a state space object must return a
           list of successor states, each with the data items "action"
           the action used to generate this successor state, and parent set to self.
           Also any problem specific data must be specified property.'''
        raise Exception("Must be overridden in subclass.")

    def print_state(self):
        '''Print a representation of the state'''
        raise Exception("Must be overridden in subclass.")

    def print_path(self):
        '''print the sequence of actions used to reach self'''
        #can be over ridden to print problem specific information
        s = self
        states = []
        while s:
            states.append(s)
            s = s.parent
        states.pop().print_state()
        while states:
            print(" ==> ", end="")
            states.pop().print_state()
        print("")

# Constants
O = 0
X = 1
EMPTY = -1
EMPTY_BOARD = {i:EMPTY for i in range(1,10)}

class TictactoeState(StateSpace):
    '''Create a 3x3 Tic-Tac-Toe Board State.'''
    
    def __init__(self, parent = None, action = (0,-1), marks = copy.deepcopy(EMPTY_BOARD)):
        """
        Create a new Tic-Tac-Toe state.

        @param action: A tuple of (position, mark) used to generate this state from parent.
                       If it is the initial state, a tuple (0, -1) is supplied.
                       The position is encoded by the numbers 1 to 9 as follows:
                       1|2|3
                       -----
                       4|5|6
                       -----
                       7|8|9
        @param marks: A dictionary where the keys are the coordinates of each position, and the value is the type of marks (EMPTY:-1, X:1, or O:0) at the position.
        """
        StateSpace.__init__(self, parent, action)
        self.action = action
        self.marks = marks
        if parent != None and action != (0,-1):
            self.marks = copy.deepcopy(parent.marks)
            self.marks[action[0]] = action[1]
        self.heuristic = self.calcHeuristic()

    def calcHeuristic(self):
        # add heuristic score if forms a stright line
        tripleList = np.array([[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7]])
        doubleList = np.array([0,1])
        marksToList = np.array([self.marks[i] for i in range(1,10)])
        heuristic = 0
        for indexList in tripleList:
            if marksToList[indexList-1].all() == X:
                heuristic -= 1
                break
            elif marksToList[indexList-1].all() == 0:
                heuristic += 1
                break

        # add heuristic score if forms a stright line of two
        for indexList in tripleList:
            if marksToList[indexList-1][0] == X and marksToList[indexList-1][1] == -1 and marksToList[indexList-1][2] == X:
                heuristic -= 1
            elif marksToList[indexList-1][0] == -1 and marksToList[indexList-1][1] == X and marksToList[indexList-1][2] == X:
                heuristic -= 1
            elif marksToList[indexList-1][0] == X and marksToList[indexList-1][1] == X and marksToList[indexList-1][2] == -1:
                heuristic -= 1
            elif marksToList[indexList-1][0] == 0 and marksToList[indexList-1][1] == -1 and marksToList[indexList-1][2] == 0:
                heuristic += 1
            elif marksToList[indexList-1][0] == -1 and marksToList[indexList-1][1] == 0 and marksToList[indexList-1][2] == 0:
                heuristic += 1
            elif marksToList[indexList-1][0] == 0 and marksToList[indexList-1][1] == 0 and marksToList[indexList-1][2] == -1:
                heuristic += 1
        return heuristic
    
    def successors(self, player):
        """
        Generate all the actions that can be performed from this state, and the states those actions will create. If it is the initial state, assume 'O' plays first.
        
        @param player: Specifies the current player (O or X).
        """
        successors = []
        for pos, mark in self.marks.items():
            new_marks = self.marks.copy()
            if mark == EMPTY:
                new_marks[pos] = player
            else:
                continue

            new_state = TictactoeState(parent=self, action=(pos,player),
                                       marks=new_marks)
            successors.append(new_state)

        return successors
    
    def goal_state(self):
        """
        Return the winner (O:0 or X:1) if a player has won this tic-tac-toe board;
        else, return -1.
        """
        # 8 Winning positions
        winning = []
        for i in range(3):
            winning.append({self.marks[3*i+1], self.marks[3*i+2], self.marks[3*i+3]})
            winning.append({self.marks[1+i], self.marks[4+i], self.marks[7+i]})
        winning.append({self.marks[1], self.marks[5], self.marks[9]})
        winning.append({self.marks[3], self.marks[5], self.marks[7]})
        
        if {O} in winning:
            return O
        elif {X} in winning:
            return X
        else:
            return -1

    def avail_marks(self):
        return [i for i in range(1, 10) if self.marks[i] == -1]

    def state_string(self):
        """
        Return a string representation of a state that can be printed to stdout.
        """
        s = []
        for pos in range(1,10):
            if self.marks[pos] == EMPTY:
                s.append(" ")
            elif self.marks[pos] == O:
                s.append("O")
            else:
                s.append("X")
            s.append("|")
        s.pop() # remove the last "|"
        s[5] = '\n-----\n'
        s[11] = '\n-----\n'

        return "".join(s)

    def print_state(self):
        """
        Print the string representation of the state.
        """
        if self.action[1] == -1:
            act = "None"
            pos = "None"
        elif self.action[1] == O:
            act = "'O'"
            pos = self.action[0]
        else:
            act = "'X'"
            pos = self.action[0]
        print("ACTION was " + str(act) + " at position " + str(pos))
        print(self.state_string())

# Constants
EMPTY_UTTT = {i:TictactoeState() for i in range(1,10)}

class UtttState(StateSpace):
    '''Create a 3x3 ULTIMATE Tic-Tac-Toe Board State.'''
    
    def __init__(self, parent = None,
                 action = (0, 0, -1), boards = copy.deepcopy(EMPTY_UTTT)):
        """
        Create a new ULTIMATE Tic-Tac-Toe state.

        @param action: A tuple of (board position, mark position within the board, mark)
                       used to generate this state from parent.
                       If it is the initial state, a tuple (0, 0, -1) is supplied.
                       The position is encoded by the numbers 1 to 9 as follows:
                       1|2|3
                       -----
                       4|5|6
                       -----
                       7|8|9
        @param boards: A dictionary where the keys are the coordinates of each position,
                       and the value is a TictactoeState at the position.
        """
        StateSpace.__init__(self, parent, action)
        self.action = action
        self.parent = parent
        self.boards = boards
        self.heuristic = 0
        if parent != None and action != (0, 0, -1):
            self.boards = copy.deepcopy(parent.boards)
            stateMark = copy.deepcopy(self.boards.get(action[0]).marks)
            stateMark[action[1]] = action[2]
            self.boards[action[0]] = TictactoeState(marks = stateMark)
            self.heuristic = parent.heuristic + self.calcHeuristic()

    def calcHeuristic(self):
        minus_ = self.parent.boards[self.action[0]].calcHeuristic()
        plus_ = self.boards[self.action[0]].calcHeuristic()
        return plus_ - minus_
    
    def successors(self):
        """
        Generate all the actions that can be performed from this state, and the states those actions will create. If it is the initial state, assume 'O' plays first.
        """
        successors = []
        
        new_pos = self.action[1]
        new_mark = max(0,self.action[2] ^ 1) # initial state or opposite player
        
        # Case 1: the designated board by previous action is still available to play
        if new_pos != 0 and self.boards[new_pos].goal_state() == -1:
            new_succ = self.boards[new_pos].successors(new_mark)
            for succ in new_succ:
                new_boards = copy.deepcopy(self.boards)
                new_boards[new_pos] = succ
                new_state = UtttState(parent=self,
                                      action=(new_pos,succ.action[0],new_mark),
                                      boards=new_boards)
                successors.append(new_state)
        
        # Case 2: initial state, or
        #         the designated board by previous action has already been won,
        #         then the player can play in any other board
        else:
            for i in range(1,10):
                if self.boards[i].goal_state() == -1:
                    new_succ = self.boards[i].successors(new_mark)
                    for succ in new_succ:
                        new_boards = copy.deepcopy(self.boards)
                        new_boards[i] = succ
                        new_state = UtttState(parent=self,
                                              action=(i,succ.action[0],new_mark),
                                              boards=new_boards)
                        successors.append(new_state)

        return successors
    
    def goal_state(self):
        """
        Return the winner (O:0 or X:1) if a player has won this tic-tac-toe board;
        else, return -1.
        """
        # find status of all tic-tac-toe boards
        status = {}
        for i in range(1,10):
            status[i] = self.boards[i].goal_state()
            
        # 8 Winning positions
        winning = []
        for i in range(3):
            winning.append({status[3*i+1], status[3*i+2], status[3*i+3]})
            winning.append({status[1+i], status[4+i], status[7+i]})
        winning.append({status[1], status[5], status[9]})
        winning.append({status[3], status[5], status[7]})
        
        if {O} in winning:
            return O
        elif {X} in winning:
            return X
        else:
            return -1

    def avail_marks(self):
        return [(i, j) for i in range(1, 10) for j in self.boards[i].avail_marks()]

    def state_string(self):
        """
        Return a string representation of a state that can be printed to stdout.
        """
        s = []
        tmp = []
        for pos in range(1,10):
            state_str = self.boards[pos].state_string().split("\n")
            if pos % 3 != 0:
                state_str = [string+" | " for string in state_str]
                tmp.extend(state_str)
            else:
                state_str = [string+"\n" for string in state_str]
                tmp.extend(state_str)
                for i in range(5):
                    s.extend([tmp[i], tmp[i+5], tmp[i+10]])
                if pos != 9:
                    s.append('---------------------\n')
                tmp = []

        return "".join(s)

    def print_state(self):
        """
        Print the string representation of the state.
        """
        if self.action[2] == -1:
            act = "None"
            board = "None"
            pos = "None"
        elif self.action[2] == O:
            act = "'O'"
            board = self.action[0]
            pos = self.action[1]
        else:
            act = "'X'"
            board = self.action[0]
            pos = self.action[1]
        
        print("ACTION was " + str(act) + " in board " + str(board)
              + " at position " + str(pos))
        print(self.state_string())

if __name__ == "__main__":
    # default empty board
    init_state = UtttState()
    second_state = UtttState(parent=init_state, action=(1,1,1))

    second_state.print_state()
    print(len(second_state.successors()))