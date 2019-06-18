import numpy as np

#TO DO: add virtual loss variable in backup see pg.22
class State:
    """
    A state class that represents the board state of chess
    """
    def __init__(self):




class Node:
    """
    A Node class that represents a node in a MCTS as stated in the paper
    "Mastering the game of Go without human knowledge"
    """
    def __init__(self, state, NN):
        '''
        We use the state and Neural Network to obtain prior probability (P_)
        and evaluation of value of the node (V_)
        '''
        self.state_ = state # state of the chess env
        self.N_ = 0.0 # count
        self.W_ = 0.0 # total action value
        self.Q_ = 0.0 # action value
        self.prevNode_ = None # None value idicates that it is root node
        self.children_ = [] # list of children nodes

        self.P_ = NN.getP(state)

    def getChildren(self, env):
        '''
        obtains all the children nodes from interacting with the chess env
        '''
        possible_child_nodes =[]

        self.children_ = possible_child_nodes

    def select(self, c_puct):
        '''
        Clarification: this function does not pick a move to play
        from the given parameters pick a child node to traverse to
        the P_ (prior probability) is obtained from running the Neural Network
        when the node was initialized. We assume P is a np array
        '''
        Q = []# initialize to obtain Q values
        N_child = [] # initialize to obtain N of each child
        if self.children_ == None:
            print("children nodes not initialized")
            return None
        for child in self.children_ :
            Q.append(child.Q_)
            N_child.append(child.N_)
        Q = np.array(Q)
        N_child = np.array(N_child)
        N_all = self.N_ # we assume N of all children == N of parent
        U = c_puct* self.P_* np.sqrt(N_all)/ (1 + N_child) #obtain the U values
        selected_child = np.argmax(Q+U)
        return selected_child

    def backTrack(self, value):
        '''
        recursive backtracking method for MCTS
        '''
        self.N_ += 1
        self.W_ += value
        self.Q_ = self.W_ /self.N_
        if self.prevNode_ != None:
            self.prevNode_.backTrack(value)

    def runSimulation(self, state, NN, remain_iter):
        '''
        runs MCTS simulation once
        remain_iter argument gives remaining number of simulations iterations
        that we still need to do. The number is decremented by one everytime
        a simulation goes to the next child node
        '''
        if remainIter == 0: #last node. This means we have to decide the winner
            v = NN.getV(state)
            self.backTrack(v) #start back tracking
        else:
            next_node = self.select()
            next_node.prevNode_ = self # link selected child to parent
            next_remain_iter = remain_iter - 1
            next_node.runSimulation(next_remain_iter)


    def pickMove(self):
        '''
        Makes a move and picks the next node (state) to go to
        '''
        # next_move_child =
        self.children_ = None # frees not picked children to be deleted
        # still not sure if this is a valid way to free up memory
        next_move_child.prevNode_ = None #the child is now root node
        return next_move_child

class NN:
    """
    A Neural Network Class that plays the role of the neural network function
    in the paper "Mastering the game of Go without human knowledge"
    """
    def __init__(self):


    def train(self, train_data):
        '''
        trains the network with given training data
        '''
    def run(self. data):
        '''
        runs the network with data (analogous to "testing") to obtain
        policy (P), a vector quantity, and value (V), a scalar quantity
        '''
        P = None
        V = None
        return P, V

    def getP(self, state):
        '''
        uses the Neural Network to obtain the prior probability (P) for MCTS
        '''
        P, V = self.run(state)
        return P

    def getV(self, state):
        '''
        uses the Neural Network to obtain the value (V) of the state for MCTS
        '''
        P, V = self.run(state)
        return V
