import numpy as np


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
        self.prevNode_ = None # previous node variable for backtracking
        self.children_ = [] # list of children nodes

        self.P_ , self.V_ = NN.run(state)

    def getChildren(self, env):
        '''
        obtains all the children nodes from interacting with the chess env
        '''
        possible_child_nodes =[]
        self.children_ = possible_child_nodes

    def backTrack(self, value):
        '''
        recursive backtracking method for MCTS
        '''
        self.N_ += 1
        self.W_ += value
        self.Q_ = self.W_ /self.N_
        if self.prevNode_ != None:
            self.prevNode_.backTrack(value)

    def select(self):
        '''
        from the given parameters pick a child node to traverse to
        the P_ (prior probability) is obtained from running the Neural Network
        when the node was initialized. We assume P is a np array
        '''
        Q = []# initialize to obtain Q values
        N_child = [] #initialize to obtain N of each child
        if self.children_ == None:
            print("children nodes not initialized")
            return None
        for child in self.children_ :
            Q.append(child.Q_)
            N_child.append(child.N_)
        Q = np.array(Q)
        N_child = np.array(N_child)
        N_all = self.N_ # we assume N of all children == N of parent
        U = self.P_* np.sqrt(N_all)/ (1 + N_child) #obtain the U values
        selected_child = np.argmax(Q+U)
        return selected_child

    def runSimulation(self):
        '''
        runs MCTS simulation once
        '''


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
