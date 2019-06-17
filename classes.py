
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
        self.state_ = state # state of the chess env
        self.N_ = 0.0 # count
        self.W_ = 0.0 # total action value
        self.Q_ = 0.0 # action value
        self.P_ = 0.0 # prior probability
        self.prevNode_ = None # previous node variable for backtracking
        self.children_ = [] # list of children nodes


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
        self.N_ +=1
        self.W_ += value
        self.Q_ = self.W_ /self.N_
        if self.prevNode_ != None:
            self.prevNode_.backTrack(value)

    def takeAction(self):
        '''
        from the given parameters pick a child node to traverse to
        '''

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
        runs the network with data (analogous to "testing")
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
