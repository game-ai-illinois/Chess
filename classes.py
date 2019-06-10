 
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
    def __init__(self):
        self.N_ = 0.0
        self.W_ = 0.0
        self.Q_ = 0.0
        self.P_ = 0.0
        self.prevNode = None # previous node variable for backtracking
   
    def getChildren(self, env):
        '''
        obtains all the children nodes from interacting with the chess env
        '''

class NN:
    """
    A Neural Network Class that plays the role of the neural network function 
    in the paper "Mastering the game of Go without human knowledge"
    """
    