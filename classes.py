import numpy as np
import torch
import torch.nn as nn

#TO DO: add virtual loss variable in backup see pg.22

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


    def pickMove(self, temp):
        '''
        Makes a move and picks the next node (state) to go to
        '''
        N_vector = [] # vector of N^(1/temp) values of all children
        for child in self.children_:
            N_vector.append(child.N_**(1/temp))
        N_vector = np.array([N_vector]) # make it np array
        N_vector = N_vector/np.sum(N_vector) # normalize to make it a probability distribution
        next_move_idx = np.random.choice([i in range(N_vector.shape[0])], p = N_vector)
        # next_move_arg picks the index of the N_vector according to its probability distribution
        next_move_child = self.children_[next_move_idx] # picks the next child
        self.children_ = None # frees not picked children to be deleted \
        # still not sure if this is a valid way to free up memory however
        next_move_child.prevNode_ = None #the child is now root node
        return next_move_child

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(num_features, num_features),
            nn.Conv2d(num_features, num_features, 3, stride=1, padding=1),# filters = output_size
            nn.BatchNorm2d(num_features),# batch normalization
        )
        self.nonlinear_out = nn.ReLU()

    def forward(self, x):
        route = self.block(x)
        skip = x + route
        return self.nonlinear_out(skip)

class ConvBlock(nn.Module):
    def __init__(self, input_size, num_features):
        super(ConvBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_size, num_features, 3, stride=1, padding=1),# filters = output_size
            nn.BatchNorm2d(num_features),# batch normalization
            nn.ReLU(),
        )

    def forward(self, x):
        return self.module(x)

class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(start_dim=1)

class NN(nn.Module):
    """
    A Neural Network Class that plays the role of the neural network function
    in the paper "Mastering the game of Go without human knowledge"
    """
    def __init__(self, input_size, num_features, num_residual_layers, action_depth=4672, board_width=8):

        super(NN, self).__init__()
        self.tower = nn.Sequential(
            ConvBlock(input_size, num_features)
        )

        for i in range(num_residual_layers):
            self.tower.add_module("resid"+str(i+1),
                                    ResidualBlock(num_features))

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_features, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2 * (board_width ** 2), action_depth)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1 * (board_width ** 2), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

        self.prob_mapper = nn.Softmax()

    def train(self, train_data):
        '''
        trains the network with given training data
        '''

    def run(self, state, avail_actions):
        '''
        runs the network with data (analogous to "forward") to obtain
        policy (P), a vector quantity, and value (V), a scalar quantity
        '''
        tower = self.tower(state)
        policy_logits = self.policy_head(tower)
        policy_logits = (avail_actions * (policy_logits- torch.min(policy_logits)))
        print("sum: ", torch.sum(policy_logits))
        policy_logits = policy_logits/torch.sum(policy_logits)
        policy_logits = torch.autograd.Variable(policy_logits, requires_grad=False)
        value = self.value_head(tower)
        return policy_logits, value

    def getP(self, state, avail_actions):
        '''
        runs the network with data (analogous to "forward") to obtain prior probability (P) for MCTS
        '''
        P, V = self.run(state)
        return P

    def getV(self, state, avail_actions):
        '''
        runs the network with data (analogous to "forward") to obtain the value (V) of the state for MCTS
        '''
        P, V = self.run(state)
        return V
