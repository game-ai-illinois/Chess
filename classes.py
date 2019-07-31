import numpy as np
import torch
import torch.nn as nn
from chess_env import input_state, is_white, return_legal_moves

#TO DO: add virtual loss variable in backup see pg.22

class Edge:
    """
    an edge class
    """
    def __init__(self, parent_node, probability, move_idx, move_text):
        self.parent_node_ = parent_node #node that the edge belongs to
        self.child_node_ = None #node that the edge connects to, first initialized to None
        self.N_ = 0.0 # count
        self.W_ = 0.0 # total action value
        self.Q_ = 0.0 # action value
        self.P_ = probability
        self.move_idx_ = move_idx
        self.move_text_ = move_text

    def __del__(self): 
        del self.child_node_ 
        del self.N_
        del self.W_
        del self.Q_
        del self.P_
        del self.move_text_ 
        del self.parent_node_
        print("destruct edge")
        # I wonder if I have to desctuct "self" as well

    def rollback(self, value):
        '''
        recursive backtracking method for MCTS
        '''
        self.N_ += 1
        self.W_ += value
        self.Q_ = self.W_ / self.N_
        if self.parent_node_.parent_edge_ != None :
            self.prevNode_.parent_edge_.rollback(value)


class Node:
    """
    A Node class that represents a node in a MCTS as stated in the paper
    "Mastering the game of Go without human knowledge"
    """
    def __init__(self, board, parent_edge = None):
        '''
        We use the state and Neural Network to obtain prior probability (P_)
        and evaluation of value of the node (V_)
        '''
        self.board_ = board # state of the chess env
        self.parent_edge_ = parent_edge # None value idicates that the node is starting game node
        #root nodes should otherwise still have parent edges to access its value for stop condition
        self.children_edges_ = [] # list of edges nodes
        board_state_string = self.board_.fen()
        self.state_ = input_state(board_state_string)
        self.is_black_ = not is_white(board_state_string)

    def __del__(self, keep_child_idx = None):
         
        """
        Delete function for Node class
        """
        del self.board_
        if keep_child_idx != None : # delete all edges except the one specified by keep_child_idx
            for edge_idx in range(len(self.children_edges_)):
                if edge_idx != keep_child_idx:
                    del self.children_edges_[0]
                else:
                    l.pop(0) #pop the desired edge out
        else:
            for edge in self.children_edges_:
                del edge
        del self.children_edges_ 
        self.board_ = None
        del self.parent_edge_
        print("destruct node")

    def getChildrenEdges(self, NN): 
        '''
        obtains all the children nodes from interacting with the chess env
        '''
        board_state_string = self.board_.fen()
        state_array = input_state(board_state_string) #turn into arrays for NN
        is_black = not is_white(board_state_string)
        legal_moves_array, move_dict = return_legal_moves(self.board_, is_black)
        print("state array shape: ", state_array.shape)
        print("legal array shape: ", legal_moves_array.shape)
        prin(legal_moves_array)
        P, _ = (NN.run(state_array, legal_moves_array))
        for idx in range(len(legal_moves_array)):
            if legal_moves_array[idx] != 0:
                edge = Edge(node, p[idx], idx, move_dict[idx])
                self.children_edges_.append(edge)

    def select(self, c_puct):
        '''
        returns indx of a child edge to select to play
        Clarification: this function does not pick a move to play
        from the given parameters pick a child node to traverse to
        the P_ (prior probability) is obtained from running the Neural Network
        when the node was initialized. We assume P is a np array
        '''
        if node.P_:
            print("node's probability not defined and is None")
            return None #end function
        if self.children_edges_ == []:
            print("children nodes not initialized")
            return None
        Q = []# initialize to obtain Q values
        N_child = [] # initialize to obtain N of each child
        P = []
        for edge in self.children_edges_ :
            Q.append(edge.Q_)
            N_child.append(edge.N_)
            P.append(edge.P_)
        Q = np.array(Q)
        N_child = np.array(N_child)
        p = np.array(P)
        N_all = np.sum(N_child)
        U = c_puct* P* np.sqrt(N_all)/ (1 + N_child) #obtain the U values
        selected_child_idx = np.argmax(Q+U)
        return selected_child_idx

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

    # def train(self, train_data):
        

    def run(self, state, avail_actions):
        '''
        runs the network with data (analogous to "forward") to obtain
        policy (P), a vector quantity, and value (V), a scalar quantity
        both the state and avail_actions are assumed to be np arrays
        '''
        state = torch.from_numpy(state).float() #turn state and avail_actions into torch tensors
        avail_actions = torch.from_numpy(avail_actions).float()
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


    def optimizer(self, learning_rate, c):
        return torch.optim.Adam({self.tower.parameters(), self.policy_head.parameters(), self.value_head.parameters()}, lr=learning_rate, weight_decay= c)

    def optimization(self, data, learning_rate, c):
        '''
        trains the network with given training data
        '''
        state = data[0]
        search_policy = data[1]
        z = data[-1]
        avail_actions = search_policy != 0
        P, V = self.run(state, avail_actions)
        loss = torch.mm((z-V), (z-V)) - torch.mm(search_policy, torch.log(P))
        print("Loss: ", loss)
        optimizer = self.optimizer(learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
