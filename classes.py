import numpy as np
import torch
import torch.nn as nn
from chess_env import input_state, is_white, return_legal_moves
import chess
import random
import math
import time

#TO DO: add virtual loss variable in backup see pg.22


class NN(nn.Module):
    """
    A Neural Network Class that plays the role of the neural network function
    in the paper "Mastering the game of Go without human knowledge"
    """
    def __init__(self,
                    input_size,
                    num_features,
                    num_residual_layers,
                    action_depth=4672,
                    board_width=8,
                    learning_rate = 0.001,
                    C = 0.0001, # L2 regularization
                    batch_size=32):

        super(NN, self).__init__()

        self.batch_size = batch_size

        self.tower = nn.Sequential(
            ConvBlock(input_size, num_features)
        )

        for i in range(num_residual_layers):
            self.tower.add_module("resid"+str(i+1),
                                    ResidualBlock(num_features))
        # torch.nn.init.xavier_uniform(self.tower.weight)

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay= C)
        # self.optimizer = torch.optim.sgd(self.parameters(), lr=learning_rate, weight_decay= C, momentum = 0.9)

    # def train(self, train_data):


    def run(self, state, avail_actions, device="cpu"):
        '''
        runs the network with data (analogous to "forward") to obtain
        policy (P), a vector quantity, and value (V), a scalar quantity
        both the state and avail_actions are assumed to be np arrays
        '''
        state = torch.from_numpy(state).float().to(device) #turn state and avail_actions into torch tensors
        avail_actions = torch.from_numpy(avail_actions.astype(np.float32)).to(device)
        tower = self.tower(state)
        policy_logits = self.policy_head(tower)
        policy_logits = (avail_actions * (policy_logits- torch.min(policy_logits)))
        # print("sum: ", torch.sum(policy_logits))
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


    # def optimizer(self, learning_rate, C):
    #     return torch.optim.Adam({self.tower.parameters(), self.policy_head.parameters(), self.value_head.parameters()}, lr=learning_rate, weight_decay= C)

    def optimization(self, archive, device="cpu"):
        '''
        trains the network with given training data
        '''
        """
        state = np.empty((1, 12, 8, 8))
        search_policy =  np.empty((1, 4672))
        z = []
        for data in archive:
            np.vstack((state, data[0]))
            np.vstack((search_policy, data[1]))
            z.append(data[-1])
        z = torch.FloatTensor(z)  #turn list into torch tensor
        avail_actions = search_policy != 0
        P, V = self.run(state, avail_actions)
        P[P == 0] = 1 # assign zero values to one so the log with make it zero
        loss = torch.mm((z-V), (z-V).t()) - torch.mm(torch.from_numpy(search_policy).float(), torch.log(P).t())
        # print("loss first term: ",torch.mm((z-V), (z-V).t()))
        # print("loss secod term ", torch.mm(torch.from_numpy(search_policy).float(), torch.log(P).t()))
        # print(torch.from_numpy(search_policy).float())
        # print(torch.log(P).t())
        print("Loss: ", loss)
        # optimizer = self.optimizer(learning_rate, C)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        """

        state = []
        search_policy =  []
        z = []
        for data in archive:
            state.append(data[0])
            search_policy.append(data[1])
            z.append(data[-1])
        state = np.vstack(state)
        search_policy = np.vstack(search_policy)
        z = torch.FloatTensor(z).to(device)  #turn list into torch tensor
        avail_actions = search_policy != 0
        indices = list(range(len(archive)))
        random.shuffle(indices)
        count = 0
        for i in range(math.ceil(len(archive) / self.batch_size)): #not sure if math ceil is necessary bc len() gives int
            curr_batch_size = min(self.batch_size, len(archive) - count)
            upper = count + curr_batch_size
            curr_indices = indices[count : upper]
            batch_state, batch_search_policy, batch_actions = state[curr_indices], search_policy[curr_indices], avail_actions[curr_indices]
            batch_z = z[curr_indices]
            P, V = self.run(batch_state, batch_actions, device=device)
            P[P == 0] = 1 # assign zero values to one so the log with make it zero
            # print(batch_z.shape)
            # print(V.shape)
            # print((batch_z-V).shape)
            # print((batch_z-V.t()).shape)
            # print(torch.from_numpy(batch_search_policy).shape)
            # print(torch.diag(torch.mm(torch.from_numpy(batch_search_policy).float(), torch.log(P).t())).shape)
            loss_first_term = torch.mm((batch_z-V.t()).t(), (batch_z-V.t()))  # shape = (batch_size, batch_size)
            # print(loss_first_term.shape)
            # print(loss_first_term >= 0)
            # print("log p: ", (torch.log(P).numpy() >= 0).all())
            # print("pi : ", (batch_search_policy >= 0).all())
            loss_second_term = torch.mm(torch.from_numpy(batch_search_policy).float().to(device), torch.log(P).t()) # shape = (batch_size, batch_size)
            # print(loss_second_term.shape)
            # print(loss_second_term )
            # print("batch search policy: ", torch.from_numpy(batch_search_policy).float() >= 0)
            # print("log: ",torch.log(P).t() >= 0)
            loss = torch.diag(loss_first_term + loss_second_term) #take the diag of the two terms to obtain the shape (batch_size, 1)
            print("loss shape: ", loss.shape)
            # print("loss non mean: ", loss)
            loss = torch.mean(loss)
            print("Loss: ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count += curr_batch_size



class Edge:
    """
    an edge class
    """
    def __init__(self, parent_node, probability, move_idx, move_text):
        self.parent_node_ = parent_node #node that the edge belongs to
        #if parent node is none, then the child node is root
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
        # print("destruct edge")
        # I wonder if I have to desctuct "self" as well

    def rollback(self, value):
        '''
        recursive backtracking method for MCTS
        '''
        self.N_ += 1
        self.W_ += value
        self.Q_ = self.W_ / self.N_
        if self.parent_node_ != None :
            self.parent_node_.parent_edge_.rollback(value)

class TraversedEdge:

    def __init__(self, edge_dict, ind, child_node=None):
        self.edge_dict = edge_dict
        self.ind = ind
        self.parent_node_ = edge_dict.parent_node
        self.child_node_ = child_node
        self.N_, self.V_, self.W_, self.P_ = self.edge_dict.edges_[ind]
        self.move_idx_ = ind
        self.move_text_ = self.edge_dict.move_dict[ind]

        """
        if (str(self.move_inds[ind]) in self.edge_dict.move_inds):
            self.move_text_ = self.edge_dict.move_dict[ind]
        """

    def rollback(self, value, move_count_dict):

        if (self.parent_node_ == None):
            return
        dict_in = self.parent_node_.s # + " WITH ACTION " + str(self.ind)

        if (dict_in not in move_count_dict.keys()):
            move_count_dict[dict_in] = {str(self.ind): [1, value, value, self.P_]}
            self.edge_dict.edges_[self.ind,0] += 1
            self.edge_dict.edges_[self.ind,1] += value
            self.edge_dict.edges_[self.ind,2] = self.edge_dict.edges_[self.ind,1] / self.edge_dict.edges_[self.ind,0]
        else:
            [N,V,W,P] = move_count_dict[dict_in]
            N += 1
            V += value
            W = V / N
            move_count_dict[dict_in][str(self.ind)] = [N,V,W,P]
            self.edge_dict.edges_[self.ind,0] = N
            self.edge_dict.edges_[self.ind,1] = V
            self.edge_dict.edges_[self.ind,2] = W
        if self.parent_node_ != None and self.parent_node_.parent_edge_ != None:
            self.parent_node_.parent_edge_.rollback(value, move_count_dict)
        self.N_, self.V_, self.W_, self.P_ = self.edge_dict.edges_[self.ind]




class ChildrenEdgeDict:

    """
        Class for storing all the information from all possible child edges in one large array

        Vectorized data structure to speed up execution of following logic in getChildrenEdges:

        for idx in range(4672): #as the legal_moves_array.shape is (1, 4672)
            if legal_moves_array[0, idx] != 0:
                edge = Edge(self, P[0, idx], idx, move_dict[idx])
                self.children_edges_.append(edge)

        size: number of edges
        parent_node: parent node pointer
        legal_moves_array: binary array indicating which moves are available
        probabilities: prob array of size (size,)
        move_dict: dictionary mapping action indices to next states

    """
    def __init__(self, size, parent_node, legal_moves_array, probabilities, move_dict, move_count_dict):
        self.edges_ = np.zeros((size, 4))
        self.parent_node = parent_node
        self.edges_[:,:3] = 0.0 # [0,1,2] -> [N_, W_, Q_]

        self.legal_inds = legal_moves_array.nonzero()
        self.move_inds, self.move_texts = list(move_dict.keys()), list(move_dict.values())
        #print(list(self.legal_inds))
        self.move_dict = move_dict
        #print(self.legal_inds, self.move_inds)
        #assert ((self.legal_inds == self.move_inds))
        self.edges_[self.legal_inds,3] = probabilities[self.legal_inds] # [3] -> [P_]

    def __getitem__(self, key):
        return TraversedEdge(self, key)


class Node:
    """
    A Node class that represents a node in a MCTS as stated in the paper
    "Mastering the game of Go without human knowledge"
    """
    def __init__(self, board, parent_edge, device="cpu", resign=False):
        '''
        We use the state and Neural Network to obtain prior probability (P_)
        and evaluation of value of the node (V_)
        '''
        self.board_ = board # state of the chess env
        self.s = board.fen()
        self.parent_edge_ = parent_edge # None value idicates that the node is starting game node
        #root nodes should otherwise still have parent edges to access its value for stop condition
        self.children_edges_ = [] # list of edges nodes
        board_state_string = self.board_.fen()
        self.state_ = input_state(board_state_string)
        self.is_black_ = not is_white(board_state_string)
        self.resign = resign

    def __del__(self):

        """
        Delete function for Node class
        """
        del self.board_
        del self.children_edges_
        self.board_ = None
        del self.parent_edge_
        # print("destruct node")

    def getChildrenEdges(self, NN, move_count_dict, device):
        '''
        obtains all the children nodes from interacting with the chess env
        '''
        board_state_string = self.s
        state_array = input_state(board_state_string) #turn into arrays for NN
        is_black = not is_white(board_state_string)
        legal_moves_array, move_dict = return_legal_moves(self.board_, is_black)
        self.move_dict = move_dict
        # print("state array shape: ", state_array.shape)
        # print("legal array shape: ", legal_moves_array.shape)
        # print(type(legal_moves_array))
        t1 = time.time()
        P, _ = (NN.run(state_array, legal_moves_array, device=device)) # P.shape is (1, 4672)
        P = P.cpu().data.numpy()
        t2 = time.time()
        #print("forward time: %f" % (time.time() - t1))
        # print("P shape: ", P.shape)


        # UNCOMMENT THIS MULTI LINE COMMENT TO ACTIVATE VECTORIZED EDGE PROCESSING

        # __init__(self, size, parent_node, legal_moves_array, probabilities, move_dict)
        self.children_edges_ = ChildrenEdgeDict(4672, self,legal_moves_array, P, move_dict, move_count_dict)
        if (board_state_string in move_count_dict.keys()):
            edges = move_count_dict[board_state_string]
            arr = self.children_edges_.edges_
            inds = edges.keys()
            for i in range(len(edges.keys())):
                arr[int(inds[i])] = edges[inds[i]]

        """
        # BEGIN OF OLD NONVECTORIZED CODE
        """
        """
        for idx in range(4672): #as the legal_moves_array.shape is (1, 4672)
            if legal_moves_array[0, idx] != 0:
                # print("legal move recognized")
                edge = Edge(self, P[0, idx], idx, move_dict[idx])
                # print("ege: ", edge)
                self.children_edges_.append(edge)
        """
        """
        # END OF OLD NONVECTORIZED CODE
        """
        # print("child edges size: ", len(self.children_edges_))
        # print("child edges appended")
        # print(self.children_edges_)

    def select(self, c_puct, self_play):
        '''
        returns indx of a child edge to select to play
        Clarification: this function does not pick a move to play
        from the given parameters pick a child node to traverse to
        the P_ (prior probability) is obtained from running the Neural Network
        when the node was initialized. We assume P is a np array
        '''
        if self.children_edges_ == []:
            print("children nodes not initialized")
            return None



        # UNCOMMENT THIS MULTI LINE COMMENT TO ACTIVATE VECTORIZED EDGE PROCESSING
        arr = self.children_edges_.edges_
        valid_moves = list(self.move_dict.keys())
        [N_child, W, Q, P] = arr[valid_moves].transpose()


        """
        # BEGIN OF OLD NONVECTORIZED CODE
        """
        """
        Q = []# initialize to obtain Q values
        N_child = [] # initialize to obtain N of each child
        P = []
        for edge in self.children_edges_ :
            Q.append(edge.Q_)
            N_child.append(edge.N_)
            P.append(edge.P_)
        Q = np.array(Q)
        N_child = np.array(N_child)
        P = np.array(P)
        """
        """
        # END OF OLD NONVECTORIZED CODE
        """
        if self_play:
            eps = 0.25
            # print("P b4 dirichlet: ", P)
            P = (1- eps)*P + eps*np.random.dirichlet(0.3 * np.ones(P.shape)) #add dirichlet noise if self play
            # print("P after dirichlet: ", P)
            #dirichlet parameter is 0.3 for chess from https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
        N_all = np.sum(N_child) + 1
        U = c_puct* P* np.sqrt(N_all)/ (1 + N_child) #obtain the U values
        #print(list(Q+U), list(P))
        selected_child_idx = np.argmax(Q+U)
        return valid_moves[selected_child_idx]

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
