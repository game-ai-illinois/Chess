import numpy as np
import torch
import torch.nn as nn
import chess
from classes import *
from chess_env import *

def tree_search(node, NN, remain_iter, c_puct):
    '''
    runs MCTS simulation once
    remain_iter argument gives remaining number of simulations iterations
    that we still need to do. The number is decremented by one everytime
    a simulation goes to the next child node
    '''
    if remain_iter == 0: #evaluate value from NN
        state_array = node.state_ #obtain state for NN
        is_black = node.is_black_
        _, value = (NN.run(state_array, np.ones(4672)))
        node.parent_edge_.rollback(value) #rollback value
    else:
        if node.children_edges_ == [] : #if children edges not expanded
            node.getChildrenEdges(NN)#expand children edges
        select_idx = node.select(c_puct)
        selected_edge = node.children_edges_[select_idx]
        if selected_edge.child_node_ == None: #check if the edge has child node. if it doesn't, initiate node
            new_board = node.board_.copy()
            edge_move = chess.Move.from_uci(selected_edge.move_text_)
            new_board.push(edge_move)
            selected_edge.child_node_ = Node(new_board, parent_edge = selected_edge)
        tree_search(selected_edge.child_node_, NN, remain_iter-1, c_puct)

def pickMove(node, temp, archive):
    '''
    Makes a move and picks the next node (state) to go to
    archive is a list of data to later used for training
    '''
    N_vector = [] # vector of N^(1/temp) values of all children
    N_vector_array = np.zeros(4672)
    # print("node.children_edges_ length: ",len(node.children_edges_))
    for child_edge in node.children_edges_:
        value = child_edge.N_**(1/temp)
        # print("value: ", value)
        N_vector.append(value)
        N_vector_array[child_edge.move_idx_] = value
    N_vector = np.array(N_vector) # make it np array
    # print("N_vector: ", N_vector)
    N_vector = N_vector/np.sum(N_vector) # normalize to make it a probability distribution
    N_vector_array = N_vector_array / np.sum(N_vector_array) 
    white_turn = is_white(node.board_.fen())
    archive.append([node.state_, N_vector_array, white_turn, None]) # before picking the move, add data to archive. The winner is added later
    # print("N_vector shape: ", N_vector.shape)
    next_edge_idx = np.random.choice([value for value in range(N_vector.shape[0])], p = N_vector)
    # next_move_arg picks the index of the N_vector according to its probability distribution
    next_move_edge = node.children_edges_[next_edge_idx] # picks the next child
    if next_move_edge.child_node_ == None: #check if the edge has child node. if it doesn't, initiate node
        new_board = node.board_.board.copy()
        edge_move = chess.Move.from_uci(selected_edge.move_text_)
        new_board.push(edge_move)
        next_move_edge.child_node_ = Node(new_board, parent_edge = selected_edge)
    old_node = node
    new_node = next_move_edge.child_node_
    node = new_node
    del old_node # destruct the parent node and
    new_node.parent_edge_.parent_node_ = None #the child node is now root node, thus its parent edge points to none
    print("archive length: ", len(archive))
    return new_node

def stop_condition(node, v_resign):
    if node.parent_edge_ == None : #start position node
        return False
    #obtain best child value
    values = []
    for edge in node.children_edges_ : 
        values.append(edge.Q_)
    best_value = max(values)
    if len(values) == 0:
        print("Error: children edges not expanded")
        return True
    if (node.parent_edge_.Q_ and best_value) < v_resign :
        return True
    else:
        return False



def test( newgame= True):
    if newgame == True:
        nn = NN(12, 1, 1)
    c_puct = 0.5
    MCTS_iter = 10 #number of time we iterate over
    board = chess.Board() # initialize new game
    start_edge = Edge(None, 0, 0, "a0b1") #initialize edge to notify starting node as root
    current_node = Node(board, start_edge) #initialize node
    archive = [] #initialize archive
    temp = 0.4 
    done = False
    while not done:
        tree_search(current_node, nn, MCTS_iter, c_puct)
        # if stop_condition(current_node, 0.5):
        #     done = True
        #     break
        current_node = pickMove(current_node, temp, archive) 
        done = True

    winner_is_white = 1 #not sure how to figure this out yet
    postProcess(archive, winner_is_white)
    return archive

# def train(archive):

    

def postProcess(archive, winner_is_white):

    """
    when the game is finished and data is updated for training
    """
    for data in archive:
        if data[-2] == winner_is_white: #if winner matches with the current player
            data[-1] = 1.0
        else:
            data[-1] = -1.0
    print("archive length: ", len(archive))

