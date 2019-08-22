import numpy as np
import torch
import torch.nn as nn
import chess
from classes import *
from chess_env import *

def tree_search(node, NN, remain_iter, c_puct, self_play, device="cpu"):
    '''
    runs MCTS simulation once
    remain_iter argument gives remaining number of simulations iterations
    that we still need to do. The number is decremented by one everytime
    a simulation goes to the next child node
    '''
    if remain_iter == 0 or node.board_.is_game_over(): #evaluate value from NN
        state_array = node.state_ #obtain state for NN
        is_black = node.is_black_
        _, value = (NN.run(state_array, np.ones(4672), device=device))
        node.parent_edge_.rollback(value) #rollback value
    else:
        if node.children_edges_ == [] : #if children edges not expanded
            node.getChildrenEdges(NN, device=device)#expand children edges
            # print("children edges size: ", len(node.children_edges_))
            # print("children edges expanded")
        select_idx = node.select(c_puct, self_play)
        selected_edge = node.children_edges_[select_idx]
        if selected_edge.child_node_ == None: #check if the edge has child node. if it doesn't, initiate node
            new_board = node.board_.copy()
            edge_move = chess.Move.from_uci(selected_edge.move_text_)
            new_board.push(edge_move)
            # print("picked move: ", selected_edge.move_text_)
            selected_edge.child_node_ = Node(new_board, parent_edge = selected_edge)
        tree_search(selected_edge.child_node_, NN, remain_iter-1, c_puct, self_play, device=device)

def pickMove(node, temp, archive, v_resign):
    '''
    Makes a move and picks the next node (state) to go to
    archive is a list of data to later used for training
    '''
    if v_resign != None: #if we are assigned a v_resign value
        child_action_values = []
        for child_edge in node.children_edges_: #obtain the aciton values of child edges
            child_action_values.append(child_edge.Q_)
        if node.parent_edge_.Q_ and max(child_action_values) < v_resign : #resign
            out_node = Node(node.board_.board.copy(), selected_edge, resign=True)
            return out_node
    N_vector = [] # vector of N^(1/temp) values of all children
    N_vector_array = np.zeros(4672)
    # print("node.children_edges_ length: ",len(node.children_edges_))
    for child_edge in node.children_edges_:
        # print("child_edge.N_: ",child_edge.N_)
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
        next_move_edge.child_node_ = Node(new_board, selected_edge)
    new_node = next_move_edge.child_node_
    del node # destruct the old parent node
    new_node.parent_edge_.parent_node_ = None #the child node is now root node, thus its parent edge points to none
    # print("check new node pointer: ", next_move_edge.child_node_.parent_edge_.parent_node_ == None)
    # print("archive length: ", len(archive))
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



def test( white, black, v_resign = None, self_play = True, device="cpu"):
    """
    function that takes in white and black players (neural network) and
    make them play against one another
    """
    c_puct = 0.5
    MCTS_iter = 25 # how deep we do MCTS
    board = chess.Board() # initialize new game
    start_edge = Edge(None, 0, 0, "a0b1") #initialize edge to notify starting node as root
    current_node = Node(board, start_edge) #initialize node
    archive = [] #initialize archive
    done = False
    num_steps = 0
    resign = False
    time_out = False
    step = 1
    if self_play: #when test is used for self play, temp starts with 1
        temp = 1
    else:
        temp = 0.01 #when test is used for evaluation, temp starts very small
    which_network = 0
    while not done:
        # print("current board: \n", current_node.board_)
        # print("tree search start")
        if which_network % 2 == 0:
            network = white
        else:
            network = black
        tree_search(current_node, network, MCTS_iter, c_puct, self_play, device=device)
        # if stop_condition(current_node, 0.5):
        #     done = True
        #     break
        # print("temp: ", temp)
        current_node = pickMove(current_node, temp, archive, v_resign)
        if current_node == "resign":
            resign = True
            print("resign game")
            break
        if current_node.board_.is_game_over():
            done = True
            print("done game")
        # print("board: ", current_node.board_)
        num_steps += 1
        if num_steps >= 200 :
            time_out = True
            # print("time out")
            break
        if step > 30:
            temp = 0.01

    winner_is_white = 0.0
    current_player_is_white = is_white(current_node.board_.fen())
    # print("is current player white?: ", current_player_is_white)
    if resign:
        if not current_player_is_white:
            winner_is_white = 1.0
        # else we do nothing as it's already 0.0
    else: #if gone all the way
        # print("result: ",current_node.board_.result())
        if current_node.board_.result() == ("0-1" or "*"): # loss
            if not current_player_is_white:
                winner_is_white = 1.0
            print("lose")
            # else we do nothing as it's already 0.0
        elif current_node.board_.result() == "1/2-1/2" : # draw
            winner_is_white = 0.5
            print("draw")
        elif time_out : # draw
            winner_is_white = 0.5 # score is same as draw
            print("time_out")
        else: # win
            if current_player_is_white:
                winner_is_white = 1.0
            print("Win")
            # else we do nothing as it's already 0.0
    # print("winner_is_white ", winner_is_white)
    postProcess(archive, winner_is_white)
    return archive, winner_is_white

def train(NN, archive, device="cpu"):
    NN.optimization(archive, device=device)

def postProcess(archive, winner_is_white):

    """
    when the game is finished and data is updated for training
    """
    for data in archive:
        if data[-2] == winner_is_white: #if winner matches with the current player
            data[-1] = 1.0
        else:
            data[-1] = -1.0
    # print("archive length: ", len(archive))
