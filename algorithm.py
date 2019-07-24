import numpy as np
import torch
import torch.nn as nn
import chess
from classes import *
from chess_env import *

def tree_search(node, NN, L, c_puct):
    if L == 0: #evaluate value from NN
        board_state_string = node.board_.fen()
        state_array = input_state(board_state_string) #turn into arrays for NN
        is_black = not is_white(board_state_string)
        _, value = NN.run(state_array)
        node.rollback(value) #rollback value
    else:
        if node.children_edges_ == [] : #if children edges not expanded
            node.getChildrenEdges(NN)#expand children edges
        select_idx = node.select(c_puct)
        selected_edge = node.children_edges_[select_idx]
        if selected_edge.child_node_ == None: #check if the edge has child node. if it doesn't, initiate node
            new_board = node.board_.board.copy()
            edge_move = chess.Move.from_uci(selected_edge.move_text_)
            new_board.push(edge_move)
            selected_edge.child_node_ = Node(new_board, parent_edge = selected_edge)
        tree_search(selected_edge.child_node_, NN, L-1, c_puct)
