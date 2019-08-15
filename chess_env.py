import chess
import numpy as np
import torch


def layer_num(char):
    """
    helper funtion for input_state() method
    bunch of if statement to tell you which
    layer of the state the chess board.fen()
    character belongs to
    """
    if char == 'p' :
        return 0
    elif char == 'r' :
        return 1
    elif char == 'n' :
        return 2
    elif char == 'b' :
        return 3
    elif char == 'q' :
        return 4
    elif char == 'k' :
        return 5
    elif char == 'P' :
        return 6
    elif char == 'R' :
        return 7
    elif char == 'N' :
        return 8
    elif char == 'B' :
        return 9
    elif char == 'Q' :
        return 10
    elif char == 'K' :
        return 11
    else:
        print("invalid chess character")
        print(char)
        return None

def input_state(string):
    """
    takes in string variable of python-chess board.fen()
    and creates a np array input state for AI
    the input state is (2 * 6) x 8 x 8,
    basically 12 layers of 8 by 8 board
    where the first dimension is the layers for
    each piece type of the player's pieces and
    then opponent's pieces. The order goes as:
    pawn, rook, bishop, knight, queen, king.
    """
    input_state = np.zeros([1,12, 8, 8]) #the extra dim is there bc torch requires an additional dim for batch run
    row = 0
    col = 0
    for char in string:
        if char == ' ':
            break
        elif char == '/': # new row
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            piece_idx = layer_num(char)
            input_state[0, piece_idx, (7 - row), col] = 1
            col += 1
    return input_state

def is_white(string):
    """
    takes in string variable of python-chess board.fen()
    and determins if the string is in the perspective of
    white player by looking to see if there's w in the
    string
    """
    result = False
    for char in string:
        if char == 'w' :
            result = True
    return result

def text_to_position(string, is_black):
    """
    returns position of the chesss piece
    from text coordinates eg: a1, h3, etc
    under the perspective of the player
    whose turn is the current turn
    text manifests row numbers through numbers
    1 to 8, and manifests col numbers through
    letters a to h (lower cases)
    position on the other hand is a tuple of
    numerical coordinates that go from index
    zero to seven ie (0,7)
    """
    # print("string: ", string)
    y = int(string[1]) - 1
    x = ord(string[0]) - ord('a')
    if is_black: #under perspective of black player, the position is flipped
        x = 7 - x
        y = 7 - y
    # print("x,y: ", (x, y))
    return (x, y)


"""
not used function
"""
# def array_to_text(array, is_black):
#     """
#     takes in numpy array action and translates it
#     into action understandable for chess env
#     """
#     row_increment = 0
#     col_increment = 0
#     if is_black: #this is because the command changes if the action is taken by black player
#         row_increment = -row_increment
#         col_increment = -col_increment
#     return


def is_straight(distance_travel_x, distance_travel_y):
    """
    Verifying that it's not diagonal or a knight move.
    We do this by checking that the distance traveled
    is only more than zero in one of the axis
    """
    if (distance_travel_x > 0 and distance_travel_y == 0) or (distance_travel_x == 0 and distance_travel_y > 0):
        return True
    else:
        return False

def is_knight_move(distance_travel_x, distance_travel_y):
    if is_straight(distance_travel_x, distance_travel_y):
        return False
    else:
        if not distance_travel_x == distance_travel_y:
            return True
        else:
            return False



def squaresANDdirections(displacement_x, displacement_y):
    squares_travel = 0
    direction = 0 # index of the direction of the move in the list [N,NE,E,SE,S,SW,W,NW]
    if is_straight(abs(displacement_x), abs(displacement_y)): #if straight move
        squares_travel = abs(displacement_x) + abs(displacement_y)
        if displacement_y > 0: #if N
            direction = 0
        elif displacement_x > 0: # f E
            direction = 2
        elif displacement_y < 0: #if S
            direction = 4
        else: #if W
            direction = 6
    else: #if diagonal move
        squares_travel = abs(displacement_x)
        if displacement_y > 0 and displacement_x > 0: #if NE
            direction = 1
        elif displacement_y < 0 and displacement_x > 0: #if SE
            direction = 3
        elif displacement_y < 0 and displacement_x < 0: #if SW
            direction = 5
        else: #if NW
            direction = 7
    return squares_travel, direction

def if_under_promotion(move_string):
    return_value = False
    if len(move_string) > 4 and move_string[-1] != 'q':
        return_value = True
    return return_value

def position_index(x, y):
    """
    uses x and y coordinates
    to return position index (only position)
    the index of the action node layer represents
    position from left to right, bottom to top,
    (like reading a book, but from bottom of the page to top)
    """
    position_action_idx = x + y*8
    return position_action_idx


def legal_move_array_index(move_string, is_black, move_dict):
    start_x, start_y = text_to_position(move_string[0:2], is_black)
    end_x, end_y = text_to_position(move_string[2:4], is_black)
    displacement_x, displacement_y = (end_x- start_x), (end_y - start_y)
    # print("displacements: ",displacement_x," ", displacement_y )
    action_idx = 0 #initialize action index
    position_idx = position_index(start_x, start_y)
    # print("position_idx: ", position_idx)
    if is_knight_move(abs(displacement_x), abs(displacement_y)):
        # print("is Knight move")
        if displacement_x > 0 and displacement_y > 0 :
            if displacement_y > displacement_x: #if NNE
                # print("NNE")
                action_idx = position_idx + 8*8*56
            else: #if NEE
                # print("NEE")
                action_idx = position_idx + 8*8*56 + 8*8
        elif displacement_x > 0 and displacement_y < 0:
            if abs(displacement_y) < displacement_x: #if SEE
                # print("SEE")
                action_idx = position_idx + 8*8*56 + 8*8*2
            else: #if SSE
                # print("SSE")
                action_idx = position_idx + 8*8*56 + 8*8*3
        elif displacement_x < 0 and displacement_y < 0:
            if abs(displacement_y) > abs(displacement_x): #if SSW
                # print("SSW")
                action_idx = position_idx + 8*8*56 + 8*8*4
            else: #if SWW
                # print("SWW")
                action_idx = position_idx + 8*8*56 + 8*8*5
        elif displacement_x < 0 and displacement_y > 0:
            if displacement_y < abs(displacement_x): #if NWW
                # print("NWW")
                action_idx = position_idx + 8*8*56 + 8*8*6
            else: #if NNW
                # print("NNW")
                action_idx = position_idx + 8*8*56 + 8*8*7
    elif if_under_promotion(move_string):
        squares, direction = squaresANDdirections(displacement_x, displacement_y)
        if direction == 0: #center move
            if move_string[-1] == 'b': #if bishop
                action_idx = position_idx + 8*8*(56+8)
            if move_string[-1] == 'n': #if knight
                action_idx = position_idx + 8*8*(56+8) + 8*8
            else: #if rook
                action_idx = position_idx + 8*8*(56+8) + 8*8*2
        elif direction == 1: #right move
            if move_string[-1] == 'b': #if bishop
                action_idx = position_idx + 8*8*(56+8) + 8*8*3
            if move_string[-1] == 'n': #if knight
                action_idx = position_idx + 8*8*(56+8) + 8*8*4
            else: #if rook
                action_idx = position_idx + 8*8*(56+8) + 8*8*5
        else: #left move
            if move_string[-1] == 'b': #if bishop
                action_idx = position_idx + 8*8*(56+8) + 8*8*6
            if move_string[-1] == 'n': #if knight
                action_idx = position_idx + 8*8*(56+8) + 8*8*7
            else: #if rook
                action_idx = position_idx + 8*8*(56+8) + 8*8*8
    else: #normal queen move
        squares, direction = squaresANDdirections(displacement_x, displacement_y)
        action_idx = position_idx + (squares -1)*64 + direction*(64*8)
    move_dict[action_idx] = move_string
    return action_idx


"""
in our neural network, position is prioritized, then number of squares traversed,
and then the direction of the traverse. if the order of: 64 nodes (8 by 8 chess grid),
[1 to 7], and [N,NE,E,SE,S,SW,W,NW]
ie, first 64 nodes all correspond to moving one square in the N direction. Then the next 64 correspond to
moving two squares in N direction, the eighth group of 64 nodes correspond to moving a square in the NE direction
in all, the queen moves takes the first 8 x 8 x 56 moves
the next 8 planes (8 x 8 x 8 nodes) indicate a knight move, starting from the north north east move,
and then rotating clockwise
the last nine planes indicates the nine different underpromotions. We order them such that
pawn's direction move is prioritized, and then underpromotion piece is prioritized in
such order: bishop, knight and rook. Therefore, the order goes:
center bishop, center knight, center rook, right bishop, right knight, right rook,
left bishop, left knight and left rook.

"""

def return_legal_moves(board, is_black):
    legal_moves_array = np.zeros([4672]) # initialize array of legal moves
    move_dict = {} #for translating back to move string
    # flag = 0
    for move in board.legal_moves:
        # flag += 1
        legal_move_array_idx = legal_move_array_index(move.uci(), is_black, move_dict)
        legal_moves_array[legal_move_array_idx] = 1
    legal_moves_array = legal_moves_array.reshape(1, *legal_moves_array.shape)
    # if flag > 0 :
    #     # print("moves exist")
    #     # print("board: ", board)
    #     # print(board.legal_moves)
    #     # print([move for move in board.legal_moves])
    # else:
    #     # print("moves don't exist")
    #     # print("board: ", board)
    #     # print("done game: " , done_game(board))
    #     # print(board.legal_moves)
    #     # print([move for move in board.legal_moves])
    return legal_moves_array, move_dict

def random_play(board, NN):
    """
    takes in neural network and the list of legal moves from chess env
    obtains numpy array of probability distribution of legal moves
    takes a action from the numpy array, returns numpy array equivalent
    of that action and text equivalent of that action for chess env
    """
    board_state_string = board.fen() # obtain state from board
    state_array = input_state(board_state_string) # turn state into an array format for NN
    is_black = not is_white(board_state_string)
    print("is black: ",is_black)
    legal_moves_array = np.zeros([4672]) # initialize array of legal moves
    legal_moves_array, move_dict = return_legal_moves(board, is_black)
    # print("state array shape: ", state_array.shape)
    # print("legal array sahpe: ", legal_moves_array.shape)
    legal_moves_prob_distribution, _ = (NN.run(state_array, legal_moves_array))  #we're assuming that NN forward runs the neural network
    # legal_moves_prob_distribution = legal_moves_prob_distribution / np.sum(legal_moves_prob_distribution) # normalize
    legal_moves_prob_distribution = legal_moves_prob_distribution.numpy().reshape(4672)
    # legal_moves_prob_distribution = legal_moves_prob_distribution - np.min(legal_moves_prob_distribution)
    # legal_moves_prob_distribution = legal_moves_prob_distribution /legal_moves_prob_distribution.sum()
    # print("legal_moves_prob_distribution sum ",abs(legal_moves_prob_distribution).sum())
    # print("legal_moves_prob_distribution sum ",(legal_moves_prob_distribution* legal_moves_arrayCopy).sum())
    # print("legal_moves_prob_distribution sum ",(legal_moves_prob_distribution).sum())
    action_idx = np.random.choice(4672, p = legal_moves_prob_distribution )
    print("action idx: ", action_idx)
    action_array = np.zeros([4672])
    action_array[action_idx] = 1
    move_text = move_dict[action_idx]
    print("move text: ", move_text)
    env_move = chess.Move.from_uci(move_text)
    board.push(env_move)
    return action_array


def done_game(board):
    if board.is_game_over():
        print("board result: ",board.result())
        return board.result()
    else:
        return False
    # if  board.is_insufficient_material() or board.is_variant_draw() :
    #     return "draw"
    # elif board.is_variant_win():
    #     return "win"
    # elif board.is_variant_loss():
    #     return "loss"
    # elif board.is_game_over():
    #     print("anomaly")
    #     return False
    # else:
    #     return False
