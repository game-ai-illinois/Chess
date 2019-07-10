import chess
import numpy as np 


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
    input_state = np.zeros([12, 8, 8])
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
            input_state[piece_idx, (7 - row), col] = 1
            col += 1
    return input_state

def is_white(string)
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
    text manifests row numbers through numbers
    1 to 8, and manifests col numbers through
    letters a to h (lower cases)
    position on the other hand is a tuple of
    numerical coordinates that go from index
    zero to seven ie (0,7)
    """
    #under perspective of black player, the position is flipped
    y = int(string[1] - 1)
    x = string[0] - ord('a')
    if is_black:
        x = 7 - x
        y = 7 - y
    return (x, y)
    

def array_to_text(array, is_black):
    """
    takes in numpy array action and translates it 
    into action understandable for chess env
    """
    row_increment = 0
    col_increment = 0
    if is_black: #this is because the command changes if the action is taken by black player
        row_increment = -row_increment
        col_increment = -col_increment
    

    
def play(board, NN, legal_moves_list, board_state_string):
    """
    takes in neural network and the list of legal moves from chess env
    obtains numpy array of probability distribution of legal moves
    takes a action from the numpy array, returns numpy array equivalent
    of that action and text equivalent of that action for chess env
    """
    legal_moves_array = np.zeros([4672]) #initialize array of legal moves 
    is_black = not is_white(board_state_string)
    move_text = array_to_text(move_array, is_black)
    env_move = chess.Move.from_uci(move_text)
    board.push(env_move)




"""
in our neural network, position is prioritized, then number of squares traversed,
and then the direction of the traverse. if the order of: 64 nodes (8 by 8 chess grid),
[1 to 7], and [N,NE,E,SE,S,SW,W,NW]
ie, first 64 nodes all correspond to moving one square in the N direction. Then the next 64 correspond to 
moving two squares in N direction, the eighth group of 64 nodes correspond to moving a square in the NE direction
"""

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

def method():
    squares_travel = 0
    start_x, start_y = text_to_position(string[0:2], is_black)
    end_x, end_y = text_to_position(string[2:4], is_black)
    distance_travel_x, distance_travel_y = abs(start_x - end_x) + abs(start_y - end_y)
    if is_straight(distance_travel_x, distance_travel_y):
        squares_travel = distance_travel_x + distance_travel_y
    elif is_knight_move(distance_travel_x, distance_travel_y):
        

    else: #diagonal move

