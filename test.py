import chess
import numpy as np
from chess_env import *
from classes import *
from algorithm import *

board = chess.Board()
input_size = 12
network = NN( input_size, 1, 1)
archive = test()
# print(archive)
print("finished")