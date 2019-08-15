import chess
import numpy as np
from chess_env import *
from classes import *
from algorithm import *

board = chess.Board()
input_size = 12
network = NN( input_size, 1, 1)

learning_rate = 0.001 # random value
C = 0.001 # random value
for i in range(20):
    archive = test()
    train(network, archive, learning_rate, C)
# print(archive)
print("finished")