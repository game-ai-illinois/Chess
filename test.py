import chess
import numpy as np
import torch
from chess_env import *
from classes import *
from algorithm import *
import copy
mport time
device = "cuda:0" if torch.cuda.is_available() else "cpu"

t0 = time.time()
board = chess.Board()
input_size = 12
n_iterations = 400
white = NN( input_size, 256, 19).to(device)
black = NN( input_size, 256, 19).to(device)
white.load_state_dict(torch.load("./models/model.json"))
black.load_state_dict(torch.load("./models/model.json"))
self_play_n_games = 9#400 #on the paper it's 25k and 16k simulation b4 each move, though the paper is on go where average legal moves are much higher
for iteration in range(n_iterations):
    #self play
    v_resign = 0.05 #arbitrary high number
    archive = []
    for game in range(self_play_n_games):
        if game <= self_play_n_games/10 :#for the first 10% of games play until the end to find v_resign
            smaller_archive, winner_is_white = test(black, black, device=device)
        else:
            smaller_archive, winner_is_white = test(black, black, device=device)
            # smaller_archive, winner_is_white = test(black, black, v_resign=v_resign, device=device)# we self train black
        if winner_is_white != 0.5: #if draw, don't add it to archive
            archive += smaller_archive
    train(black, archive, device=device)
    del archive # empty archive to relieve ram
    #it is my understanding that declining temperature and dirichlet noise is only used in self play
    #evaluate
    # evaluate_n_games = 80
    # n_white_wins = 0
    # for game in range(evaluate_n_games):
    #     _, winner_is_white = test(white, black, self_play = False, device=device) #if draw, winner_is_white == 0.5
    #     n_white_wins += winner_is_white
    # print("ratio: ", (evaluate_n_games - n_white_wins)/evaluate_n_games)
    # if (evaluate_n_games - n_white_wins)/evaluate_n_games > 0.55: #if black won at least 55 % of the games then black becomes the white network
    #     print("black is the new best")
    #     torch.save(black.state_dict(), "./models/model.json") #save the model
    #     print("model saved")
    #     white = copy.deepcopy(black)
    # else:
    #     print("white is still the best")
    #     black = copy.deepcopy(white) # reassign black as the current best: white
    # print("iteration: ", iteration)

# print(archive)
print("finished")
print("elapsed time: ", time.time() - t0)
