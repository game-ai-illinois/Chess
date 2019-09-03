import chess
import numpy as np
import torch
from chess_env import *
from classes import *
from algorithm import *
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import time
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    t0 = time.time()
    mp.set_start_method("spawn")
    board = chess.Board()
    input_size = 12
    n_iterations = 400
    white = NN( input_size, 256, 19).to(device)
    black = NN( input_size, 256, 19).to(device)
    white.load_state_dict(torch.load("./models/model.json"))
    black.load_state_dict(torch.load("./models/model.json"))
    self_play_n_games = 400# was 9 #on the paper it's 25k and 16k simulation b4 each move, though the paper is on go where average legal moves are much higher
    manager = Manager()
    archive = manager.list()


    for iteration in range(n_iterations):
        #self play
        p1 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//3, white, black, device))
        p2 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//3, white, black, device))
        p3 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//3, white, black, device))
        # p4 = mp.Process(target=test_multiprocess, args=(self_play_n_games//4, white, black, device))
        p1.start()
        p2.start()
        p3.start()
        # p4.start()
        p1.join()
        p2.join()
        p3.join()
        # p4.join()
        if archive == []:
            print("archive empty")
        train(black, archive, device=device)

        del archive  # empty archive to relieve ram
        archive = manager.list()
        #it is my understanding that declining temperature and dirichlet noise is only used in self play
        #evaluate
        evaluate_n_games = 80
        n_white_wins = 0.0
        how_many_black_won = 0
        how_many_draws = 0
        for game in range(evaluate_n_games):
            _, winner_is_white = test(white, black, self_play = False, device=device) #if draw, winner_is_white == 0.5
            n_white_wins += winner_is_white
            if winner_is_white ==1.0 :
                how_many_black_won += 1
            elif winner_is_white == 0.5:
                how_many_draws += 1
        print("ratio: ", (evaluate_n_games - n_white_wins)/evaluate_n_games)
        print("number of blacks win: ", how_many_black_won)
        print("number of draws: ", how_many_draws)
        if (evaluate_n_games - n_white_wins)/evaluate_n_games > 0.55: #if black won at least 55 % of the games then black becomes the new current best network
            print("black is the new best")
            torch.save(black.state_dict(), "./models/model.json") #save the model
            print("model saved")
            white = copy.deepcopy(black)
        else:
            print("white is still the best")
            black = copy.deepcopy(white) # reassign black as the current best: white
        print("iteration: ", iteration)

    # print(archive)
    print("finished")
    print("elapsed time: ", time.time() - t0)
