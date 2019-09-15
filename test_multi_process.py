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
    self_play_n_games = 480# was 480 #on the paper it's 25k and 16k simulation b4 each move, though the paper is on go where average legal moves are much higher
    manager = Manager()
    archive = manager.list()
    for iteration in range(n_iterations): # prev n_iterations
        #self play
        white = NN( input_size, 256, 19).to(device)
        black = NN( input_size, 256, 19).to(device)
        # white.load_state_dict(torch.load("./models/model.json"))
        # black.load_state_dict(torch.load("./models/model.json"))
        with torch.no_grad():
            self_play = True
            total_n_black_wins = None
            total_n_draws = None
            p1 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
            p2 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
            p3 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
            p4 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
            p5 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
            # p6 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//7, white, black, device, self_play, total_n_black_wins, total_n_draws))
            # p7 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//7, white, white, device, self_play, total_n_black_wins, total_n_draws))
            # p8 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//7, white, white, device, self_play, total_n_black_wins, total_n_draws))
            # p9 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//5, white, white, device, self_play, total_n_black_wins, total_n_draws))
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            # p6.start()
            # p7.start()
            # p8.start()
            # p9.start()
            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            # p6.join()
            # p7.join()
            # p8.join()
            # p9.join()
        if archive == []:
            print("archive empty")
        for i in range(10):
            # I AM TRAINING THE WHITE NETWORK
            train(white, archive, device=device)
        del archive  # empty archive to relieve ram
        archive = manager.list()
        # I AM SAVING THE WHITE NETWORK
        torch.save(white.state_dict(), "./models/model_draw_is_0.json") #save the model
        print("model saved")
        #it is my understanding that declining temperature and dirichlet noise is only used in self play
        # #evaluate
        # total_n_black_wins = manager.list()
        # total_n_black_wins.append(0)
        # total_n_draws = manager.list()
        # total_n_draws.append(0)
        # evaluate_n_games = 80 # prev 80
        # white = NN( input_size, 256, 19).to(device)
        # black = NN( input_size, 256, 19).to(device)
        # white.load_state_dict(torch.load("./models/model.json"))
        # # black.load_state_dict(torch.load("./models/model.json"))
        # with torch.no_grad():
        #     self_play = False
        #     p1 = mp.Process(target=test_multiprocess, args=(archive, evaluate_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
        #     p2 = mp.Process(target=test_multiprocess, args=(archive, evaluate_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
        #     p3 = mp.Process(target=test_multiprocess, args=(archive, evaluate_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
        #     p4 = mp.Process(target=test_multiprocess, args=(archive, evaluate_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
        #     p5 = mp.Process(target=test_multiprocess, args=(archive, evaluate_n_games//5, white, black, device, self_play, total_n_black_wins, total_n_draws))
        #     # p6 = mp.Process(target=test_multiprocess, args=(archive, self_play_n_games//6, white, black, device))
        #     p1.start()
        #     p2.start()
        #     p3.start()
        #     p4.start()
        #     p5.start()
        #     # p6.start()
        #     p1.join()
        #     p2.join()
        #     p3.join()
        #     p4.join()
        #     p5.join()
        #     # p6.join()
        # # for game in range(evaluate_n_games):
        # #     _, winner_is_white = test(white, black, self_play = False, device=device) #if draw, winner_is_white == 0.5
        # #     n_white_wins += winner_is_white
        # #     if winner_is_white ==1.0 :
        # #         how_many_black_won += 1
        # #     elif winner_is_white == 0.5:
        # #         how_many_draws += 1
        # print("ratio: ", (total_n_black_wins[0]+ total_n_draws[0]/2)/evaluate_n_games)
        # print("number of black win: ", total_n_black_wins[0])
        # print("number of draws: ", total_n_draws[0])
        # if (total_n_black_wins[0] + total_n_draws[0]/2)/evaluate_n_games > 0.55: #if black won at least 55 % of the games then black becomes the new current best network
        #     print("black is the new best")
        #     torch.save(black.state_dict(), "./models/model.json") #save the model
        #     print("model saved")
        #     white = copy.deepcopy(black)
        # else:
        #     print("white is still the best")
            # black = copy.deepcopy(white) # reassign black as the current best: white
        print("iteration: ", iteration)

    # print(archive)
    print("finished")
    print("elapsed time: ", time.time() - t0)
