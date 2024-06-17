import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from environment import TicTacToe,HumanPlayer
import time
from DQN_models import DQN


class Config:
    '''
    hyperparameters
    '''

    def __init__(self, curr_path, curr_time):

        self.curr_path = curr_path
        self.curr_time = curr_time

        ################################## env hyperparameters ###################################
        self.algo_name = 'DQN'  # algorithmic name
        self.env_name = 'TicTacToe'  # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU

        self.seed = 11  # random seed

        self.train_eps = 30000  # training episodes #訓練次數
        self.state_space_dim = 9
        self.action_space_dim = 9
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.9  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        # attenuation rate of epsilon in e-greedy policy # decay機制 用來讓智能體慢慢減少探索的機率
        self.epsilon_decay = 30000
        self.lr = 0.001  # learning rate # 這邊類似於Q-learning裡面的ALPHA
        # self.memory_capacity = 1000  # capacity of experience replay
        self.batch_size = 32  # size of mini-batch SGD
        # self.target_update = 4  # update frequency of target network
        # self.hidden_dim = 128  # dimension of hidden layer
        ################################################################################

        ################################# save path ##############################
        # self.result_path = self.curr_path + "/outputs/" + self.env_name + \
        #     '/' + self.curr_time + '/results/'
        # self.model_path = self.curr_path + "/outputs/" + self.env_name + \
        #     '/' + self.curr_time + '/models/'
        # self.save = True  # whether to save the image
        ################################################################################

def count():
    curr_path = os.path.dirname(__file__)
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = Config(curr_path, curr_time)


    agent1 = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg, playerSymbol=1)
    agent2 = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg, playerSymbol=-1)

    winners =[]
    # 訓練循環
    for episode in range(cfg.train_eps):
        if episode % 1000 == 0:
            print(episode)
            
        env = TicTacToe()
        env.reset()

        while not env.isEnd:
            # Player 1
            positions = env.availablePositions()
            p1_action = agent1.choose_action(positions, env.board)  # 使用ε-greedy策略
            env.step(p1_action, agent1.playerSymbol)
            
            
            positions = env.availablePositions()
            if not positions or env.isEnd:
                reward_p1, reward_p2 = env.giveReward(env.winner())
                winners.append(env.winner())
                agent1.end_episode(reward_p1)
                agent2.end_episode(reward_p2)
                break

            p2_action = agent2.choose_action(positions, env.board)  # 使用ε-greedy策略
            env.step(p2_action, agent2.playerSymbol)
            

            if env.isEnd:
                reward_p1, reward_p2 = env.giveReward(env.winner())
                winners.append(env.winner())
                agent1.end_episode(reward_p1)
                agent2.end_episode(reward_p2)
                break
        
    print(winners[-300:])
    # 保存智能體
    torch.save(agent1.state_dict(), 'agent1.pth')


if __name__ == '__main__':
    count()