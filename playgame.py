from pygame.locals import QUIT, MOUSEBUTTONDOWN
import pygame
from DQN_models import DQN
from RL_train import Config
import os
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from environment import TicTacToe, HumanPlayer


curr_path = os.path.dirname(__file__)
curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
cfg = Config(curr_path, curr_time)

# 如果想加載智能體
agent1 = DQN(cfg.state_space_dim, cfg.action_space_dim, cfg, playerSymbol=1)
agent1.load_state_dict(torch.load('agent1.pth'))
agent1.eval()  # 設置為評估模式，不使用Dropout等訓練特定功能


# 初始化 pygame
pygame.init()

# 定義常數
WIDTH, HEIGHT = 300, 300
GRID_SIZE = WIDTH // 3

# 初始化視窗
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')

# 人機對戰
print("開始人機對戰")
print('*'*120)
user = HumanPlayer(name='lewis')
env = TicTacToe()


def draw_board(board):
    """繪製棋盤和棋子"""
    screen.fill((255, 255, 255))
    for row in range(3):
        for col in range(3):
            if board[row][col] == 1:  # AI
                pygame.draw.circle(screen, (242, 85, 96), (col * GRID_SIZE +
                                   GRID_SIZE // 2, row * GRID_SIZE + GRID_SIZE // 2), 40, 2)
            elif board[row][col] == -1:  # Human
                pygame.draw.line(screen, (85, 85, 242), (col * GRID_SIZE, row *
                                 GRID_SIZE), ((col + 1) * GRID_SIZE, (row + 1) * GRID_SIZE), 2)
                pygame.draw.line(screen, (85, 85, 242), ((
                    col + 1) * GRID_SIZE, row * GRID_SIZE), (col * GRID_SIZE, (row + 1) * GRID_SIZE), 2)

    for x in range(1, 3):
        pygame.draw.line(screen, (0, 0, 0), (GRID_SIZE * x, 0),
                         (GRID_SIZE * x, HEIGHT), 2)
        pygame.draw.line(screen, (0, 0, 0), (0, GRID_SIZE * x),
                         (WIDTH, GRID_SIZE * x), 2)

    pygame.display.flip()


# 在遊戲開始前，讓代理進行第一次動作
positions = env.availablePositions()
action = agent1.choose_action(positions, env.board, model='formal')
env.step(action, agent1.playerSymbol)
draw_board(env.board)  # 用當前的棋盤狀態重新繪製畫面

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
            
        elif event.type == MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            # 計算九宮格的 (row, col)
            col = x // GRID_SIZE
            row = y // GRID_SIZE

            positions = env.availablePositions()
            action = user.GamechooseAction(positions, col, row)
            env.step(action, -1)  # 假設人類玩家的符號是-1

            draw_board(env.board)  # 用當前的棋盤狀態重新繪製畫面

            if not positions or env.isEnd:
                pygame.quit()
                exit()

            # 人類玩家進行動作後，代理再次輪到它進行動作
            positions = env.availablePositions()
            action = agent1.choose_action(positions, env.board, model='formal')
            env.step(action, agent1.playerSymbol)

            draw_board(env.board)  # 用當前的棋盤狀態重新繪製畫面

            if env.isEnd:
                pygame.quit()
                exit()
