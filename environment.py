import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3


class TicTacToe:
    def __init__(self):
        # 定義遊戲狀態
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))  # 3x3的空白棋盤
        self.max_steps = len([' ' for _ in range(3) for _ in range(3)])
        self.current_player = 'X'  # 設置起始玩家
        self.isEnd = False

    # only when game ends
    def giveReward(self, result):
        # backpropagate reward
        if result == 1:  # Player 1 wins
            reward_p1 = 1
            reward_p2 = -1
        elif result == -1:  # Player 2 wins
            reward_p1 = -1
            reward_p2 = 1
        else:  # Tie game
            reward_p1 = 0.1
            reward_p2 = 0.5

        return reward_p1, reward_p2

    def step(self, action, playerSymbol: int):
        self.updateState(action, playerSymbol)
        win = self.winner()
        if win is not None:
            self.isEnd = True

    # board reset

    def reset(self):
        """
            self.board:
                [[0. 0. 0.]
                [0. 0. 0.]
                [0. 0. 0.]]
        """
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.isEnd = False

    def render(self):
        # 打印當前棋盤狀態
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)

    def availablePositions(self):
        """
        return:
            positions[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        """
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position, playerSymbol: int):
        self.board[position] = playerSymbol

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1]
                        for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        print(positions)
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    def GamechooseAction(self, positions, col, row):
        print(positions)
        while True:
            row = row
            col = col
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass
