import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
import time
import copy

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, cfg , playerSymbol:int, buffer_capacity=10000):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)  # 新增的層
        self.cfg = cfg
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.frame_idx = 0  # attenuation   
        self.playerSymbol = playerSymbol
        
        self.memory = ReplayBuffer(buffer_capacity)        
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.lr)  # optimizer
        
        # 由於獎勵是最後才會給 先將每一步的獎勵先記下來
        self.episode_memory = [] # Temporary storage for states and actions
        
    def forward(self, state):
        # 假設狀態是3x3的棋盤，可以展平成長度為9的向量
        state = state.view(-1, 9)
        x = F.relu(self.fc1(state))
        x = self.fc2(x)  # 新增的層的前向傳遞
        return x
    
    def prepare_state(self, board):
        return torch.tensor(board, dtype=torch.float).view(1, -1)  # 轉換為1x9的張量

    def choose_action(self, positions, state,model='train'):
        """_summary_

        Args:
            positions (_type_): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            state (_type_): [[0. 0. 0.] [0. 0. 0.] [0. 0. 0.]]

        Returns:
            _type_: _description_
        """
        
        self.frame_idx +=1
        if self.frame_idx % 1000==0:
            print("目前探索率:",self.epsilon(self.frame_idx))
        
        if model == 'formal':
            state_tensor = self.prepare_state(state)  # 將當前遊戲狀態轉換為神經網絡可以接受的形式
            q_values = self.forward(state_tensor)  # 計算每個可能動作的Q值
            valid_q_values = {pos: q_values[0][pos[0]*3 + pos[1]] for pos in positions}  # 獲取剩餘位置的Q值
            action = max(valid_q_values, key=valid_q_values.get)  # 選擇具有最高Q值的動作
            
            return action
        
        if torch.rand(1).item() < self.epsilon(self.frame_idx):  # 探索
            idx = (torch.randint(0, len(positions), (1,)).item()) # 隨機選擇一個有效的動作
            action = positions[idx]
        else:
            state_tensor = self.prepare_state(state)  # 將當前遊戲狀態轉換為神經網絡可以接受的形式
            q_values = self.forward(state_tensor)  # 計算每個可能動作的Q值
            valid_q_values = {pos: q_values[0][pos[0]*3 + pos[1]] for pos in positions}  # 獲取剩餘位置的Q值
            action = max(valid_q_values, key=valid_q_values.get)  # 選擇具有最高Q值的動作

        self.episode_memory.append((copy.deepcopy(state), action))
        return action
    
    def end_episode(self, reward):
        # At the end of an episode, store the entire sequence in the replay buffer with the final reward
        for state, action in self.episode_memory:
            next_state = None # There's no next state, since the game is over
            self.memory.push(state, action, reward, next_state)
        
        self.episode_memory = [] # Clear the temporary storage
        self.update(self.cfg.gamma) # Update the DQN based on the memory
        
    def update(self, gamma=0.99):
        if len(self.memory) < self.cfg.batch_size:
            return  # Not enough samples in the replay buffer to update

        state, action, reward, next_state = self.memory.sample(self.cfg.batch_size)
        state = torch.stack([torch.tensor(s, dtype=torch.float) for s in state])
        next_states_tensors = [torch.tensor(s, dtype=torch.float) for s in next_state if s is not None]
        next_state_tensor = torch.stack(next_states_tensors) if next_states_tensors else None
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        # Compute Q values
        q_values = self(state)
        if next_state_tensor is not None:
            next_q_values_raw = self(next_state_tensor)
            next_q_values = next_q_values_raw.max(1)[0] # Use the max along dimension 1
        else:
            next_q_values = torch.tensor([0.0] * len(reward))  # assuming reward has the same length as the number of terminal states

        action_indices = action[:, 0] * 3 + action[:, 1]
        q_value = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.detach()
        # Compute target Q value
        target_q_value = reward + gamma * next_q_value

        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)
        if self.playerSymbol == 1 and self.frame_idx % 1000 == 0:
            print(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

