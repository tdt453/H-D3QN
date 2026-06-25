import torch
import torch.nn as nn
import numpy as np
import random
import os


def set_seed(seed=42):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Đảm bảo các phép toán convolution/linear deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- 1. KIẾN TRÚC DUELING DQN ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature(x)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# --- 2. BỘ NHỚ PER
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        # FIX Điểm Deep: Dùng running_max thay vì hard limit cứng 1.0
        self.max_priority = 1.0 
        self.memory = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(self.max_priority)
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0: 
            return None
        
        actual_batch_size = min(batch_size, len(self.memory))
        
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.memory), actual_batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance Sampling Weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        
        # FIX 2.3: Chuẩn hóa an toàn, tránh chia cho 0 nếu weight max là 0
        weight_max = weights.max()
        if weight_max > 0:
            weights /= weight_max
            
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            # FIX 2.1 & 2.2: Đảm bảo priority LUÔN DƯƠNG bằng hàm abs()
            prio_val = abs(float(prio)) + 1e-6
            self.priorities[idx] = prio_val
            
            
            self.max_priority = max(self.max_priority, prio_val)

    def __len__(self):
        return len(self.memory)
    # Trong file models.py
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.fc(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
    def push(self, transition):
        if len(self.memory) < self.capacity: self.memory.append(transition)
        else: self.memory[random.randint(0, self.capacity-1)] = transition
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch, None, np.ones(batch_size) # Weights = 1.0 (Uniform)
    def update_priorities(self, indices, td_errors): pass
    def __len__(self): return len(self.memory)