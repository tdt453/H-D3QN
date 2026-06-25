# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.optim as optim, numpy as np, random
from models import DuelingDQN, PrioritizedReplayMemory, set_seed
from environment import TelecomIoTEnv
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# --- 1. Classes phụ trợ ---
class SimpleDQN(nn.Module):
    def __init__(self, dim, act):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, act))
    def forward(self, x): return self.fc(x)

class ReplayMemory:
    def __init__(self, cap): self.memory = []; self.cap = cap; self.pos = 0
    def push(self, t):
        if len(self.memory) < self.cap: self.memory.append(t)
        else: self.memory[self.pos] = t
        self.pos = (self.pos + 1) % self.cap
    def sample(self, bs): 
        b = random.sample(self.memory, bs)
        return b, None, np.ones(bs)
    def update_priorities(self, idx, err): pass
    def __len__(self): return len(self.memory)

# --- 2. HÀM HELPER (Bị thiếu dẫn đến ImportError) ---
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_isolation_forest(X_scaled, contamination=0.05):
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso_forest.fit(X_scaled)
    return iso_forest 

# --- 3. Training Loop ---
def train_dueling_double_dqn(states, cfg, rf, if_model, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim, act = states.shape[1], 5
    
    policy_net = (DuelingDQN(dim, act) if cfg.get("duel") else SimpleDQN(dim, act)).to(device)
    target_net = (DuelingDQN(dim, act) if cfg.get("duel") else SimpleDQN(dim, act)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    memory = PrioritizedReplayMemory(10000) if cfg.get("per") else ReplayMemory(10000)
    
    env = TelecomIoTEnv(states, cfg.get("guard"), cfg.get("guard"), cfg.get("rs"), rf, if_model)
    
    total_rewards, cat_hits, dec_hits, total_steps = [], 0, 0, 0
    for episode in range(100):
        state = env.reset()
        ep_rew = 0
        epsilon = max(0.05, 1.0 - episode / 80)
        while True:
            total_steps += 1
            if random.random() < epsilon: action = random.randrange(act)
            else: action = policy_net(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()
            
            # Guard logic
            if cfg.get("guard"):
                rf_ok = rf.predict(state.reshape(1, -1))[0] == 1 if rf else True
                if_ok = if_model.decision_function(state.reshape(1, -1))[0] >= 0 if if_model else True
                if not (rf_ok and if_ok): action = 4 
            
            ns, r, done, info = env.step(action)
            cat_hits += info['catastrophic_hits']; dec_hits += info['deceptive_hits']
            memory.push((state, action, r, ns, float(done)))
            state = ns; ep_rew += r
            
            if len(memory) > 64:
                samples, indices, weights = memory.sample(64)
                batch = list(zip(*samples))
                b_s, b_a, b_r, b_ns, b_d, b_w = [torch.tensor(np.array(x), dtype=torch.float32, device=device) if i!=1 else torch.tensor(np.array(x), device=device).unsqueeze(1) for i, x in enumerate(batch)]
                
                q = policy_net(b_s).gather(1, b_a.long())
                with torch.no_grad():
                    if cfg.get("double"):
                        next_a = policy_net(b_ns).argmax(1, keepdim=True)
                        next_q = target_net(b_ns).gather(1, next_a).squeeze()
                    else: next_q = target_net(b_ns).max(1)[0]
                
                target = b_r + (0.99 * next_q * (1 - b_d))
                loss = (b_w * nn.MSELoss(reduction='none')(q.squeeze(), target)).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if cfg.get("per"): memory.update_priorities(indices, torch.abs(q.squeeze() - target).detach().cpu().numpy())
            if done: break
        total_rewards.append(ep_rew)
        if episode % 10 == 0: target_net.load_state_dict(policy_net.state_dict())
        
    return {"Reward": np.mean(total_rewards[-20:]), "Reliability": 100*(1-cat_hits/max(1, total_steps)), 
            "Failure Rate": 100*(cat_hits/max(1, total_steps)), "Trap Rate": 100*(dec_hits/max(1, total_steps))}

def train_variant(cfg, states, rf, if_model, seed): return train_dueling_double_dqn(states, cfg, rf, if_model, seed=seed)