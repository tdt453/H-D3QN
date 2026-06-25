import numpy as np

class TelecomIoTEnv:
    def __init__(self, initial_states, use_rf=False, use_if=False, use_reward_shaping=False, rf_model=None, anomaly_detector=None):
        self.initial_states = initial_states
        self.use_rf, self.use_if, self.use_rs = use_rf, use_if, use_reward_shaping
        self.rf_model, self.anomaly_detector = rf_model, anomaly_detector
        self.max_steps = 100
        self.reset()

    def reset(self):
        self.current_state = np.copy(self.initial_states[np.random.randint(0, len(self.initial_states))])
        self.current_step = 0
        return self.current_state

    def step(self, action):
        # 1. Physics & Action Dynamics (Action ảnh hưởng thực sự đến state)
        next_state = np.copy(self.current_state)
        # Random drift
        next_state[0] += np.random.normal(0, 0.01) # RSSI
        next_state[1] += np.random.normal(0, 0.01) # Load
        
        # Action effects
        if action == 1: next_state[0] += 0.08; next_state[1] += 0.05 # Aggressive
        elif action == 2: next_state[0] -= 0.05                     # Conservative
        elif action == 3: next_state[1] += 0.10                     # Throughput
        elif action == 4: next_state[0] -= 0.08; next_state[1] -= 0.05 # Safe Mode
        next_state = np.clip(next_state, 0.0, 1.0)
        
        # 2. Risk Evaluation (Deterministic, không còn random)
        risk_score = (0.45 * next_state[0] + 0.35 * next_state[1] + 0.20 * 0.5)
        cat_flag = 1 if risk_score > 0.72 else 0
        dec_flag = 1 if (risk_score > 0.60 and risk_score <= 0.72) else 0
        
        # 3. Reward Calculation
        reward = (next_state[0] * 10.0 + next_state[1] * 5.0) - (0.2 if action == 4 else 0.5)
        if self.use_rs:
            if self.use_rf and self.rf_model and self.rf_model.predict(next_state.reshape(1, -1))[0] == 1: reward += 2.0
            if self.use_if and self.anomaly_detector and self.anomaly_detector.decision_function(next_state.reshape(1, -1))[0] >= 0: reward += 2.0
        
        done = False
        if cat_flag: reward -= 150.0; done = True # Nặng nhưng cho phép học lại
        elif dec_flag: reward -= 25.0
            
        self.current_step += 1
        if self.current_step >= self.max_steps: done = True
        self.current_state = next_state
        return next_state, reward, done, {'catastrophic_hits': cat_flag, 'deceptive_hits': dec_flag}