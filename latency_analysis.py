# -*- coding: utf-8 -*-
"""Research-Grade Latency, Memory, and Statistical Inference Profiling"""

import os
import sys

# Ép Terminal sử dụng UTF-8 để tránh lỗi hiển thị tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
import torch
import joblib
import psutil
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Import cấu trúc từ dự án
from models import DuelingDQN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import MinMaxScaler
from config import PROCESSED_DATA_FILENAME
from preprocessing import get_dynamic_features

torch.set_grad_enabled(False)
torch.set_num_threads(1)

def get_current_memory_mb():
    """Đo lượng RAM Process hiện tại đang sử dụng (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def count_parameters(model):
    """Đếm tham số huấn luyện mạng Neural"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_real_pipeline(input_dim, num_actions, device):
    """Load Pre-trained Models với cấu trúc Pruned Ensemble (20 trees)"""
    policy_net = DuelingDQN(input_dim, num_actions).to(device)
    rf_model = RandomForestClassifier(n_estimators=20, max_depth=7, random_state=42)
    if_model = IsolationForest(n_estimators=20, contamination=0.05, random_state=42)
    scaler = MinMaxScaler()

    try:
        policy_net.load_state_dict(torch.load("policy_net.pth", map_location=device))
        rf_model = joblib.load("rf_model.pkl")
        if_model = joblib.load("if_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("[OK] Đã load thành công Real Weights & Real Scaler.")
        
        # Kiểm tra tính đồng bộ của Pruned Ensemble
        if getattr(rf_model, 'n_estimators', 0) > 20:
            print("[WARNING] File rf_model.pkl chứa > 20 cây. Suy luận có thể bị chậm!")
    except Exception as e:
        print(f"[WARNING] Không tìm thấy đủ file weights ({e}). Chạy Dummy Initiation (Pruned 20-trees)...")
        dummy_X = np.random.rand(100, input_dim)
        rf_model.fit(dummy_X, np.random.randint(0, 2, 100))
        if_model.fit(dummy_X)
        scaler.fit(dummy_X)
    
    policy_net.eval()
    return policy_net, rf_model, if_model, scaler

def load_real_states(num_samples=10000):
    """Load Validation States từ tập dữ liệu thực tế"""
    try:
        data_path = f"data/{PROCESSED_DATA_FILENAME}"
        df = pd.read_csv(data_path).dropna()
        features = [f for f in get_dynamic_features() if f in df.columns]
        real_states = df[features].values
        
        indices = np.random.choice(len(real_states), num_samples, replace=True)
        print(f"[OK] Đã load {num_samples} trạng thái từ Real Dataset.")
        return real_states[indices]
    except Exception as e:
        print(f"[WARNING] Lỗi load data ({e}). Chuyển sang dữ liệu ngẫu nhiên.")
        return np.random.rand(num_samples, 5)

def plot_latency_stack(medians, p95_e2e_pure, p95_e2e_sota, save_path="latency_breakdown.pdf"):
    """Vẽ biểu đồ Stacked Bar Chart chuẩn IEEE Grayscale (Asymmetric Error Bars)"""
    labels = ['Pure D3QN', 'Hybrid SOTA']
    
    fe_data = np.array([medians['FE'], medians['FE']])
    if_data = np.array([0, medians['IF']])
    rf_data = np.array([0, medians['RF']])
    dqn_data = np.array([medians['DQN'], medians['DQN']])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(labels, fe_data, label='Feature Extraction (Scaler)', color='#E0E0E0', edgecolor='black', hatch='\\\\')
    ax.bar(labels, if_data, bottom=fe_data, label='Isolation Forest', color='#BDBDBD', edgecolor='black', hatch='..')
    ax.bar(labels, rf_data, bottom=fe_data+if_data, label='Random Forest', color='#757575', edgecolor='black', hatch='xx')
    ax.bar(labels, dqn_data, bottom=fe_data+if_data+rf_data, label='D3QN Agent', color='#424242', edgecolor='black', hatch='//')
    
    e2e_medians = [medians['FE'] + medians['DQN'], sum(medians.values())]
    
    # [FIX] ASYMMETRIC ERROR BARS: Ngăn chặn thanh sai số bị kéo xuống mức âm
    lower_errors = [0, 0]
    upper_errors = [p95_e2e_pure - e2e_medians[0], p95_e2e_sota - e2e_medians[1]]
    asymmetric_error = [lower_errors, upper_errors]
    
    ax.errorbar(
        labels, 
        e2e_medians, 
        yerr=asymmetric_error, 
        fmt='none', 
        ecolor='black', 
        capsize=5, 
        capthick=2, 
        label='P95 Worst-case Latency'
    )
    
    ax.set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Latency Breakdown (Pruned Ensemble)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_latency_analysis():
    print("\n" + "="*70)
    print("      SYSTEM PROFILING & END-TO-END LATENCY BENCHMARK      ")
    print("="*70)
    
    device = torch.device("cpu") 
    input_dim = 5
    num_actions = 5
    
    mem_baseline = get_current_memory_mb()
    policy_net, rf_model, if_model, scaler = load_real_pipeline(input_dim, num_actions, device)
    mem_after_load = get_current_memory_mb()
    model_memory = mem_after_load - mem_baseline
    
    print("\n--- 1. COMPLEXITY & MEMORY FOOTPRINT ---")
    print(f"- D3QN Parameters : {count_parameters(policy_net):,} trainable params")
    print(f"- Ensemble Setup  : RF({getattr(rf_model, 'n_estimators', 20)} trees), IF({getattr(if_model, 'n_estimators', 20)} trees)")
    print(f"- Peak RAM Usage  : ~{abs(model_memory):.2f} MB")
    print("-" * 40)

    print("\n[INFO] Khởi chạy vòng lặp Cache Warm-up...")
    dummy_state = np.random.rand(1, input_dim)
    for _ in range(200):
        s_scaled = scaler.transform(dummy_state)
        _ = if_model.decision_function(s_scaled)
        _ = rf_model.predict(s_scaled)
        _ = policy_net(torch.as_tensor(s_scaled, dtype=torch.float32, device=device))

    iterations = 10000
    validation_states = load_real_states(iterations)
    
    times_fe, times_if, times_rf, times_dqn = (np.zeros(iterations) for _ in range(4))
    e2e_pure, e2e_sota = np.zeros(iterations), np.zeros(iterations)
    
    print(f"[INFO] Bắt đầu đo lường End-to-End trên {iterations} samples...")
    for i in range(iterations):
        raw_state = validation_states[i:i+1] 
        
        # === KỊCH BẢN 1: PURE D3QN ===
        t_pure_start = time.perf_counter()
        state_scaled = scaler.transform(raw_state)
        tensor_state = torch.as_tensor(state_scaled, dtype=torch.float32, device=device)
        _ = policy_net(tensor_state).argmax().item()
        e2e_pure[i] = (time.perf_counter() - t_pure_start) * 1000
        
        
        t_sota_start = time.perf_counter()
        
        t0 = time.perf_counter()
        state_scaled = scaler.transform(raw_state)
        tensor_state = torch.as_tensor(state_scaled, dtype=torch.float32, device=device)
        times_fe[i] = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        _ = if_model.decision_function(state_scaled)
        times_if[i] = (time.perf_counter() - t1) * 1000
        
        t2 = time.perf_counter()
        _ = rf_model.predict(state_scaled)
        times_rf[i] = (time.perf_counter() - t2) * 1000
        
        t3 = time.perf_counter()
        _ = policy_net(tensor_state).argmax().item()
        times_dqn[i] = (time.perf_counter() - t3) * 1000
        
        e2e_sota[i] = (time.perf_counter() - t_sota_start) * 1000

   
    stats = {}
    for name, data in [("PURE D3QN", e2e_pure), ("SOTA HYBRID", e2e_sota)]:
        stats[name] = {
            "Median": np.median(data),
            "P95": np.percentile(data, 95),
            "Std": np.std(data),
            "IQR": np.percentile(data, 75) - np.percentile(data, 25)
        }
        
    print("\n--- 2. INFERENCE STATISTICAL CONFIDENCE ---")
    print(f"[*] PURE D3QN   : Median {stats['PURE D3QN']['Median']:.4f} ms | Std: ±{stats['PURE D3QN']['Std']:.4f} | IQR: {stats['PURE D3QN']['IQR']:.4f}")
    print(f"[*] SOTA HYBRID : Median {stats['SOTA HYBRID']['Median']:.4f} ms | Std: ±{stats['SOTA HYBRID']['Std']:.4f} | IQR: {stats['SOTA HYBRID']['IQR']:.4f}")
    print(f"    (P95 Worst-case SOTA: {stats['SOTA HYBRID']['P95']:.4f} ms)")
    print("="*70)

    # Xuất biểu đồ
    medians = {
        'FE': np.median(times_fe), 'IF': np.median(times_if),
        'RF': np.median(times_rf), 'DQN': np.median(times_dqn)
    }
    plot_latency_stack(medians, stats['PURE D3QN']['P95'], stats['SOTA HYBRID']['P95'])
    print("\n[INFO] Đã xuất thành công biểu đồ: latency_breakdown.pdf")

if __name__ == "__main__":
    run_latency_analysis()