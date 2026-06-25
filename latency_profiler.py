
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# KẾT NỐI VỚI FILE CONFIG CỦA BẠN
try:
    from config import PROCESSED_DATA_FILENAME
except ImportError:
    print("[WARNING] Could not import from config.py. Using default 'Unknown_Dataset'.")
    PROCESSED_DATA_FILENAME = "unknown"

# Cấu hình chuẩn IEEE
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

torch.set_num_threads(1) 


class DummyD3QN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.val = nn.Linear(128, 1)
        self.adv = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.val(x) + self.adv(x) - self.adv(x).mean(dim=1, keepdim=True)

class DummyAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.dec = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, dim))
    def forward(self, x): return self.dec(self.enc(x))

def get_dataset_name(filename):
    """Hàm tự động nhận diện tên Dataset dựa trên file config.py"""
    fname_lower = filename.lower()
    if "sigfox" in fname_lower:
        return "Sigfox"
    elif "lorawan" in fname_lower:
        return "LoRaWAN"
    elif "indoor" in fname_lower:
        return "Indoor"
    else:
        return "Custom_Dataset"

def run_latency_profiler():
    # Tự động lấy tên dataset từ config.py
    DATASET_NAME = get_dataset_name(PROCESSED_DATA_FILENAME)
    
    print("="*80)
    print(f" PROFILING ALL 5 HYBRID COMBINATIONS FOR: {DATASET_NAME}")
    print("="*80)
    
    n_features = 10
    n_actions = 5
    n_trials = 2000
    n_dummy_samples = 5000
    
    scaler = StandardScaler()
    dummy_X = np.random.rand(n_dummy_samples, n_features)
    dummy_y = np.random.randint(0, 2, n_dummy_samples)
    scaler.fit(dummy_X)
    
    # Khởi tạo 5 cấu hình đầy đủ
    configs = {
        "IF+RF": (IsolationForest(random_state=42, n_jobs=1), RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1)),
        "IF+XGB": (IsolationForest(random_state=42, n_jobs=1), XGBClassifier(n_estimators=100, max_depth=10, n_jobs=1)),
        "OCSVM+RF": (OneClassSVM(kernel='rbf'), RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1)),
        "LOF+RF": (LocalOutlierFactor(novelty=True, n_neighbors=20, n_jobs=1), RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1)),
        "AE+RF": (DummyAE(n_features), RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1))
    }
    
    d3qn_agent = DummyD3QN(n_features, n_actions)
    d3qn_agent.eval()
    
    latency_results = {}
    total_metrics = []
    
    for comb_name, (guard, verifier) in configs.items():
        print(f"[*] Benchmarking: {comb_name}...")
        
        # Fit models
        if comb_name == "AE+RF":
            guard.eval() 
        else:
            guard.fit(dummy_X)
        verifier.fit(dummy_X, dummy_y)
        
        # WARM-UP (Tránh Lazy Init)
        for _ in range(50):
            tmp = scaler.transform(np.random.rand(1, n_features))
            if comb_name == "AE+RF":
                with torch.no_grad():
                    _ = guard(torch.from_numpy(tmp).float())
            else:
                guard.predict(tmp)
            verifier.predict(tmp)
            with torch.no_grad():
                _ = d3qn_agent(torch.from_numpy(tmp).float())
        
        times_pre = []; times_gua = []; times_ver = []; times_d3qn = []; times_total = []
        
        # BENCHMARK LOOP
        for _ in range(n_trials):
            raw_data = np.random.rand(1, n_features)
            t_start_e2e = time.perf_counter()
            
            # 1. Preprocessing
            t0 = time.perf_counter()
            state = scaler.transform(raw_data)
            t1 = time.perf_counter()
            
            # 2. Guard
            if comb_name == "AE+RF":
                with torch.no_grad():
                    _ = guard(torch.from_numpy(state).float())
            else:
                guard.predict(state)
            t2 = time.perf_counter()
            
            # 3. Verifier
            verifier.predict(state)
            t3 = time.perf_counter()
            
            # 4. D3QN
            with torch.no_grad():
                _ = d3qn_agent(torch.from_numpy(state).float())
            t4 = time.perf_counter()
            
            t_end_e2e = time.perf_counter()
            
            times_pre.append((t1 - t0) * 1000)
            times_gua.append((t2 - t1) * 1000)
            times_ver.append((t3 - t2) * 1000)
            times_d3qn.append((t4 - t3) * 1000)
            times_total.append((t_end_e2e - t_start_e2e) * 1000)
            
        latency_results[comb_name] = {
            "Preprocessing": np.mean(times_pre),
            "Guard": np.mean(times_gua),
            "Verifier": np.mean(times_ver),
            "D3QN": np.mean(times_d3qn)
        }
        
        total_metrics.append({
            "Combination": comb_name,
            "Mean (ms)": np.mean(times_total),
            "Std (ms)": np.std(times_total),
            "P95 (ms)": np.percentile(times_total, 95),
            "Throughput (inf/s)": 1000 / np.mean(times_total)
        })
        
    df_metrics = pd.DataFrame(total_metrics)
    print("\n[LATENCY & THROUGHPUT METRICS]")
    print(df_metrics.to_string(index=False))
    
   
    tex_filename = f"table_latency_{DATASET_NAME}.tex"
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(df_metrics.to_latex(index=False, float_format="%.4f", 
                caption=f"Deterministic Edge Inference Metrics ({DATASET_NAME} Dataset)", 
                label=f"tab:latency_{DATASET_NAME.lower()}"))
        
    plot_stacked_latency(latency_results, DATASET_NAME)

def plot_stacked_latency(results, dataset_name):
    labels = list(results.keys())
    pre = [res["Preprocessing"] for res in results.values()]
    guard = [res["Guard"] for res in results.values()]
    verifier = [res["Verifier"] for res in results.values()]
    d3qn = [res["D3QN"] for res in results.values()]

    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(labels, pre, width, label='Feature Preprocessing', color='#D6EAF8', edgecolor='black')
    ax.bar(labels, guard, width, bottom=pre, label='Spatial Guard', color='#85C1E9', edgecolor='black')
    ax.bar(labels, verifier, width, bottom=np.array(pre)+np.array(guard), label='Signal Verifier', color='#3498DB', edgecolor='black')
    ax.bar(labels, d3qn, width, bottom=np.array(pre)+np.array(guard)+np.array(verifier), label='D3QN Agent', color='#21618C', edgecolor='black')

    ax.set_ylabel('Inference Latency (ms)', fontweight='bold')
    ax.set_title(f'End-to-End Processing Latency Breakdown ({dataset_name})', fontweight='bold')
    
    for i in range(len(labels)):
        total = pre[i] + guard[i] + verifier[i] + d3qn[i]
        ax.text(i, total + 0.05, f"{total:.2f} ms", ha='center', va='bottom', fontweight='bold')

    max_val = max([pre[i] + guard[i] + verifier[i] + d3qn[i] for i in range(len(labels))])
    # Tăng thêm một chút khoảng trống phía trên để số không bị đè vào viền
    ax.set_ylim(0, max_val + max_val * 0.15)
    
   
    
    handles, legends = ax.get_legend_handles_labels()
    ax.legend(handles, legends, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    
    pdf_filename = f'Fig_Latency_Breakdown_{dataset_name}.pdf'
    plt.savefig(pdf_filename)
    print(f"\n[SUCCESS] Exported table_latency_{dataset_name}.tex and {pdf_filename}!")

if __name__ == "__main__":
    run_latency_profiler()