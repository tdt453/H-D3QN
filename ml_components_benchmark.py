# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Machine Learning - Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Machine Learning - Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Import từ project của bạn
from config import PROCESSED_DATA_FILENAME
from preprocessing import prepare_classification_data, get_dynamic_features

def load_data():
    """Tải dữ liệu đã qua tiền xử lý"""
    data_path = f"data/{PROCESSED_DATA_FILENAME}"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy {data_path}. Hãy chạy file main.py để sinh data trước.")
    data = pd.read_csv(data_path).dropna()
    return data

def inject_synthetic_anomalies(X, contamination=0.05, seed=42):
    """
    Tạo Ground Truth cho bài toán Anomaly Detection bằng cách tiêm nhiễu nhân tạo.
    Đây là kỹ thuật chuẩn trong các bài báo khoa học khi tập dữ liệu gốc không có nhãn dị thường.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_anomalies = int(n_samples * contamination)
    
    # Copy data and labels (0 = Normal, 1 = Anomaly)
    X_synthetic = np.copy(X)
    y_anomaly = np.zeros(n_samples, dtype=int)
    
   
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        
        noise = np.random.uniform(low=-0.5, high=0.5, size=X.shape[1])
        X_synthetic[idx] = np.clip(X_synthetic[idx] + noise, 0.0, 1.0)
        y_anomaly[idx] = 1 # Đánh dấu là dị thường
        
    return X_synthetic, y_anomaly

def benchmark_signal_verifier(X, y):
    
    print("\n" + "="*80)
    print(" 1A. BENCHMARK SIGNAL QUALITY VERIFIER (CLASSIFICATION)")
    print("="*80)
    
    models = {
        'Random Forest (Proposed)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        print(f"[*] Đang huấn luyện và đánh giá: {name}...")
        
        acc_list, prec_list, rec_list, f1_list, inf_time_list = [], [], [], [], []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            
            model.fit(X_train, y_train)
            
            
            start_time = time.perf_counter()
            y_pred = model.predict(X_test)
            end_time = time.perf_counter()
            
            # Tính metrics
            inf_time_ms = ((end_time - start_time) / len(X_test)) * 1000 # Đổi ra milliseconds/sample
            
            acc_list.append(accuracy_score(y_test, y_pred))
            prec_list.append(precision_score(y_test, y_pred, zero_division=0))
            rec_list.append(recall_score(y_test, y_pred, zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, zero_division=0))
            inf_time_list.append(inf_time_ms)
            
        results.append({
            'Method': name,
            'Accuracy': np.mean(acc_list),
            'Precision': np.mean(prec_list),
            'Recall': np.mean(rec_list),
            'F1-Score': np.mean(f1_list),
            'Inference (ms)': np.mean(inf_time_list)
        })
        
    df_results = pd.DataFrame(results)
    
    # In ra Terminal
    print("\nKết quả Signal Verifier Benchmark:")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Lưu CSV và LaTeX
    df_results.to_csv("benchmark_1A_classification.csv", index=False)
    with open("table_1A_classification.tex", "w") as f:
        f.write(df_results.to_latex(index=False, float_format="%.4f", caption="Benchmark of Signal Verification Models", label="tab:signal_verifier"))
        
    print("✅ Đã lưu kết quả vào 'benchmark_1A_classification.csv' và mã LaTeX 'table_1A_classification.tex'")
    return df_results

def benchmark_anomaly_detector(X_base):
    """PHẦN 1B: Benchmark các mô hình Anomaly Detection (Thay thế/So sánh với Isolation Forest)"""
    print("\n" + "="*80)
    print(" 1B. BENCHMARK SPATIAL ANOMALY DETECTOR")
    print("="*80)
    
    # Tạo Ground Truth bằng cách tiêm nhiễu nhân tạo (Contamination = 5%)
    X_synthetic, y_true = inject_synthetic_anomalies(X_base, contamination=0.05)
    
    models = {
        'Isolation Forest (Proposed)': IsolationForest(contamination=0.05, random_state=42),
        'One-Class SVM': OneClassSVM(nu=0.05, kernel='rbf'),
        'Local Outlier Factor': LocalOutlierFactor(novelty=True, contamination=0.05),
        'Elliptic Envelope': EllipticEnvelope(contamination=0.05, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"[*] Đang huấn luyện và đánh giá: {name}...")
        
        # Huấn luyện trên tập KHÔNG có nhiễu (Dữ liệu an toàn - Semi-supervised approach)
        # Vì đây là Anomaly Detection, lý tưởng nhất là học từ môi trường bình thường
        if name == 'Local Outlier Factor':
            model.fit(X_base) 
        else:
            model.fit(X_base)
            
        # Suy luận trên tập dữ liệu đã bị tiêm nhiễu
        start_time = time.perf_counter()
        preds = model.predict(X_synthetic)
        end_time = time.perf_counter()
        
        inf_time_ms = ((end_time - start_time) / len(X_synthetic)) * 1000
        
        # Mapping nhãn của scikit-learn (1: Bình thường, -1: Bất thường) 
        # Sang chuẩn đánh giá (0: Bình thường, 1: Bất thường)
        y_pred = np.where(preds == -1, 1, 0)
        
        # Lấy điểm số để tính AUC (tùy thuộc vào model)
        if hasattr(model, "decision_function"):
            scores = -model.decision_function(X_synthetic) # Đảo dấu vì hàm decision trả về âm cho dị thường
        elif hasattr(model, "score_samples"):
            scores = -model.score_samples(X_synthetic)
        else:
            scores = y_pred
            
        auc = roc_auc_score(y_true, scores)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Tính False Alarm Rate (False Positive Rate)
        # FAR = FP / (FP + TN) = Báo động nhầm / Tổng số mẫu thực tế bình thường
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_alarm = fp / (fp + tn)
        
        results.append({
            'Method': name,
            'AUC': auc,
            'Recall (TPR)': recall,
            'F1-Score': f1,
            'False Alarm (FPR)': false_alarm,
            'Inference (ms)': inf_time_ms
        })
        
    df_results = pd.DataFrame(results)
    
    # In ra Terminal
    print("\nKết quả Anomaly Detector Benchmark:")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Lưu CSV và LaTeX
    df_results.to_csv("benchmark_1B_anomaly.csv", index=False)
    with open("table_1B_anomaly.tex", "w") as f:
        f.write(df_results.to_latex(index=False, float_format="%.4f", caption="Benchmark of Spatial Anomaly Detection Models", label="tab:anomaly_detector"))
        
    print("✅ Đã lưu kết quả vào 'benchmark_1B_anomaly.csv' và mã LaTeX 'table_1B_anomaly.tex'")
    return df_results

def run_full_component_benchmark():
    try:
        data = load_data()
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return
        
    # Tiền xử lý theo hàm có sẵn trong hệ thống của bạn
    X_scaled, y_class, _, _ = prepare_classification_data(data)
    
    # Chạy 1A
    benchmark_signal_verifier(X_scaled, y_class)
    
    # Chạy 1B (Dùng chính tập X_scaled làm input baseline)
    benchmark_anomaly_detector(X_scaled)
def plot_benchmark_results(csv_file, title, ylabel_bar="F1-Score (%)", ylabel_line="Inference Time (ms)"):
    """Vẽ biểu đồ so sánh cho Benchmark"""
    df = pd.read_csv(csv_file)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Vẽ Bar Chart (F1-Score hoặc Accuracy)
    color_bar = 'seagreen'
    sns.barplot(x='Method', y='F1-Score', data=df, ax=ax1, color=color_bar, alpha=0.8)
    ax1.set_ylabel(ylabel_bar, color=color_bar, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    
    # Vẽ Line Chart cho Inference Time
    ax2 = ax1.twinx()
    color_line = 'firebrick'
    ax2.plot(df['Method'], df['Inference (ms)'], color=color_line, marker='s', linewidth=2)
    ax2.set_ylabel(ylabel_line, color=color_line, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    plt.title(title, fontsize=14, fontweight='bold')
    ax1.set_xticklabels(df['Method'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    run_full_component_benchmark()