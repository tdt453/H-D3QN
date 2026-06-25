# -*- coding: utf-8 -*-
"""Main script to run the complete SOTA pipeline for IoT Communication"""

import warnings
warnings.filterwarnings('ignore')

import sys
import matplotlib
matplotlib.use("Agg") # Đảm bảo Matplotlib chạy an toàn không cần GUI
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Import các module đã được nâng cấp
from preprocessing import (preprocess_sigfox_data, prepare_classification_data, 
                           prepare_anomaly_data, get_dynamic_features, preprocess_ods_rssi_data)
from config import RAW_DATA_FILENAME, BS_MAPPING_FILENAME, PROCESSED_DATA_FILENAME

# LƯU Ý: Import hàm huấn luyện RL mới
from training import train_dueling_double_dqn, train_random_forest, train_isolation_forest
from evaluation import evaluate_classifier, evaluate_anomaly_detector, calculate_roc_metrics
from visualization import (plot_confusion_matrix, plot_roc_curve, 
                           plot_anomaly_map, plot_feature_importance, plot_combined_roc_curves)
from utils import split_data

def main():
    print("=== SOTA: Semantic-Aware Reinforcement Learning for Resource-Efficient IoT Communication ===\n")
    
    # --- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ---
    print("Step 1: Preprocessing data...")
    raw_path = f"data/{RAW_DATA_FILENAME}"
    mapping_path = f"data/{BS_MAPPING_FILENAME}"
    output_path = f"data/{PROCESSED_DATA_FILENAME}"

    if str(RAW_DATA_FILENAME).lower().endswith(".ods"):
        data = preprocess_ods_rssi_data(
            ods_path=raw_path,
            pos_coords_path=mapping_path,
            output_path=output_path,
        )
    else:
        data = preprocess_sigfox_data(
            sigfox_path=raw_path,
            bs_mapping_path=mapping_path,
            output_path=output_path,
        )
    
    # --- BƯỚC 2: PHÁT HIỆN DỊ THƯỜNG (HYBRID AI CORE) ---
    # Phải chạy trước RL để làm hệ thống giám sát môi trường
    print("\nStep 2: Training Isolation Forest for Hybrid AI Integration...")
    X_scaled_anomaly, feature_names_anomaly, scaler_anomaly = prepare_anomaly_data(data)
    iso_forest, anomaly_labels = train_isolation_forest(X_scaled_anomaly)
    
    data_with_anomalies = data.copy()
    data_with_anomalies["anomaly"] = anomaly_labels
    y_true_anomaly, anomaly_scores, auc_anomaly = evaluate_anomaly_detector(
        iso_forest, X_scaled_anomaly, anomaly_labels
    )
    
    # --- BƯỚC 3: HUẤN LUYỆN REINFORCEMENT LEARNING ---
    print("\nStep 3: Dueling Double DQN Training with PER & Hybrid AI...")
    features = [f for f in get_dynamic_features() if f in data.columns]
    
    # Lấy state đã normalize làm dữ liệu mồi cho Environment
    states = X_scaled_anomaly 
    
    # Huấn luyện với 100 episodes, truyền iso_forest vào làm Anomaly Detector
    policy_net, target_net, losses, total_rewards = train_dueling_double_dqn(
        states, 
        num_episodes=100, 
        anomaly_detector=iso_forest
    )
    
    # --- BƯỚC 4: PHÂN LOẠI RANDOM FOREST (BENCHMARK TRUYỀN THỐNG) ---
    print("\nStep 4: Random Forest Classification (Baseline Benchmark)...")
    X_scaled, y, feature_names, scaler = prepare_classification_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.3)
    
    rf_clf = train_random_forest(X_train, y_train)
    y_pred, y_prob, accuracy = evaluate_classifier(rf_clf, X_test, y_test)
    
    # --- BƯỚC 5: TRỰC QUAN HÓA KẾT QUẢ ---
    print("\nStep 5: Generating publication-ready visualizations...")
    
    # 5.1 Biểu đồ Huấn luyện RL (Vẽ trực tiếp từ list trả về, không cần file txt cũ)
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, label='Total Reward per Episode', color='tab:blue', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Dueling Double DQN Learning Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rl_training_rewards.pdf', dpi=300)
    plt.close()
    
    # Biểu đồ Loss của RL
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='TD Loss', color='tab:red', alpha=0.6)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Dueling Double DQN Loss Convergence', fontsize=14, fontweight='bold')
    plt.yscale('log') # Loss thường vẽ ở thang logarit để dễ quan sát
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rl_training_loss.pdf', dpi=300)
    plt.close()

    # 5.2 Các biểu đồ Machine Learning truyền thống
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.pdf')
    plot_feature_importance(rf_clf, feature_names, 'feature_importance.pdf')
    plot_anomaly_map(data_with_anomalies, 'anomaly_map.pdf')
    
    # 5.3 Biểu đồ ROC
    fpr_rf, tpr_rf, auc_rf = calculate_roc_metrics(y_test, y_prob)
    fpr_if, tpr_if, auc_if = calculate_roc_metrics(y_true_anomaly, anomaly_scores)
    
    plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, 
                             'combined_ROC_curves_with_labels.pdf')
    
    print("\n=== All tasks completed successfully! Check the generated PDF files. ===")

if __name__ == "__main__":
    main()