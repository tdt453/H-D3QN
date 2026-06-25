# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def setup_plot_style():
    """Set up consistent plot style"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_training_progress(episodes, rewards, epsilons, save_path=None, window=5):
    """Plot DQN training progress with Moving Average smoothing for IEEE realism"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # [FIX] Cập nhật Moving Average (Smoothing) để biểu đồ mượt mà, chân thực hơn
    rewards_arr = np.array(rewards)
    if len(rewards_arr) >= window:
        box = np.ones(window) / window
        smoothed_rewards = np.convolve(rewards_arr, box, mode='valid')
        smoothed_episodes = np.array(episodes)[(window-1):]
    else:
        smoothed_rewards = rewards_arr
        smoothed_episodes = np.array(episodes)
    
    # Plot total reward: Hiển thị mờ dữ liệu gốc, in đậm đường MA
    ax1.plot(episodes, rewards, color='tab:blue', alpha=0.25, label='Raw Total Reward')
    ax1.plot(smoothed_episodes, smoothed_rewards, color='tab:blue', linewidth=2, label=f'Smoothed Trend (MA={window})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Learning Curve Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot epsilon
    ax2.plot(episodes, epsilons, label='Epsilon Decay', color='tab:red', linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax2.set_title('Agent Exploration Strategy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close() # Thay thế plt.show() bằng plt.close() để tránh treo

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Semantic Verification Performance")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, auc, title="ROC Curve", save_path=None):
    """Plot single ROC curve"""
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_anomaly_map(df, save_path=None):
    """Plot anomaly detection results on map"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Longitude", y="Latitude",
        hue="anomaly", palette={1: "blue", -1: "red"},
        data=df, alpha=0.7
    )
    plt.legend(title="Network Status", labels=["Normal Link", "Catastrophic Anomaly"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Anomaly Distribution")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(clf, feature_names, save_path=None):
    """Plot feature importance from Random Forest"""
    importances = clf.feature_importances_
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Semantic Feature Importance Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.ylabel("State Features")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, save_path=None):
    """Plot combined ROC curves for comparison"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve - Random Forest
    axs[0].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'AUC = {auc_rf:.2f}')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.3)
    
    # ROC Curve - Isolation Forest
    axs[1].plot(fpr_if, tpr_if, color='blue', lw=2, label=f'AUC = {auc_if:.2f}')
    axs[1].plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend(loc="lower right")
    axs[1].grid(True, alpha=0.3)
    
    # Add subplot labels
    fig.text(0.23, 0.02, '(a) ROC Curve - Random Forest Guard', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.77, 0.02, '(b) ROC Curve - Isolation Forest Guard', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()