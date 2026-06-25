# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 38,            
    'axes.labelsize': 38, 
    'axes.titlesize': 42,      
    'xtick.labelsize': 38, 
    'ytick.labelsize': 38, 
    'legend.fontsize': 38,
    'figure.dpi': 300
})

def plot_paper_style_metrics():
    datasets = {
        'Sigfox': {
            'Methods': ['IF', 'OCSVM', 'LOF', 'EE', 'AE'],
            'Accuracy': [94.46, 94.56, 95.49, 92.72, 94.39],
            'Latency': [0.005, 0.036, 0.010, 0.0001, 0.001]
        },
        'LoRaWAN': {
            'Methods': ['IF', 'OCSVM', 'LOF', 'EE', 'AE'],
            'Accuracy': [92.69, 94.12, 99.42, 92.63, 97.04],
            'Latency': [0.004, 0.800, 0.006, 0.0001, 0.0001]
        },
        'Indoor': {
            'Methods': ['IF', 'OCSVM', 'LOF', 'EE', 'AE'],
            'Accuracy': [93.64, 93.55, 96.91, 90.73, 92.73],
            'Latency': [0.008, 0.006, 0.003, 0.0001, 0.001]
        }
    }

    
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    env_names = ['(a) Sigfox', '(b) LoRaWAN', '(c) Indoor']

    for i, (env, data) in enumerate(datasets.items()):
        ax1 = axes[i]
        x = np.arange(len(data['Methods']))
        
        # Bar chart
        ax1.bar(x, data['Accuracy'], 0.6, color='#5b7db1', edgecolor='black', alpha=0.9)
        ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=28)
        ax1.set_ylim(0, 100) # Nới rộng trục Y để không bị vướng chữ
        ax1.set_xticks(x)
        ax1.set_xticklabels(data['Methods'], fontsize=28)
        
        # Line chart
        ax2 = ax1.twinx()
        ax2.plot(x, data['Latency'], color='#e67e22', marker='o', linewidth=4, markersize=12)
        ax2.set_ylabel('Latency (ms)', color='#e67e22', fontweight='bold', fontsize=28)
        ax2.tick_params(axis='y', labelcolor='#e67e22', labelsize=28)
        
        # Annotate values (FontSize 24)
        for j, v in enumerate(data['Accuracy']):
            ax1.text(j, v + 5, f"{v:.1f}", ha='center', fontsize=24, fontweight='bold')
        for j, v in enumerate(data['Latency']):
            ax2.text(j, v + (max(data['Latency'])*0.08), f"{v:.4f}", ha='center', color='#e67e22', fontsize=22)
            
        ax1.set_title(env_names[i], fontsize=32, fontweight='bold', pad=20)

    plt.tight_layout(pad=3.0) 
    plt.savefig('Figure_3_Combined_Metrics.pdf')
    print(" Figure_3_Combined_Metrics.pdf")

if __name__ == "__main__":
    plot_paper_style_metrics()