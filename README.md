# H-D3QN: Hybrid-Guard Reinforcement Learning for Robust IoT Resource Allocation

This repository contains the official implementation and dataset for the paper: **"H-D3QN: Hybrid-Guard Reinforcement Learning for Robust IoT Resource Allocation"**.

## Overview
The H-D3QN framework integrates a Dueling Double Deep Q-Network (D3QN) with Prioritized Experience Replay (PER) and a multi-stage Hybrid-Guard mechanism. It is designed to optimize resource allocation in dynamic IoT environments by proactively mitigating deceptive channel conditions through spatial anomaly detection and signal quality verification.

## Repository Structure
* `data/`: Directory containing all evaluation datasets (Sigfox, LoRaWAN, and Indoor ODS).
* `config.py`: Global configuration parameters and dataset schemas.
* `preprocessing.py`: Feature extraction and data normalization pipeline.
* `environment.py`: Telecom IoT Environment simulation (MDP formulation).
* `models.py`: Architectures for the Dueling DQN and Prioritized Replay Memory.
* `training.py`: Main Deep Reinforcement Learning training loop.
* `main.py`: The entry point to execute the complete end-to-end pipeline.
* `main_results_benchmark.py`: Evaluates H-D3QN against standard DRL baselines (DQN, DDQN, Dueling DQN).
* `ml_components_benchmark.py`: Ablation studies for the Signal Verifier and Spatial Guard layers.
* `latency_analysis.py`: Profiling script to validate URLLC strict latency constraints.
* `shap_analysis.py`: Explainable AI evaluation for feature importance.
* `visualization.py` & `plot_tradeoff_analysis.py`: Scripts for generating publication-ready IEEE-style figures.
  ### Dataset Description

The datasets used in this repository are divided into three scenarios: Sigfox, LoRaWAN, and Indoor. 

| Dataset | File Name | Description |
| :--- | :--- | :--- |
| **Indoor** | `indoor_raw_rssi.ods` | Raw RSSI measurements collected in the indoor environment. |
| | `indoor_pos_coords.txt` | Coordinates for the indoor positions. |
| | `semantic_features_indoor_ods.csv` | Extracted semantic features for the indoor dataset. |
| **Sigfox** | `sigfox_dataset_antwerp.csv` | Original Sigfox dataset collected in Antwerp. |
| | `sigfox_bs_mapping.csv` | Base station mapping details for the Sigfox network. |
| | `semantic_features_antwerp.csv` | Extracted semantic features for the Sigfox dataset. |
| **LoRaWAN**| `sensors_data.csv` | Raw sensory and transmission data from LoRaWAN nodes. |
| | `nodes_coordinates.txt` | Geographical coordinates and distance to gateway for LoRaWAN nodes. |
| | `semantic_features_lorawan.csv` | Extracted semantic features for the LoRaWAN dataset. |

## Requirements
Ensure you have Python 3.8+ installed. Install all required dependencies by running:
```bash
pip install -r requirements.txt
How to Run
1. End-to-End Pipeline

To run the complete system (Preprocessing -> Hybrid Guard -> DRL Training -> Evaluation), execute:

Bash
python main.py
2. Reproduce Main Benchmarks
To reproduce the convergence, reliability, and latency results against baseline DRL models:

Bash
python main_results_benchmark.py
3. Component Ablation Studies
To evaluate the isolated performance of different Machine Learning components (Random Forest, Isolation Forest, LOF, Autoencoder, etc.):

Bash
python ml_components_benchmark.py
4. URLLC Latency Profiling
To measure the exact edge inference latency and computational complexity:

Bash
python latency_analysis.py
