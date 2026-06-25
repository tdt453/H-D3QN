# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import DATASET_SCHEMA, COORDINATES_SCHEMA, EXTRA_FEATURES, CLASSIFICATION_THRESHOLD_RSSI, CLASSIFICATION_THRESHOLD_BS

def preprocess_ods_rssi_data(ods_path, pos_coords_path, output_path, start_unix=1700000000):
    """
    Preprocess a localization-style ODS dataset with sheets pos__1..pos__N and
    columns Modo 1..Modo M containing RSSI values.

    Produces the same semantic feature schema as preprocess_sigfox_data:
    mean_rssi, num_active_bs, Latitude, Longitude, hour (+ EXTRA_FEATURES if present).
    """
    xl = pd.ExcelFile(ods_path, engine="odf")
    pos_sheets = [s for s in xl.sheet_names if str(s).startswith("pos__")]
    if not pos_sheets:
        raise ValueError(f"No sheets starting with 'pos__' found in {ods_path}")

    # Load position coordinates (one x y per line). Map to nodeid=pos__k
    coords_raw = pd.read_csv(pos_coords_path, sep=r"\s+", header=None, names=["x", "y"])
    coords_raw["nodeid"] = [f"pos__{i}" for i in range(1, len(coords_raw) + 1)]

    records = []
    for sheet in pos_sheets:
        df = xl.parse(sheet, header=0)
        df.columns = [str(c).strip() for c in df.columns]

        # Keep only Modo columns (RSSI) and a sample index column
        modo_cols = [c for c in df.columns if str(c).strip().startswith("Modo ") or str(c).strip().startswith("Modo")]
        if not modo_cols:
            modo_cols = [c for c in df.columns if "Modo" in str(c)]

        # Drop a header-like first row if it contains strings like "Channel RSSI"
        if len(df) > 0 and any(isinstance(v, str) and "RSSI" in v for v in df.iloc[0].values):
            df = df.iloc[1:].reset_index(drop=True)

        # Determine sample index (fallback to row number if missing)
        sample_col = df.columns[0] if df.columns.size > 0 else None
        if sample_col is None:
            continue

        df[sample_col] = pd.to_numeric(df[sample_col], errors="coerce")
        df = df.dropna(subset=[sample_col])

        # Coerce RSSI columns to numeric
        for c in modo_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        melted = df.melt(id_vars=[sample_col], value_vars=modo_cols, var_name="gtw_id", value_name="gtw_rssi")
        melted["gtw_rssi"] = pd.to_numeric(melted["gtw_rssi"], errors="coerce")
        melted = melted.dropna(subset=["gtw_rssi"])

        # Normalize gateway id (Modo 1..10 -> 1..10)
        melted["gtw_id"] = (
            melted["gtw_id"]
            .astype(str)
            .str.strip()
            .str.replace("Modo", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        melted["gtw_id"] = pd.to_numeric(melted["gtw_id"], errors="coerce").astype("Int64")
        melted = melted.dropna(subset=["gtw_id"])

        # Create a synthetic unix timestamp (unique per pos and sample)
        try:
            pos_idx = int(str(sheet).split("__", 1)[1])
        except Exception:
            pos_idx = 0
        sample = melted[sample_col].astype(int)
        melted["timestamp"] = start_unix + pos_idx * 100000 + sample
        melted["nodeid"] = str(sheet)

        records.append(melted[["timestamp", "nodeid", "gtw_id", "gtw_rssi"]])

    if not records:
        raise ValueError("No usable RSSI records parsed from ODS.")

    raw = pd.concat(records, ignore_index=True)
    raw["hour"] = pd.to_datetime(raw["timestamp"], unit="s", errors="coerce").dt.hour

    # Aggregate to semantic state (1 row per timestamp,nodeid)
    agg = raw.groupby(["timestamp", "nodeid"], as_index=False).agg(
        mean_rssi=("gtw_rssi", "mean"),
        num_active_bs=("gtw_id", "nunique"),
        hour=("hour", "first"),
    )

    merged = pd.merge(agg, coords_raw, on="nodeid", how="left")
    merged["Latitude"] = pd.to_numeric(merged["x"], errors="coerce").fillna(0.0)
    merged["Longitude"] = pd.to_numeric(merged["y"], errors="coerce").fillna(0.0)

    core_features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    valid_extra = [f for f in EXTRA_FEATURES if f in merged.columns]
    final_cols = core_features + valid_extra

    semantic_df = merged[final_cols].dropna()
    semantic_df.to_csv(output_path, index=False)

    print(f"✅ ODS semantic features saved to '{output_path}'")
    print(f"✅ Generated Model Input Variables ({len(final_cols)} Features): {final_cols}")
    return semantic_df

def preprocess_sigfox_data(sigfox_path, bs_mapping_path, output_path):
    # 1. Load Raw Data
    df = pd.read_csv(sigfox_path, sep=DATASET_SCHEMA.get("separator", ","))
    df.columns = [str(col).strip().replace("'", "") for col in df.columns]

    # 2. Extract Time
    time_col = DATASET_SCHEMA["time_col"]
    if DATASET_SCHEMA["time_format"] == "unix":
        df["hour"] = pd.to_datetime(df[time_col], unit='s', errors='coerce').dt.hour
    else:
        df["hour"] = pd.to_datetime(df[time_col].str.replace("'", ""), errors='coerce').dt.hour

    # 3. Process RSSI and Dynamically merge by Time and Node
    rssi_col = DATASET_SCHEMA["rssi_col"]
    if rssi_col in df.columns:
        # Dynamic Mode (Lora Dataset style - Melted format)
        df["mean_rssi"] = pd.to_numeric(df[rssi_col], errors='coerce')
        node_col = DATASET_SCHEMA["node_id_col"]
        bs_col = DATASET_SCHEMA["bs_id_col"]
        
        agg_dict = {
            "mean_rssi": (rssi_col, "mean"),
            "num_active_bs": (bs_col, "nunique"),
            "hour": ("hour", "first")
        }
        # Add any EXTRA_FEATURES dynamically if they exist in the raw data
        for ef in EXTRA_FEATURES:
            if ef in df.columns:
                df[ef] = pd.to_numeric(df[ef], errors='coerce')
                agg_dict[ef] = (ef, "mean")
                
        # Aggregate to represent proper Reinforcement Learning States (1 transmission = 1 stat row)
        df = df.groupby([time_col, node_col]).agg(**agg_dict).reset_index()
    else:
        # Legacy Mode (Antwerp Dataset style - Pivoted BS format)
        rssi_cols = [col for col in df.columns if col.startswith(rssi_col)]
        rssi_data = df[rssi_cols].replace(-200, np.nan)
        df["mean_rssi"] = rssi_data.mean(axis=1)
        df["num_active_bs"] = rssi_data.notna().sum(axis=1)

    # 4. Integrate Geographical Coordinates
    try:
        coord_df = pd.read_csv(bs_mapping_path, sep=COORDINATES_SCHEMA.get("separator", ","))
        coord_df.columns = [str(col).strip() for col in coord_df.columns]
        
        node_id_raw = DATASET_SCHEMA.get("node_id_col", "nodeid")
        node_id_coord = COORDINATES_SCHEMA.get("node_id_col", "device ID")
        
        if node_id_raw in df.columns and node_id_coord in coord_df.columns:
            df[node_id_raw] = df[node_id_raw].astype(str).str.strip()
            coord_df[node_id_coord] = coord_df[node_id_coord].astype(str).str.strip()
            
            merged = pd.merge(df, coord_df, left_on=node_id_raw, right_on=node_id_coord, how='left')
            
            lat_col = COORDINATES_SCHEMA["latitude_col"]
            lon_col = COORDINATES_SCHEMA["longitude_col"]
            if lat_col in merged.columns and lon_col in merged.columns:
                merged["Latitude"] = pd.to_numeric(merged[lat_col], errors='coerce')
                merged["Longitude"] = pd.to_numeric(merged[lon_col], errors='coerce')
            df = merged
    except FileNotFoundError:
        print(f"Warning: Coordinate mapping file not found at {bs_mapping_path}. Proceeding with isolated semantic data.")

    # Fill default zeros if mapping failed
    if "Latitude" not in df.columns: df["Latitude"] = 0.0
    if "Longitude" not in df.columns: df["Longitude"] = 0.0

    # 5. Pack final customized Semantic State Vector
    core_features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    valid_extra = [f for f in EXTRA_FEATURES if f in df.columns]
    
    final_cols = core_features + valid_extra
    semantic_df = df[final_cols].dropna()

    semantic_df.to_csv(output_path, index=False)
    print(f"✅ Dynamic Semantic features saved to '{output_path}'")
    print(f"✅ Generated Model Input Variables ({len(final_cols)} Features): {final_cols}")
    
    return semantic_df

def get_dynamic_features():
    """Returns the standardized base features + any customized extra features"""
    return ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"] + EXTRA_FEATURES

def prepare_classification_data(data):
    """Dynamic Preparation for classification task"""
    features = [f for f in get_dynamic_features() if f in data.columns]
    
    # Calculate connection quality dynamically based on configured thresholds
    rssi_thresh = data['mean_rssi'].median() if CLASSIFICATION_THRESHOLD_RSSI == "auto" else CLASSIFICATION_THRESHOLD_RSSI
    if CLASSIFICATION_THRESHOLD_RSSI == "auto":
        print(f"🎯 Tự động tính điểm chuẩn RSSI: {rssi_thresh:.2f} dBm")
        
    data['label'] = ((data['mean_rssi'] > rssi_thresh) & 
                     (data['num_active_bs'] >= CLASSIFICATION_THRESHOLD_BS)).astype(int)
                     
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    return X_scaled, data['label'], features, scaler

def prepare_anomaly_data(data):
    """Dynamic Preparation for anomaly detection"""
    features = [f for f in get_dynamic_features() if f in data.columns]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    return X_scaled, features, scaler
