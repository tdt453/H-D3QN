# --- CẤU HÌNH INDOOR ---

# Tên file nguồn và file đích
RAW_DATA_FILENAME = "semantic_features_indoor_ods.csv" # Sử dụng file đã xử lý
BS_MAPPING_FILENAME = "indoor_pos_coords.txt"
PROCESSED_DATA_FILENAME = "semantic_features_indoor_ods.csv"

# Cấu hình schema cho dữ liệu Indoor (RSSI và vị trí tương đối)
DATASET_SCHEMA = {
    "time_col": "hour",           # Cột thời gian (nếu có)
    "time_format": None,
    "rssi_col": "mean_rssi",      # Cột cường độ tín hiệu trung bình
    "bs_id_col": "num_active_bs", # Số lượng BS kết nối
    "node_id_col": None,          # Indoor dataset thường gắn trực tiếp tọa độ
    "separator": ","              # Phân cách bằng dấu phẩy
}

# Cấu hình schema cho file tọa độ (Indoor vị trí x, y)
COORDINATES_SCHEMA = {
    "node_id_col": None,          # Sử dụng index hoặc tọa độ trực tiếp
    "latitude_col": "Latitude",   # Tương ứng với tọa độ X trong indoor_pos_coords.txt
    "longitude_col": "Longitude", # Tương ứng với tọa độ Y trong indoor_pos_coords.txt
    "separator": r"\s+"           # Phân cách bằng khoảng trắng (tab/space)
}

# Indoor thường cần các đặc trưng về mật độ kết nối hoặc biến thiên tín hiệu
EXTRA_FEATURES = ["num_active_bs", "hour"] 

# Ngưỡng RSSI trong nhà thường mạnh hơn, khoảng -75 đến -80 dBm
CLASSIFICATION_THRESHOLD_RSSI = -75 
CLASSIFICATION_THRESHOLD_BS = 1