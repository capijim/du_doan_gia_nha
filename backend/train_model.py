"""
Script huấn luyện mô hình dự đoán giá bất động sản TP.HCM.
- Đọc dữ liệu CSV (real_estate_listings.csv) bằng Pandas & NumPy
- Parse giá, diện tích, số phòng, quận từ text
- Gán tọa độ lat/lng theo quận → feature vị trí
- Xử lý giá trị thiếu (NaN)
- Huấn luyện RandomForestRegressor (Scikit-learn)
- Đánh giá R²
- Vẽ biểu đồ Heatmap, Scatter, Feature Importance (Matplotlib & Seaborn)
- Lưu mô hình bằng Joblib (.pkl)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from districts import DISTRICT_COORDS, DISTRICT_LABELS, extract_district

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "real_estate_listings.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# Features dùng cho mô hình
FEATURE_NAMES = ["land_area", "bedrooms", "toilets", "total_floors", "house_type_enc", "lat", "lng"]

FEATURE_LABELS = {
    "land_area": "Diện tích (m²)",
    "bedrooms": "Số phòng ngủ",
    "toilets": "Số WC",
    "total_floors": "Số tầng",
    "house_type_enc": "Loại nhà",
    "lat": "Vĩ độ (Latitude)",
    "lng": "Kinh độ (Longitude)",
    "price_ty": "Giá (tỷ VNĐ)",
}


def parse_price(price_str: str) -> float:
    """Parse giá từ text sang số (đơn vị: tỷ VNĐ).
    VD: '4,16 tỷ' → 4.16, '6 tỷ 500 triệu' → 6.5, '800 triệu' → 0.8
    """
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).strip().lower()
    total = 0.0
    # Match "X tỷ"
    m_ty = re.search(r"([\d.,]+)\s*tỷ", s)
    if m_ty:
        total += float(m_ty.group(1).replace(",", "."))
    # Match "X triệu"
    m_tr = re.search(r"([\d.,]+)\s*triệu", s)
    if m_tr:
        total += float(m_tr.group(1).replace(",", ".")) / 1000
    if total == 0:
        return np.nan
    return total


def parse_area(area_str: str) -> float:
    """Parse diện tích: '36 m² (3,2x12,0)' → 36.0, '133,4 m²' → 133.4, '3.554,3 m²' → 3554.3"""
    if pd.isna(area_str):
        return np.nan
    m = re.search(r"([\d.,]+)\s*m²", str(area_str))
    if m:
        raw = m.group(1)
        # Vietnamese number format: 3.554,3 means 3554.3 (dot=thousands, comma=decimal)
        raw = raw.replace(".", "")  # remove thousands separator
        raw = raw.replace(",", ".")  # decimal comma → dot
        try:
            return float(raw)
        except ValueError:
            return np.nan
    return np.nan


def parse_int_from_text(text: str) -> float:
    """Parse số nguyên từ text: '4 phòng' → 4, '3 WC' → 3."""
    if pd.isna(text):
        return np.nan
    m = re.search(r"(\d+)", str(text))
    return float(m.group(1)) if m else np.nan


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Đọc CSV, parse các cột, gán tọa độ, xử lý NaN."""
    df = pd.read_csv(path)
    print(f"[INFO] Đọc được {len(df)} dòng dữ liệu.")

    # Parse các cột
    df["price_ty"] = df["Price"].apply(parse_price)
    df["land_area"] = df["Land Area"].apply(parse_area)
    df["bedrooms"] = df["Bedrooms"].apply(parse_int_from_text)
    df["toilets"] = df["Toilets"].apply(parse_int_from_text)
    df["total_floors"] = df["Total Floors"]

    # Encode loại nhà
    df["house_type_enc"] = LabelEncoder().fit_transform(df["Type of House"])

    # Trích xuất quận và gán tọa độ
    df["district"] = df["Location"].apply(extract_district)
    df["lat"] = df["district"].map(lambda d: DISTRICT_COORDS.get(d, {}).get("lat") if d else None)
    df["lng"] = df["district"].map(lambda d: DISTRICT_COORDS.get(d, {}).get("lng") if d else None)

    # Bỏ dòng không có giá hoặc không xác định quận
    before = len(df)
    df = df.dropna(subset=["price_ty", "lat", "lng", "land_area"])
    print(f"[INFO] Bỏ {before - len(df)} dòng thiếu giá/vị trí/diện tích → còn {len(df)}.")

    # Điền NaN cho cột phụ bằng median
    for col in ["bedrooms", "toilets", "total_floors"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Lọc outlier giá (percentile 1–99)
    q_low = df["price_ty"].quantile(0.01)
    q_high = df["price_ty"].quantile(0.99)
    before = len(df)
    df = df[(df["price_ty"] >= q_low) & (df["price_ty"] <= q_high)]
    print(f"[INFO] Lọc outlier giá ({q_low:.2f}–{q_high:.2f} tỷ) → còn {len(df)} dòng.")

    # Lọc diện tích hợp lý
    df = df[(df["land_area"] >= 5) & (df["land_area"] <= 2000)]
    print(f"[INFO] Sau lọc diện tích: {len(df)} dòng.")

    return df


def plot_heatmap(df: pd.DataFrame):
    """Vẽ heatmap tương quan."""
    plt.figure(figsize=(9, 7))
    plot_cols = FEATURE_NAMES + ["price_ty"]
    corr = df[plot_cols].corr()
    labels = [FEATURE_LABELS.get(c, c) for c in plot_cols]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                xticklabels=labels, yticklabels=labels)
    plt.title("Ma trận tương quan – BĐS TP.HCM")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Lưu heatmap: {path}")


def plot_scatter(y_test, y_pred):
    """Scatter plot giá thực tế vs dự đoán."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=15, edgecolors="k", linewidths=0.2)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Đường lý tưởng")
    plt.xlabel("Giá thực tế (tỷ VNĐ)")
    plt.ylabel("Giá dự đoán (tỷ VNĐ)")
    plt.title("So sánh giá thực tế và giá dự đoán")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Lưu scatter plot: {path}")


def plot_feature_importance(model, feature_names):
    """Biểu đồ feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in indices]
    plt.barh(range(len(indices)), importances[indices], align="center", color="steelblue")
    plt.yticks(range(len(indices)), labels)
    plt.xlabel("Mức độ quan trọng")
    plt.title("Feature Importance – RandomForestRegressor")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Lưu feature importance: {path}")


def plot_price_by_district(df: pd.DataFrame):
    """Biểu đồ giá trung bình theo quận."""
    avg = df.groupby("district")["price_ty"].median().sort_values(ascending=True)
    labels_vi = [DISTRICT_LABELS.get(d, d) for d in avg.index]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(avg)), avg.values, color="coral", edgecolor="k", linewidth=0.3)
    plt.yticks(range(len(avg)), labels_vi)
    plt.xlabel("Giá trung vị (tỷ VNĐ)")
    plt.title("Giá BĐS trung vị theo quận – TP.HCM")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "price_by_district.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Lưu price_by_district: {path}")


def train_model():
    """Pipeline: load → clean → train → evaluate → save."""
    # 1. Đọc & xử lý
    df = load_and_clean_data(DATA_PATH)

    # 2. Vẽ biểu đồ phân tích
    plot_heatmap(df)
    plot_price_by_district(df)

    # 3. Lưu house type mapping để dùng khi predict
    house_type_le = LabelEncoder()
    house_type_le.fit(df["Type of House"])
    df["house_type_enc"] = house_type_le.transform(df["Type of House"])

    # 4. Tách features / target
    X = df[FEATURE_NAMES].values
    y = df["price_ty"].values

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Huấn luyện
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Huấn luyện mô hình hoàn tất.")

    # 7. Đánh giá
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[METRIC] R² = {r2:.4f}")
    print(f"[METRIC] MAE = {mae:.2f} tỷ VNĐ")
    print(f"[METRIC] RMSE = {rmse:.2f} tỷ VNĐ")

    # 8. Vẽ biểu đồ
    plot_scatter(y_test, y_pred)
    plot_feature_importance(model, FEATURE_NAMES)

    # 9. Lưu mô hình
    model_path = os.path.join(MODEL_DIR, "house_price_model.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] Lưu mô hình: {model_path}")

    # 10. Lưu metadata + house type encoder
    house_type_mapping = {name: int(code) for name, code in
                          zip(house_type_le.classes_, house_type_le.transform(house_type_le.classes_))}
    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "house_type_mapping": house_type_mapping,
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "n_samples": len(df),
        "price_unit": "tỷ VNĐ",
    }
    metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"[INFO] Lưu metadata: {metadata_path}")
    print(f"[INFO] House type mapping: {house_type_mapping}")

    return model, metadata


if __name__ == "__main__":
    train_model()
