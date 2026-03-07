"""
Flask API – Dự đoán giá bất động sản TP.HCM.
Load mô hình .pkl đã train sẵn, nhận input từ frontend (bao gồm vị trí bản đồ) và trả về giá dự đoán.
"""

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from districts import DISTRICT_COORDS, DISTRICT_LABELS, get_nearest_district

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

model = None
metadata = None


def load_model():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Chưa có mô hình! Hãy chạy train_model.py trước."
        )
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    print("[INFO] Đã load mô hình thành công.")


# ─── API Endpoints ────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Nhận JSON, trả về giá dự đoán (tỷ VNĐ)."""
    if model is None:
        return jsonify({"error": "Mô hình chưa được load."}), 500

    data = request.get_json(force=True)

    required = ["land_area", "bedrooms", "toilets", "total_floors", "house_type", "lat", "lng"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Thiếu trường: {missing}"}), 400

    try:
        land_area = float(data["land_area"])
        bedrooms = float(data["bedrooms"])
        toilets = float(data["toilets"])
        total_floors = float(data["total_floors"])
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Dữ liệu không hợp lệ: {e}"}), 400

    # Encode house type
    house_type_str = str(data["house_type"])
    house_type_mapping = metadata.get("house_type_mapping", {})
    house_type_enc = house_type_mapping.get(house_type_str, 0)

    # Tìm quận gần nhất
    district = get_nearest_district(lat, lng)
    district_label = DISTRICT_LABELS.get(district, district)

    features = np.array([[land_area, bedrooms, toilets, total_floors, house_type_enc, lat, lng]])
    prediction = model.predict(features)[0]

    return jsonify({
        "gia_du_doan": round(float(prediction), 2),
        "don_vi": metadata.get("price_unit", "tỷ VNĐ"),
        "district": district,
        "district_label": district_label,
    })


@app.route("/api/info", methods=["GET"])
def info():
    """Trả về thông tin mô hình, chỉ số đánh giá, và house type mapping."""
    if metadata is None:
        return jsonify({"error": "Metadata chưa được load."}), 500
    return jsonify({
        "feature_names": metadata["feature_names"],
        "feature_labels": metadata["feature_labels"],
        "house_type_mapping": metadata.get("house_type_mapping", {}),
        "r2": metadata["r2"],
        "mae": metadata["mae"],
        "rmse": metadata["rmse"],
        "n_samples": metadata["n_samples"],
        "price_unit": metadata.get("price_unit", "tỷ VNĐ"),
    })


@app.route("/api/districts", methods=["GET"])
def districts():
    """Trả về danh sách quận với tọa độ."""
    result = []
    for name, coord in DISTRICT_COORDS.items():
        result.append({
            "name": name,
            "label": DISTRICT_LABELS.get(name, name),
            "lat": coord["lat"],
            "lng": coord["lng"],
        })
    return jsonify(result)


@app.route("/api/charts/<chart_name>")
def get_chart(chart_name):
    """Trả về file ảnh biểu đồ."""
    charts_dir = os.path.join(STATIC_DIR, "charts")
    return send_from_directory(charts_dir, chart_name)


# ─── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
