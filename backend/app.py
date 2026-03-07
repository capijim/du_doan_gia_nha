"""
Flask API – Dự đoán giá bất động sản TP.HCM.
Load mô hình .pkl đã train sẵn, nhận input từ frontend (bao gồm vị trí bản đồ) và trả về giá dự đoán.
"""

import os
import math
import numpy as np
import joblib
import requests as http_requests
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


# ─── POI / Tiện ích xung quanh ────────────────────────────────

# Các loại tiện ích và hệ số điều chỉnh giá (% tăng khi có trong bán kính)
POI_CATEGORIES = {
    "hospital": {
        "label": "Bệnh viện",
        "icon": "🏥",
        "tags": '["amenity"="hospital"]',
        "boost_per_item": 0.015,   # +1.5% mỗi cơ sở, tối đa 3
        "max_count": 3,
    },
    "school": {
        "label": "Trường học",
        "icon": "🏫",
        "tags": '["amenity"~"school|university|college"]',
        "boost_per_item": 0.012,
        "max_count": 3,
    },
    "market": {
        "label": "Chợ / Siêu thị",
        "icon": "🛒",
        "tags": '["shop"~"supermarket|mall"]["name"],node["amenity"="marketplace"]',
        "boost_per_item": 0.010,
        "max_count": 3,
    },
    "park": {
        "label": "Công viên",
        "icon": "🌳",
        "tags": '["leisure"="park"]',
        "boost_per_item": 0.008,
        "max_count": 2,
    },
    "transport": {
        "label": "Bến xe / Ga tàu",
        "icon": "🚉",
        "tags": '["amenity"="bus_station"],node["railway"="station"]',
        "boost_per_item": 0.012,
        "max_count": 2,
    },
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _haversine_m(lat1, lon1, lat2, lon2):
    """Khoảng cách Haversine (mét)."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.route("/api/nearby-pois", methods=["POST"])
def nearby_pois():
    """Tìm tiện ích xung quanh vị trí, trả về danh sách + hệ số điều chỉnh giá."""
    data = request.get_json(force=True)
    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Cần lat và lng."}), 400

    radius = min(int(data.get("radius", 1000)), 3000)  # mét, tối đa 3km

    # Xây dựng Overpass query cho tất cả loại cùng lúc
    parts = []
    for cat_key, cat in POI_CATEGORIES.items():
        for tag_block in cat["tags"].split("],"):
            tag_block = tag_block.strip()
            if not tag_block.endswith("]"):
                tag_block += "]"
            # Determine element type prefix
            if tag_block.startswith("node"):
                parts.append(f"{tag_block}(around:{radius},{lat},{lng});")
            else:
                parts.append(f"node{tag_block}(around:{radius},{lat},{lng});")
                parts.append(f"way{tag_block}(around:{radius},{lat},{lng});")

    query = f"[out:json][timeout:10];(" + "".join(parts) + ");out center tags 50;"

    results = {}
    total_boost = 0.0

    try:
        resp = http_requests.post(OVERPASS_URL, data={"data": query}, timeout=12)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        # Overpass không khả dụng → trả kết quả rỗng, không lỗi
        for cat_key, cat in POI_CATEGORIES.items():
            results[cat_key] = {
                "label": cat["label"],
                "icon": cat["icon"],
                "count": 0,
                "items": [],
                "boost_pct": 0,
            }
        return jsonify({"pois": results, "total_boost_pct": 0, "total_multiplier": 1.0, "radius": radius})

    # Phân loại kết quả
    for cat_key, cat in POI_CATEGORIES.items():
        results[cat_key] = {
            "label": cat["label"],
            "icon": cat["icon"],
            "count": 0,
            "items": [],
            "boost_pct": 0,
        }

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "Không tên")
        # Tọa độ trung tâm
        el_lat = el.get("lat") or (el.get("center", {}).get("lat"))
        el_lng = el.get("lon") or (el.get("center", {}).get("lon"))
        if not el_lat or not el_lng:
            continue
        dist_m = round(_haversine_m(lat, lng, el_lat, el_lng))

        amenity = tags.get("amenity", "")
        shop = tags.get("shop", "")
        leisure = tags.get("leisure", "")
        railway = tags.get("railway", "")

        cat_key = None
        if amenity == "hospital":
            cat_key = "hospital"
        elif amenity in ("school", "university", "college"):
            cat_key = "school"
        elif amenity in ("marketplace", "bus_station") or shop in ("supermarket", "mall"):
            cat_key = "market" if shop or amenity == "marketplace" else "transport"
        elif leisure == "park":
            cat_key = "park"
        elif railway == "station":
            cat_key = "transport"

        if cat_key and results[cat_key]["count"] < POI_CATEGORIES[cat_key]["max_count"] * 3:
            results[cat_key]["items"].append({"name": name, "distance_m": dist_m})
            results[cat_key]["count"] += 1

    # Sắp xếp theo khoảng cách và tính boost
    for cat_key, cat in POI_CATEGORIES.items():
        items = results[cat_key]["items"]
        items.sort(key=lambda x: x["distance_m"])
        # Chỉ lấy top N gần nhất để tính boost
        effective = min(len(items), cat["max_count"])
        boost = round(effective * cat["boost_per_item"] * 100, 1)
        results[cat_key]["boost_pct"] = boost
        results[cat_key]["items"] = items[:6]  # Giới hạn hiển thị 6
        total_boost += boost

    total_boost = round(total_boost, 1)
    multiplier = round(1 + total_boost / 100, 4)

    return jsonify({
        "pois": results,
        "total_boost_pct": total_boost,
        "total_multiplier": multiplier,
        "radius": radius,
    })


@app.route("/api/charts/<chart_name>")
def get_chart(chart_name):
    """Trả về file ảnh biểu đồ."""
    charts_dir = os.path.join(STATIC_DIR, "charts")
    return send_from_directory(charts_dir, chart_name)


# ─── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
