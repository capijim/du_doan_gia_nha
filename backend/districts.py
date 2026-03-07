"""
Tọa độ trung tâm 23 quận/huyện TP.HCM – dùng làm feature cho mô hình.
Tên quận sử dụng tiếng Việt (khớp với dataset real_estate_listings.csv).
"""

import re

DISTRICT_COORDS = {
    "Quận 1":            {"lat": 10.7756, "lng": 106.7019},
    "Quận 3":            {"lat": 10.7835, "lng": 106.6862},
    "Quận 4":            {"lat": 10.7578, "lng": 106.7013},
    "Quận 5":            {"lat": 10.7540, "lng": 106.6633},
    "Quận 6":            {"lat": 10.7480, "lng": 106.6352},
    "Quận 7":            {"lat": 10.7340, "lng": 106.7220},
    "Quận 8":            {"lat": 10.7400, "lng": 106.6600},
    "Quận 10":           {"lat": 10.7730, "lng": 106.6680},
    "Quận 11":           {"lat": 10.7630, "lng": 106.6500},
    "Quận 12":           {"lat": 10.8670, "lng": 106.6540},
    "Quận Bình Thạnh":   {"lat": 10.8105, "lng": 106.7091},
    "Quận Bình Tân":     {"lat": 10.7652, "lng": 106.6040},
    "Quận Gò Vấp":       {"lat": 10.8386, "lng": 106.6652},
    "Quận Phú Nhuận":    {"lat": 10.7990, "lng": 106.6800},
    "Quận Tân Bình":     {"lat": 10.8014, "lng": 106.6528},
    "Quận Tân Phú":      {"lat": 10.7920, "lng": 106.6280},
    "TP. Thủ Đức":       {"lat": 10.8450, "lng": 106.7700},
    "Huyện Bình Chánh":  {"lat": 10.7380, "lng": 106.5935},
    "Huyện Nhà Bè":      {"lat": 10.6880, "lng": 106.7320},
    "Huyện Củ Chi":       {"lat": 11.0100, "lng": 106.4930},
    "Huyện Hóc Môn":     {"lat": 10.8860, "lng": 106.5920},
    "Huyện Cần Giờ":     {"lat": 10.4110, "lng": 106.9520},
}

# Tên hiển thị cho UI
DISTRICT_LABELS = {k: k for k in DISTRICT_COORDS}


def extract_district(location: str) -> str:
    """Trích xuất tên quận/huyện từ chuỗi Location.
    VD: 'Phường 15, Quận Bình Thạnh' → 'Quận Bình Thạnh'
        'Huyện Nhà Bè, TP.HCM' → 'Huyện Nhà Bè'
        'Phường X, TP. Thủ Đức - Quận 9' → 'TP. Thủ Đức'
    """
    parts = [p.strip() for p in location.split(",")]

    for part in parts:
        if "TP. Thủ Đức" in part:
            return "TP. Thủ Đức"

    for part in reversed(parts):
        if part == "TP.HCM":
            continue
        for district_name in DISTRICT_COORDS:
            if district_name in part:
                return district_name

    return None


def get_nearest_district(lat: float, lng: float) -> str:
    """Tìm quận gần nhất dựa trên tọa độ (Euclidean distance)."""
    min_dist = float("inf")
    nearest = None
    for name, coord in DISTRICT_COORDS.items():
        dist = (lat - coord["lat"]) ** 2 + (lng - coord["lng"]) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest = name
    return nearest
