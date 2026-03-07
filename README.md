# 🏠 Dự Đoán Giá Nhà

Ứng dụng web dự đoán giá nhà sử dụng **RandomForestRegressor** với giao diện trực quan.

## Công nghệ sử dụng

| Thư viện | Mục đích |
|---|---|
| **Pandas & NumPy** | Đọc file CSV, xử lý giá trị thiếu (NaN), tính toán ma trận |
| **Scikit-learn** | Huấn luyện mô hình RandomForestRegressor, đánh giá R² |
| **Matplotlib & Seaborn** | Vẽ Heatmap, Scatter plot, Feature Importance |
| **Joblib** | Lưu/load mô hình .pkl (không cần train lại) |
| **Flask** | Backend API phục vụ frontend |

## Cấu trúc dự án

```
du_doan_gia_nha/
├── backend/
│   ├── data/
│   │   └── housing_data.csv      # Dữ liệu mẫu (100 dòng)
│   ├── model/                    # Mô hình .pkl (tạo sau khi train)
│   ├── static/charts/            # Biểu đồ PNG (tạo sau khi train)
│   ├── train_model.py            # Script huấn luyện mô hình
│   └── app.py                    # Flask API server
├── frontend/
│   ├── index.html                # Giao diện chính
│   ├── style.css                 # CSS
│   └── app.js                    # Frontend logic
├── requirements.txt
└── README.md
```

## Hướng dẫn cài đặt & chạy

### 1. Cài Python
Tải và cài Python 3.9+ từ [python.org](https://www.python.org/downloads/).  
**Lưu ý:** Tick chọn "Add Python to PATH" khi cài đặt.

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Huấn luyện mô hình
```bash
cd backend
python train_model.py
```

Sau khi chạy, bạn sẽ thấy:
- File mô hình `backend/model/house_price_model.pkl`
- File metadata `backend/model/metadata.pkl`
- Biểu đồ trong `backend/static/charts/` (heatmap, scatter, feature importance)
- Các chỉ số đánh giá (R², MAE, RMSE) in trên terminal

### 4. Chạy web server
```bash
cd backend
python app.py
```

### 5. Mở trình duyệt
Truy cập **http://localhost:5000**

## Cách sử dụng
1. Nhập thông tin nhà (diện tích, số phòng, tuổi nhà, ...)
2. Nhấn **"Dự đoán giá"**
3. Xem kết quả giá dự đoán và các biểu đồ phân tích

## Đặc trưng (Features)

| Đặc trưng | Mô tả |
|---|---|
| Diện tích (m²) | Diện tích sử dụng |
| Số phòng ngủ | 1–5 phòng |
| Số phòng tắm | 1–4 phòng |
| Tuổi nhà (năm) | Số năm đã xây |
| Khoảng cách trung tâm (km) | Cách trung tâm thành phố |
| Tầng | Vị trí tầng |
| Có ban công | Có/Không |
| Có garage | Có/Không |

