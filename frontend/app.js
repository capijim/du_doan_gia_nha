/**
 * Frontend logic – Dự đoán giá BĐS TP.HCM với bản đồ
 */
(function () {
    "use strict";

    var API_BASE = window.location.origin;
    var HCMC_CENTER = [10.7769, 106.7009];

    var map, marker;
    var selectedLat = null, selectedLng = null, selectedDistrict = "";
    var districts = [];
    var houseTypeMapping = {};
    var priceUnit = "tỷ VNĐ";

    // ── Khởi tạo bản đồ Leaflet ──────────────────────
    function initMap() {
        map = L.map("map").setView(HCMC_CENTER, 12);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            maxZoom: 18,
        }).addTo(map);

        map.on("click", function (e) {
            setLocation(e.latlng.lat, e.latlng.lng);
        });
    }

    function setLocation(lat, lng) {
        selectedLat = lat;
        selectedLng = lng;
        document.getElementById("lat").value = lat;
        document.getElementById("lng").value = lng;

        if (marker) {
            marker.setLatLng([lat, lng]);
        } else {
            marker = L.marker([lat, lng]).addTo(map);
        }
        marker.bindPopup("Vị trí đã chọn").openPopup();
        map.setView([lat, lng], 14);

        selectedDistrict = findNearestDistrict(lat, lng);
        var badge = document.getElementById("location-badge");
        badge.textContent = selectedDistrict.label || selectedDistrict.name;
        badge.classList.add("active");
        document.getElementById("location-coords").textContent =
            "(" + lat.toFixed(4) + ", " + lng.toFixed(4) + ")";

        document.getElementById("district-select").value = selectedDistrict.name;
    }

    function findNearestDistrict(lat, lng) {
        var minDist = Infinity, nearest = null;
        for (var i = 0; i < districts.length; i++) {
            var d = districts[i];
            var dist = Math.pow(lat - d.lat, 2) + Math.pow(lng - d.lng, 2);
            if (dist < minDist) {
                minDist = dist;
                nearest = d;
            }
        }
        return nearest || { name: "Unknown", label: "Không xác định", lat: lat, lng: lng };
    }

    // ── Load danh sách quận ───────────────────────────
    async function loadDistricts() {
        try {
            var res = await fetch(API_BASE + "/api/districts");
            districts = await res.json();

            var select = document.getElementById("district-select");
            districts.sort(function(a, b) { return a.label.localeCompare(b.label); });

            for (var i = 0; i < districts.length; i++) {
                var d = districts[i];
                L.circleMarker([d.lat, d.lng], {
                    radius: 6, color: "#2563eb", fillColor: "#93c5fd",
                    fillOpacity: 0.7, weight: 1.5
                }).addTo(map).bindTooltip(d.label, { permanent: false, direction: "top" });

                var opt = document.createElement("option");
                opt.value = d.name;
                opt.textContent = d.label;
                select.appendChild(opt);
            }

            select.addEventListener("change", function () {
                var val = this.value;
                if (!val) return;
                var found = districts.find(function(d) { return d.name === val; });
                if (found) setLocation(found.lat, found.lng);
            });
        } catch (e) { /* silent */ }
    }

    // ── Load thông tin mô hình ──────────────────────────
    async function loadModelInfo() {
        try {
            var res = await fetch(API_BASE + "/api/info");
            if (!res.ok) return;
            var data = await res.json();
            document.getElementById("metric-r2").textContent = data.r2;
            document.getElementById("metric-mae").textContent = data.mae;
            document.getElementById("metric-rmse").textContent = data.rmse;
            document.getElementById("metric-samples").textContent = data.n_samples;

            priceUnit = data.price_unit || "tỷ VNĐ";

            // Populate house type dropdown
            houseTypeMapping = data.house_type_mapping || {};
            var htSelect = document.getElementById("house_type");
            var types = Object.keys(houseTypeMapping).sort();
            for (var i = 0; i < types.length; i++) {
                var opt = document.createElement("option");
                opt.value = types[i];
                opt.textContent = types[i];
                htSelect.appendChild(opt);
            }
        } catch (e) { /* silent */ }
    }

    // ── Dự đoán ─────────────────────────────────────────
    var form = document.getElementById("predict-form");
    var btn = document.getElementById("btn-predict");
    var resultArea = document.getElementById("result-area");
    var resultSummary = document.getElementById("result-summary");

    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        if (!selectedLat || !selectedLng) {
            resultArea.innerHTML = '<p class="error-msg">Vui lòng chọn vị trí trên bản đồ trước!</p>';
            resultSummary.style.display = "none";
            return;
        }

        var payload = {
            land_area: parseFloat(document.getElementById("land_area").value),
            house_type: document.getElementById("house_type").value,
            bedrooms: parseInt(document.getElementById("bedrooms").value),
            toilets: parseInt(document.getElementById("toilets").value),
            total_floors: parseInt(document.getElementById("total_floors").value),
            lat: selectedLat,
            lng: selectedLng,
        };

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Đang dự đoán...';
        resultArea.innerHTML = "";
        resultSummary.style.display = "none";

        try {
            var res = await fetch(API_BASE + "/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            var data = await res.json();

            if (!res.ok) {
                resultArea.innerHTML = '<p class="error-msg">' + escapeHtml(data.error || "Lỗi không xác định.") + "</p>";
                return;
            }

            resultArea.innerHTML = "";
            resultSummary.style.display = "block";
            document.getElementById("price-value").textContent = data.gia_du_doan.toLocaleString("vi-VN");
            document.getElementById("price-unit").textContent = data.don_vi;

            var summary = "";
            summary += "<strong>Vị trí:</strong> " + escapeHtml(data.district_label) + "<br/>";
            summary += "<strong>Tọa độ:</strong> " + selectedLat.toFixed(4) + ", " + selectedLng.toFixed(4) + "<br/>";
            summary += "<strong>Diện tích:</strong> " + payload.land_area + " m²<br/>";
            summary += "<strong>Loại:</strong> " + escapeHtml(payload.house_type) + "<br/>";
            summary += "<strong>Phòng ngủ:</strong> " + payload.bedrooms + "<br/>";
            summary += "<strong>WC:</strong> " + payload.toilets + "<br/>";
            summary += "<strong>Số tầng:</strong> " + payload.total_floors + "<br/>";
            document.getElementById("input-summary").innerHTML = summary;

            if (marker) {
                marker.bindPopup(
                    "<b>" + escapeHtml(data.district_label) + "</b><br/>" +
                    "Giá dự đoán: <b>" + data.gia_du_doan.toLocaleString("vi-VN") + " " + escapeHtml(data.don_vi) + "</b>"
                ).openPopup();
            }
        } catch (err) {
            resultArea.innerHTML = '<p class="error-msg">Không kết nối được server. Hãy đảm bảo backend đang chạy.</p>';
        } finally {
            btn.disabled = false;
            btn.innerHTML = "🔮 Dự đoán giá";
        }
    });

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // ── Init ────────────────────────────────────────────
    initMap();
    loadDistricts();
    loadModelInfo();
})();
