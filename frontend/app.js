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
            var basePrice = data.gia_du_doan;
            document.getElementById("price-value").textContent = basePrice.toLocaleString("vi-VN");
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

            // Fetch nearby POIs and adjust price
            fetchNearbyPois(selectedLat, selectedLng, basePrice, data.don_vi, data.district_label);

            if (marker) {
                marker.bindPopup(
                    "<b>" + escapeHtml(data.district_label) + "</b><br/>" +
                    "Giá dự đoán: <b>" + basePrice.toLocaleString("vi-VN") + " " + escapeHtml(data.don_vi) + "</b>"
                ).openPopup();
            }
        } catch (err) {
            resultArea.innerHTML = '<p class="error-msg">Không kết nối được server. Hãy đảm bảo backend đang chạy.</p>';
        } finally {
            btn.disabled = false;
            btn.innerHTML =
                '<span class="btn-icon" aria-hidden="true">' +
                '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.3 4.7L19 9l-4.7 2.3L12 16l-2.3-4.7L5 9l4.7-2.3L12 2zm-6.5 12l1.4 2.8L9.7 18l-2.8 1.3L5.5 22l-1.4-2.7L1.3 18l2.8-1.2L5.5 14zm13 0l1.4 2.8 2.8 1.2-2.8 1.3L18.5 22l-1.4-2.7-2.8-1.3 2.8-1.2 1.4-2.8z"/></svg>' +
                '</span>' +
                '<span>Dự đoán giá</span>';
        }
    });

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function getPoiIconSvg(catKey) {
        var icons = {
            hospital: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9.5 3v3.2H6.3V9h3.2v3.2h2.8V9h3.2V6.2h-3.2V3H9.5zM5 13.5h14V21H5v-7.5zm2.4 2.1v3.3h9.2v-3.3H7.4z"/></svg>',
            school: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3l9.5 4.8L12 12.6 2.5 7.8 12 3zm-6.8 7.3V15c0 2.8 3 5 6.8 5s6.8-2.2 6.8-5v-4.7L12 14 5.2 10.3z"/></svg>',
            market: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 6.2l1.2-2.7h15.6L21 6.2v2.4a2.4 2.4 0 01-1.8 2.3V20H4.8v-9.1A2.4 2.4 0 013 8.6V6.2zm3.3 6.6V18h11.4v-5.2H6.3z"/></svg>',
            park: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M11.1 3.4a4.2 4.2 0 00-4.2 4.2c0 1.3.6 2.5 1.5 3.3H6.1a3.6 3.6 0 100 7.2H11v2.9H9.2V22h5.6v-1.9H13v-2.9h4.8a3.6 3.6 0 100-7.2h-2.3A4.2 4.2 0 0011.1 3.4z"/></svg>',
            transport: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 3.5h12a2 2 0 012 2v9.7a2 2 0 01-2 2h-.8l1.4 2.3-1.7 1-2-3.3H9l-2 3.3-1.7-1L6.8 17.2H6a2 2 0 01-2-2V5.5a2 2 0 012-2zm0 2v5.2h12V5.5H6zm0 7.3v2.4h12v-2.4H6z"/></svg>'
        };
        return icons[catKey] || '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="7"/></svg>';
    }

    // ── Quét tiện ích xung quanh (POI) ─────────────────
    var poiMarkers = [];

    async function fetchNearbyPois(lat, lng, basePrice, unit, districtLabel) {
        var poiSection = document.getElementById("poi-section");
        var poiLoading = document.getElementById("poi-loading");
        var poiGrid = document.getElementById("poi-grid");
        var poiBoost = document.getElementById("poi-boost");

        poiSection.style.display = "block";
        poiLoading.style.display = "flex";
        poiGrid.innerHTML = "";
        poiBoost.style.display = "none";

        // Remove old POI markers from map
        for (var k = 0; k < poiMarkers.length; k++) {
            map.removeLayer(poiMarkers[k]);
        }
        poiMarkers = [];

        try {
            var res = await fetch(API_BASE + "/api/nearby-pois", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat: lat, lng: lng, radius: 1200 }),
            });
            var data = await res.json();
            poiLoading.style.display = "none";

            document.getElementById("poi-radius").textContent = "(bán kính " + data.radius + "m)";

            var cats = Object.keys(data.pois);
            for (var i = 0; i < cats.length; i++) {
                var catKey = cats[i];
                var cat = data.pois[catKey];
                var card = document.createElement("div");
                card.className = "poi-card" + (cat.count > 0 ? " has-items" : "");

                var itemsHtml = "";
                if (cat.items && cat.items.length) {
                    for (var j = 0; j < cat.items.length; j++) {
                        var it = cat.items[j];
                        itemsHtml += '<div class="poi-item"><span>' + escapeHtml(it.name) + '</span><span class="poi-dist">' + it.distance_m + 'm</span></div>';
                    }
                } else {
                    itemsHtml = '<div class="poi-empty">Không tìm thấy</div>';
                }

                card.innerHTML =
                    '<div class="poi-card-header">' +
                    '<span class="poi-icon" aria-hidden="true">' + getPoiIconSvg(catKey) + '</span>' +
                    '<span class="poi-label">' + escapeHtml(cat.label) + '</span>' +
                    '<span class="poi-count">' + cat.count + '</span>' +
                    '</div>' +
                    '<div class="poi-items">' + itemsHtml + '</div>' +
                    (cat.boost_pct > 0 ? '<div class="poi-boost-tag">+' + cat.boost_pct + '%</div>' : '');

                poiGrid.appendChild(card);

                // Add markers on map for POI items
                if (cat.items) {
                    for (var m2 = 0; m2 < cat.items.length; m2++) {
                        var poi = cat.items[m2];
                        if (typeof poi.lat === "number" && typeof poi.lng === "number") {
                            var poiMarker = L.circleMarker([poi.lat, poi.lng], {
                                radius: 5,
                                color: "#ef4444",
                                fillColor: "#fca5a5",
                                fillOpacity: 0.85,
                                weight: 1,
                            }).addTo(map);
                            poiMarker.bindPopup(
                                "<b>" + escapeHtml(cat.label) + "</b><br/>" +
                                escapeHtml(poi.name) + "<br/>" +
                                "Khoảng cách: <b>" + poi.distance_m + "m</b>"
                            );
                            poiMarkers.push(poiMarker);
                        }
                    }
                }
            }

            // Show boost summary
            if (data.total_boost_pct > 0) {
                poiBoost.style.display = "block";
                document.getElementById("boost-value").textContent = "+" + data.total_boost_pct + "%";
                var adjusted = (basePrice * data.total_multiplier).toFixed(2);
                document.getElementById("adjusted-price").textContent = parseFloat(adjusted).toLocaleString("vi-VN") + " " + unit;

                // Update main price to adjusted
                document.getElementById("price-value").textContent = parseFloat(adjusted).toLocaleString("vi-VN");
                document.getElementById("price-unit").textContent = unit + " (đã điều chỉnh)";

                if (marker) {
                    marker.bindPopup(
                        "<b>" + escapeHtml(districtLabel) + "</b><br/>" +
                        "Giá gốc: <b>" + basePrice.toLocaleString("vi-VN") + " " + escapeHtml(unit) + "</b><br/>" +
                        "Tiện ích: <b>+" + data.total_boost_pct + "%</b><br/>" +
                        "Giá điều chỉnh: <b>" + parseFloat(adjusted).toLocaleString("vi-VN") + " " + escapeHtml(unit) + "</b>"
                    ).openPopup();
                }
            } else {
                poiBoost.style.display = "block";
                document.getElementById("boost-value").textContent = "+0%";
                document.getElementById("adjusted-price").textContent = basePrice.toLocaleString("vi-VN") + " " + unit;
            }
        } catch (e) {
            poiLoading.style.display = "none";
            poiGrid.innerHTML = '<div class="poi-empty">Không thể quét khu vực.</div>';
        }
    }

    // ── Tìm kiếm địa chỉ (Nominatim) ────────────────────
    var searchInput = document.getElementById("address-search");
    var searchResults = document.getElementById("search-results");
    var searchClear = document.getElementById("search-clear");
    var searchTimer = null;

    searchInput.addEventListener("input", function () {
        clearTimeout(searchTimer);
        var q = searchInput.value.trim();
        searchClear.style.display = q ? "block" : "none";
        if (q.length < 2) { searchResults.innerHTML = ""; searchResults.style.display = "none"; return; }
        searchTimer = setTimeout(function () { searchAddress(q); }, 350);
    });

    searchInput.addEventListener("keydown", function (e) {
        var items = searchResults.querySelectorAll("li");
        var active = searchResults.querySelector("li.active");
        var idx = Array.prototype.indexOf.call(items, active);
        if (e.key === "ArrowDown") {
            e.preventDefault();
            if (active) active.classList.remove("active");
            idx = (idx + 1) % items.length;
            items[idx].classList.add("active");
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (active) active.classList.remove("active");
            idx = (idx - 1 + items.length) % items.length;
            items[idx].classList.add("active");
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (active) active.click();
            else if (items.length) items[0].click();
        } else if (e.key === "Escape") {
            searchResults.innerHTML = ""; searchResults.style.display = "none";
        }
    });

    searchClear.addEventListener("click", function () {
        searchInput.value = "";
        searchResults.innerHTML = ""; searchResults.style.display = "none";
        searchClear.style.display = "none";
        searchInput.focus();
    });

    // Close results when clicking outside
    document.addEventListener("click", function (e) {
        if (!e.target.closest(".search-wrap")) {
            searchResults.innerHTML = ""; searchResults.style.display = "none";
        }
    });

    async function searchAddress(query) {
        try {
            var url = "https://nominatim.openstreetmap.org/search?" +
                "format=json&q=" + encodeURIComponent(query + ", Hồ Chí Minh, Việt Nam") +
                "&countrycodes=vn&limit=6&addressdetails=1&viewbox=106.35,10.95,107.02,10.35&bounded=1";
            var res = await fetch(url, { headers: { "Accept-Language": "vi" } });
            var data = await res.json();
            renderSearchResults(data);
        } catch (e) { /* silent */ }
    }

    function renderSearchResults(results) {
        searchResults.innerHTML = "";
        if (!results || !results.length) {
            searchResults.innerHTML = '<li class="no-result">Không tìm thấy địa chỉ</li>';
            searchResults.style.display = "block";
            return;
        }
        for (var i = 0; i < results.length; i++) {
            (function (r) {
                var li = document.createElement("li");
                li.innerHTML = '<span class="sr-name">' + escapeHtml(r.display_name.split(",").slice(0, 3).join(", ")) +
                    '</span><span class="sr-detail">' + escapeHtml(r.display_name) + '</span>';
                li.addEventListener("click", function () {
                    setLocation(parseFloat(r.lat), parseFloat(r.lon));
                    searchInput.value = r.display_name.split(",").slice(0, 3).join(", ");
                    searchResults.innerHTML = ""; searchResults.style.display = "none";
                });
                searchResults.appendChild(li);
            })(results[i]);
        }
        searchResults.style.display = "block";
    }

    // ── Init ────────────────────────────────────────────
    initMap();
    loadDistricts();
    loadModelInfo();
})();
