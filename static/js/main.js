document.addEventListener("DOMContentLoaded", async () => {
    const API_BASE = window.location.origin.includes("5000") ? "" : "http://127.0.0.1:5000";

    // =========================================================================
    // 1. WELCOME POPUP MODAL & NAVIGATION SWITCHING
    // =========================================================================
    const welcomeModal = document.getElementById("welcomeModal");
    const closeWelcomeX = document.getElementById("closeWelcomeX");
    const launchDashboardBtn = document.getElementById("launchDashboardBtn");
    const openWelcomeBtn = document.getElementById("openWelcomeBtn");

    function hideWelcome() { welcomeModal.classList.remove("active"); }
    function showWelcome() { welcomeModal.classList.add("active"); }

    closeWelcomeX.addEventListener("click", hideWelcome);
    launchDashboardBtn.addEventListener("click", hideWelcome);
    openWelcomeBtn.addEventListener("click", showWelcome);

    function activateModule(targetId) {
        document.querySelectorAll(".nav-btn").forEach(b => {
            b.classList.toggle("active", b.dataset.target === targetId);
        });
        document.querySelectorAll(".module-view").forEach(v => {
            v.classList.toggle("active", v.id === targetId);
        });
    }

    document.querySelectorAll(".nav-btn").forEach(btn => {
        btn.addEventListener("click", () => activateModule(btn.dataset.target));
    });

    document.querySelectorAll(".wg-card").forEach(card => {
        card.addEventListener("click", () => {
            activateModule(card.dataset.jump);
            hideWelcome();
        });
    });

    // =========================================================================
    // 2. DOM REFERENCES & CHART INSTANCES
    // =========================================================================
    const productSelect = document.getElementById("productSelect");
    const pricingProductSelect = document.getElementById("pricingProductSelect");
    const recCategoryFilter = document.getElementById("recCategoryFilter");
    const recTopN = document.getElementById("recTopN");
    const diversitySlider = document.getElementById("diversitySlider");
    const recTableBody = document.getElementById("recTableBody");

    const timeInput = document.getElementById("timeInput");
    const pagesInput = document.getElementById("pagesInput");
    const pastInput = document.getElementById("pastInput");
    const additionsInput = document.getElementById("additionsInput");
    const discountInput = document.getElementById("discountInput");
    const cartInput = document.getElementById("cartInput");

    const demandInput = document.getElementById("demandInput");
    const compPriceInput = document.getElementById("compPriceInput");
    const timeSlotInput = document.getElementById("timeSlotInput");

    let allProducts = [];
    let simChartInst = null;
    let radarChartInst = null;
    let featImpChartInst = null;
    let convDoughnutInst = null;
    let pricingChartInst = null;
    let profitCompareInst = null;
    let adminCatChartInst = null;

    // =========================================================================
    // 3. INITIALIZE CATALOG & ADMIN TABLE HYDRATION
    // =========================================================================
    async function loadInitialState() {
        const res = await fetch(`${API_BASE}/api/init`);
        const data = await res.json();
        allProducts = data.products;
        syncDropdowns(allProducts);
        renderAdminTable(allProducts);
        renderAdminChart(data.stats);
        updateAdminKPIs(data.stats);
    }

    function syncDropdowns(products) {
        const currentRecVal = productSelect.value || products[0]?.Product_ID;
        const currentPriceVal = pricingProductSelect.value || products[0]?.Product_ID;

        productSelect.innerHTML = products.map(p =>
            `<option value="${p.Product_ID}" data-price="${p.Base_Price}">#${p.Product_ID} — ${p.Product_Name} — ₹${p.Base_Price.toLocaleString("en-IN")} (${p.Category})</option>`
        ).join("");

        pricingProductSelect.innerHTML = products.map(p =>
            `<option value="${p.Product_ID}" data-price="${p.Base_Price}">#${p.Product_ID} — ${p.Product_Name} (Base: ₹${p.Base_Price.toLocaleString("en-IN")} | Stock: ${p.Stock})</option>`
        ).join("");

        if ([...productSelect.options].some(o => o.value == currentRecVal)) productSelect.value = currentRecVal;
        if ([...pricingProductSelect.options].some(o => o.value == currentPriceVal)) pricingProductSelect.value = currentPriceVal;
    }

    // =========================================================================
    // 4. MODULE 1: HYBRID PRODUCT RECOMMENDATIONS & DUAL CHARTS
    // =========================================================================
    async function updateRecommendations() {
        const opt = productSelect.options[productSelect.selectedIndex];
        if (!opt) return;
        const productId = productSelect.value;
        const basePrice = parseInt(opt.dataset.price);
        const divWeight = parseFloat(diversitySlider.value);

        document.getElementById("diversityVal").textContent =
            `${Math.round(divWeight * 100)}% (${divWeight <= 0.15 ? "Strict Match" : divWeight >= 0.35 ? "High Discovery" : "Balanced"})`;

        cartInput.value = basePrice;

        const res = await fetch(`${API_BASE}/api/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                product_id: productId,
                top_n: recTopN.value,
                diversity_weight: divWeight,
                category_filter: recCategoryFilter.value
            })
        });
        const data = await res.json();
        const recs = data.recommendations;

        recTableBody.innerHTML = recs.map(item => `
            <tr>
                <td><strong>${item.Product_Name}</strong><br><small style="color:#94a3b8">${item.Brand} • ${item.Match_Reason}</small></td>
                <td>${item.Category}</td>
                <td>₹${item.Base_Price.toLocaleString("en-IN")}</td>
                <td>⭐ ${item.Rating}</td>
                <td><span class="sim-pill">${item.Hybrid_Score}%</span></td>
            </tr>
        `).join("");

        renderRecCharts(recs);
        updatePrediction();
    }

    function renderRecCharts(recs) {
        const ctx1 = document.getElementById("simChart").getContext("2d");
        if (simChartInst) simChartInst.destroy();
        simChartInst = new Chart(ctx1, {
            type: "bar",
            data: {
                labels: recs.map(r => r.Product_Name.slice(0, 22) + "..."),
                datasets: [
                    {
                        label: "Hybrid Precision Score (%)",
                        data: recs.map(r => r.Hybrid_Score),
                        backgroundColor: "rgba(59, 130, 246, 0.82)",
                        borderRadius: 6
                    },
                    {
                        label: "Pure TF-IDF Cosine (%)",
                        data: recs.map(r => r.Semantic_Match),
                        backgroundColor: "rgba(168, 85, 247, 0.72)",
                        borderRadius: 6
                    }
                ]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
                scales: {
                    x: { max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } },
                    y: { ticks: { color: "#e2e8f0", font: { size: 10 } }, grid: { display: false } }
                }
            }
        });

        const topItem = recs[0];
        const ctx2 = document.getElementById("recRadarChart").getContext("2d");
        if (radarChartInst) radarChartInst.destroy();
        if (!topItem) return;

        radarChartInst = new Chart(ctx2, {
            type: "radar",
            data: {
                labels: ["Semantic Match", "Rating Quality", "Price Affinity", "Hybrid Score"],
                datasets: [{
                    label: topItem.Product_Name.slice(0, 26),
                    data: [topItem.Semantic_Match, topItem.Quality_Score, topItem.Price_Affinity, topItem.Hybrid_Score],
                    backgroundColor: "rgba(16, 185, 129, 0.25)",
                    borderColor: "#34d399",
                    pointBackgroundColor: "#34d399"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#e2e8f0" } } },
                scales: {
                    r: {
                        min: 0, max: 100,
                        ticks: { display: false },
                        grid: { color: "rgba(255,255,255,0.1)" },
                        pointLabels: { color: "#cbd5e1", font: { size: 11 } }
                    }
                }
            }
        });
    }

    // =========================================================================
    // 5. MODULE 2: CUSTOMER BEHAVIOR PREDICTION & DUAL CHARTS
    // =========================================================================
    async function updatePrediction() {
        document.getElementById("timeVal").textContent = `${timeInput.value} min`;
        document.getElementById("pagesVal").textContent = `${pagesInput.value} pages`;
        document.getElementById("pastVal").textContent = `${pastInput.value} orders`;
        document.getElementById("additionsVal").textContent = `${additionsInput.value} items`;

        const res = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                time_on_site: timeInput.value,
                pages_viewed: pagesInput.value,
                past_orders: pastInput.value,
                cart_additions: additionsInput.value,
                discount_pct: discountInput.value,
                cart_value: cartInput.value
            })
        });
        const data = await res.json();

        const prob = data.probability;
        const circle = document.getElementById("gaugeCircle");
        circle.style.strokeDashoffset = 326.7 - (prob / 100) * 326.7;

        document.getElementById("probDisplay").textContent = `${prob}%`;
        document.getElementById("abandonDisplay").textContent = `${data.abandon_risk}%`;
        const badge = document.getElementById("intentStatus");
        badge.textContent = data.status;
        document.getElementById("aiStrategyText").textContent = data.strategy;

        if (data.tier === "high") {
            circle.style.stroke = "#10b981";
            badge.style.background = "rgba(16, 185, 129, 0.2)";
            badge.style.color = "#34d399";
        } else if (data.tier === "moderate") {
            circle.style.stroke = "#f59e0b";
            badge.style.background = "rgba(245, 158, 11, 0.2)";
            badge.style.color = "#fbbf24";
        } else {
            circle.style.stroke = "#f43f5e";
            badge.style.background = "rgba(244, 63, 94, 0.2)";
            badge.style.color = "#fda4af";
        }

        renderBehaviorCharts(data);
    }

    function renderBehaviorCharts(data) {
        const ctx1 = document.getElementById("featureImpChart").getContext("2d");
        if (featImpChartInst) featImpChartInst.destroy();
        featImpChartInst = new Chart(ctx1, {
            type: "bar",
            data: {
                labels: data.feature_labels,
                datasets: [{
                    label: "Feature Importance (%)",
                    data: data.feature_importances,
                    backgroundColor: ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b"],
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } },
                    y: { ticks: { color: "#e2e8f0", font: { size: 11 } }, grid: { display: false } }
                }
            }
        });

        const ctx2 = document.getElementById("conversionDoughnutChart").getContext("2d");
        if (convDoughnutInst) convDoughnutInst.destroy();
        convDoughnutInst = new Chart(ctx2, {
            type: "doughnut",
            data: {
                labels: ["Conversion Probability (%)", "Abandonment Risk (%)"],
                datasets: [{
                    data: [data.probability, data.abandon_risk],
                    backgroundColor: ["#10b981", "#f43f5e"],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "66%",
                plugins: { legend: { position: "bottom", labels: { color: "#cbd5e1" } } }
            }
        });
    }

    // =========================================================================
    // 6. MODULE 3: DYNAMIC PRICING & ELASTICITY DUAL CHARTS
    // =========================================================================
    function syncPricingBaseInput() {
        const opt = pricingProductSelect.options[pricingProductSelect.selectedIndex];
        if (opt) {
            const base = parseInt(opt.dataset.price);
            compPriceInput.value = Math.round((base * 0.96) / 10) * 10;
        }
        updateDynamicPricing();
    }

    async function updateDynamicPricing() {
        const dVal = parseFloat(demandInput.value);
        const tag = dVal > 1.15 ? "Surge" : dVal < 0.9 ? "Low" : "Normal";
        document.getElementById("demandVal").textContent = `${dVal.toFixed(2)}x (${tag})`;

        const res = await fetch(`${API_BASE}/api/pricing`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                product_id: pricingProductSelect.value,
                demand_index: dVal,
                competitor_price: compPriceInput.value,
                time_slot: timeSlotInput.value
            })
        });
        const data = await res.json();

        document.getElementById("optimalPriceDisplay").textContent = `₹${data.optimal_price.toLocaleString("en-IN")}`;
        const sign = data.delta_pct >= 0 ? "+" : "";
        document.getElementById("priceDeltaDisplay").textContent = `${sign}${data.delta_pct}% vs Base (₹${data.base_price.toLocaleString("en-IN")})`;
        document.getElementById("projUnitsDisplay").textContent = `${data.projected_units} units/day`;
        document.getElementById("stockStatusDisplay").textContent = `${data.stock} in stock • ${data.stock_status}`;
        document.getElementById("dailyProfitDisplay").textContent = `₹${data.daily_profit.toLocaleString("en-IN")}`;
        const pSign = data.profit_lift_pct >= 0 ? "+" : "";
        document.getElementById("profitLiftDisplay").textContent = `${pSign}${data.profit_lift_pct}% Net Profit Lift`;

        renderPricingCharts(data);
    }

    function renderPricingCharts(data) {
        const ctx1 = document.getElementById("pricingChart").getContext("2d");
        if (pricingChartInst) pricingChartInst.destroy();
        pricingChartInst = new Chart(ctx1, {
            type: "line",
            data: {
                labels: data.curve_labels,
                datasets: [
                    {
                        label: "AI Dynamic Price (₹)",
                        data: data.curve_prices,
                        borderColor: "#a855f7",
                        backgroundColor: "rgba(168, 85, 247, 0.16)",
                        fill: true,
                        tension: 0.35,
                        yAxisID: "y"
                    },
                    {
                        label: "Projected Daily Profit (₹K)",
                        data: data.curve_profits,
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.12)",
                        tension: 0.35,
                        yAxisID: "y1"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                    y: { position: "left", ticks: { color: "#d8b4fe" }, grid: { color: "rgba(255,255,255,0.06)" } },
                    y1: { position: "right", ticks: { color: "#6ee7b7" }, grid: { display: false } }
                }
            }
        });

        const ctx2 = document.getElementById("profitCompareChart").getContext("2d");
        if (profitCompareInst) profitCompareInst.destroy();
        profitCompareInst = new Chart(ctx2, {
            type: "bar",
            data: {
                labels: ["Daily Gross Revenue (₹)", "Daily Net Profit (₹)"],
                datasets: [
                    {
                        label: "Static Base Pricing",
                        data: [data.static_revenue, data.static_profit],
                        backgroundColor: "rgba(148, 163, 184, 0.55)",
                        borderRadius: 6
                    },
                    {
                        label: "AI Dynamic Pricing",
                        data: [data.dynamic_revenue, data.daily_profit],
                        backgroundColor: "rgba(59, 130, 246, 0.85)",
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: "#e2e8f0" }, grid: { display: false } },
                    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } }
                }
            }
        });
    }

    // =========================================================================
    // 7. DUAL-SYNCED OMNI-QUERY AI CHATBOT (MODULE 4 + CORNER FAVICON)
    // =========================================================================
    const chatWindow = document.getElementById("chatWindow");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const clearChatBtn = document.getElementById("clearChatBtn");

    const cornerBotBtn = document.getElementById("cornerBotBtn");
    const cornerBotDrawer = document.getElementById("cornerBotDrawer");
    const closeDrawerBtn = document.getElementById("closeDrawerBtn");
    const drawerChatWindow = document.getElementById("drawerChatWindow");
    const drawerChatForm = document.getElementById("drawerChatForm");
    const drawerChatInput = document.getElementById("drawerChatInput");

    cornerBotBtn.addEventListener("click", () => cornerBotDrawer.classList.toggle("open"));
    closeDrawerBtn.addEventListener("click", () => cornerBotDrawer.classList.remove("open"));

    function appendToBothChats(html, sender) {
        [chatWindow, drawerChatWindow].forEach(win => {
            const div = document.createElement("div");
            div.className = `chat-msg ${sender}`;
            div.innerHTML = html;
            win.appendChild(div);
            win.scrollTop = win.scrollHeight;
        });
    }

    async function handleChatQuery(text) {
        if (!text.trim()) return;
        appendToBothChats(text, "user");

        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        appendToBothChats(data.reply, "bot");
    }

    chatForm.addEventListener("submit", e => {
        e.preventDefault();
        handleChatQuery(chatInput.value);
        chatInput.value = "";
    });

    drawerChatForm.addEventListener("submit", e => {
        e.preventDefault();
        handleChatQuery(drawerChatInput.value);
        drawerChatInput.value = "";
    });

    document.querySelectorAll(".chip-btn").forEach(btn => {
        btn.addEventListener("click", () => handleChatQuery(btn.dataset.q));
    });

    clearChatBtn.addEventListener("click", () => {
        const resetMsg = `<div class="chat-msg bot">Chat reset! Ask me about any product, budget, comparison, coupon, or ML model metric.</div>`;
        chatWindow.innerHTML = resetMsg;
        drawerChatWindow.innerHTML = resetMsg;
    });

    // =========================================================================
    // 8. MODULE 5: ADMIN CUSTOM PRODUCT STUDIO (ADD / DELETE + LIVE RETRAIN)
    // =========================================================================
    const adminAddForm = document.getElementById("adminAddForm");
    const autoTagsBtn = document.getElementById("autoTagsBtn");
    const adminTableBody = document.getElementById("adminTableBody");
    const adminSearchInput = document.getElementById("adminSearchInput");
    const adminToast = document.getElementById("adminToast");

    autoTagsBtn.addEventListener("click", () => {
        const name = document.getElementById("admName").value;
        const cat = document.getElementById("admCategory").value;
        const brand = document.getElementById("admBrand").value;
        const raw = `${name} ${cat} ${brand} wireless smart premium fast pro`.toLowerCase();
        const uniqueWords = [...new Set(raw.match(/[a-z0-9]+/g) || [])];
        document.getElementById("admTags").value = uniqueWords.join(" ");
    });

    function updateAdminKPIs(stats) {
        document.getElementById("adminTotalSkus").textContent = stats.total_products;
        document.getElementById("wtProducts").textContent = stats.total_products;
        document.getElementById("adminTotalStock").textContent = stats.total_stock.toLocaleString("en-IN");
        document.getElementById("adminCatalogVal").textContent = `₹${stats.catalog_value.toLocaleString("en-IN")}`;
        document.getElementById("adminAvgMargin").textContent = `${stats.avg_margin}%`;
    }

    function renderAdminChart(stats) {
        const ctx = document.getElementById("adminCategoryChart").getContext("2d");
        if (adminCatChartInst) adminCatChartInst.destroy();
        adminCatChartInst = new Chart(ctx, {
            type: "bar",
            data: {
                labels: stats.category_labels,
                datasets: [
                    {
                        label: "SKU Count",
                        data: stats.category_counts,
                        backgroundColor: "rgba(59, 130, 246, 0.8)",
                        borderRadius: 6
                    },
                    {
                        label: "Total Stock Units",
                        data: stats.category_stocks,
                        backgroundColor: "rgba(16, 185, 129, 0.75)",
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: "#e2e8f0", font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } }
                }
            }
        });
    }

    function renderAdminTable(products) {
        const q = adminSearchInput.value.toLowerCase();
        const filtered = products.filter(p =>
            p.Product_Name.toLowerCase().includes(q) ||
            p.Category.toLowerCase().includes(q) ||
            p.Brand.toLowerCase().includes(q)
        );
        adminTableBody.innerHTML = filtered.map(p => `
            <tr>
                <td>#${p.Product_ID}</td>
                <td><strong>${p.Product_Name}</strong><br><small style="color:#94a3b8">${p.Brand}</small></td>
                <td>${p.Category}</td>
                <td>₹${p.Base_Price.toLocaleString("en-IN")}</td>
                <td>${p.Stock}</td>
                <td><button class="del-btn" data-id="${p.Product_ID}" title="Delete SKU"><i class="fa-solid fa-trash"></i></button></td>
            </tr>
        `).join("");

        adminTableBody.querySelectorAll(".del-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const pid = btn.dataset.id;
                const res = await fetch(`${API_BASE}/api/admin/delete-product/${pid}`, { method: "DELETE" });
                const data = await res.json();
                if (data.success) {
                    allProducts = data.products;
                    syncDropdowns(allProducts);
                    renderAdminTable(allProducts);
                    renderAdminChart(data.stats);
                    updateAdminKPIs(data.stats);
                    updateRecommendations();
                    adminToast.textContent = `✅ ${data.message}`;
                } else {
                    adminToast.textContent = `⚠️ ${data.error}`;
                }
            });
        });
    }

    adminSearchInput.addEventListener("input", () => renderAdminTable(allProducts));

    adminAddForm.addEventListener("submit", async e => {
        e.preventDefault();
        const payload = {
            name: document.getElementById("admName").value,
            category: document.getElementById("admCategory").value,
            brand: document.getElementById("admBrand").value,
            base_price: document.getElementById("admBasePrice").value,
            cost_price: document.getElementById("admCostPrice").value,
            stock: document.getElementById("admStock").value,
            rating: document.getElementById("admRating").value,
            warranty: document.getElementById("admWarranty").value,
            tags: document.getElementById("admTags").value
        };

        const res = await fetch(`${API_BASE}/api/admin/add-product`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            allProducts = data.products;
            syncDropdowns(allProducts);
            renderAdminTable(allProducts);
            renderAdminChart(data.stats);
            updateAdminKPIs(data.stats);
            updateRecommendations();
            adminToast.textContent = `✅ Added "${data.product.Product_Name}" (#${data.product.Product_ID}) to CSV & re-indexed all AI models!`;
            adminAddForm.reset();
        } else {
            adminToast.textContent = `⚠️ ${data.error}`;
        }
    });

    // =========================================================================
    // 9. EVENT BINDINGS & INITIAL BOOT
    // =========================================================================
    [productSelect, recCategoryFilter, recTopN, diversitySlider].forEach(el =>
        el.addEventListener("change", updateRecommendations)
    );
    diversitySlider.addEventListener("input", updateRecommendations);

    [timeInput, pagesInput, pastInput, additionsInput, discountInput, cartInput].forEach(el =>
        el.addEventListener("input", updatePrediction)
    );

    pricingProductSelect.addEventListener("change", syncPricingBaseInput);
    [demandInput, compPriceInput, timeSlotInput].forEach(el =>
        el.addEventListener("input", updateDynamicPricing)
    );

    await loadInitialState();
    updateRecommendations();
    syncPricingBaseInput();
});