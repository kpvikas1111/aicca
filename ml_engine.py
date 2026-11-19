import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class SmartCartAIEngine:
    def __init__(self, csv_path="data/products.csv", model_dir=("/tmp" if __import__("os").environ.get("VERCEL") else "models")):
        self.csv_path = csv_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.products_df = pd.read_csv(self.csv_path)
        if "Warranty" not in self.products_df.columns:
            self.products_df["Warranty"] = "1 Year Standard"

        self._init_hybrid_recommender()
        self._init_behavior_classifier()

    # =========================================================================
    # MODULE 1: HYBRID PRODUCT RECOMMENDATION ENGINE (TF-IDF + QUALITY + PRICE)
    # =========================================================================
    def _init_hybrid_recommender(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = (
            self.products_df["Tags"].fillna("") + " " +
            self.products_df["Category"].fillna("") + " " +
            self.products_df["Brand"].fillna("") + " " +
            self.products_df["Product_Name"].fillna("")
        )
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Persist TF-IDF artifact to disk
        joblib.dump(
            {"vectorizer": self.tfidf, "matrix": self.tfidf_matrix},
            os.path.join(self.model_dir, "tfidf_recommender.joblib")
        )

    def recommend_products(self, product_id, top_n=6, diversity_weight=0.20, category_filter="ALL"):
        matches = self.products_df[self.products_df["Product_ID"] == int(product_id)]
        idx = matches.index[0] if not matches.empty else 0
        target_row = self.products_df.iloc[idx]
        target_price = float(target_row["Base_Price"])

        scored_items = []
        for i, row in self.products_df.iterrows():
            if i == idx:
                continue
            if category_filter and category_filter != "ALL" and row["Category"] != category_filter:
                continue

            # 1. Semantic TF-IDF Cosine Similarity (0 to 1)
            semantic_sim = float(self.content_sim_matrix[idx][i])

            # 2. Normalized Rating Quality Boost (0 to 1)
            quality_score = np.clip((float(row["Rating"]) - 4.0) / 1.0, 0.0, 1.0)

            # 3. Price-Tier Affinity (Gaussian decay on relative price difference)
            price_diff_ratio = abs(float(row["Base_Price"]) - target_price) / max(target_price, 1.0)
            price_affinity = float(np.exp(-1.4 * price_diff_ratio))

            # 4. Popularity Social Proof Boost (normalized review count)
            pop_boost = min(float(row["Reviews_Count"]) / 6500.0, 1.0)

            w_sem = 1.0 - diversity_weight
            w_qual = diversity_weight * 0.45
            w_price = diversity_weight * 0.35
            w_pop = diversity_weight * 0.20

            hybrid_score = (
                (w_sem * semantic_sim)
                + (w_qual * quality_score)
                + (w_price * price_affinity)
                + (w_pop * pop_boost)
            )

            scored_items.append({
                "Product_ID": int(row["Product_ID"]),
                "Product_Name": row["Product_Name"],
                "Brand": row["Brand"],
                "Category": row["Category"],
                "Base_Price": int(row["Base_Price"]),
                "Stock": int(row["Stock"]),
                "Rating": float(row["Rating"]),
                "Reviews_Count": int(row["Reviews_Count"]),
                "Warranty": str(row["Warranty"]),
                "Semantic_Match": round(semantic_sim * 100, 1),
                "Quality_Score": round(quality_score * 100, 1),
                "Price_Affinity": round(price_affinity * 100, 1),
                "Hybrid_Score": round(min(hybrid_score * 100, 99.6), 1),
                "Match_Reason": (
                    "Same Category & Feature Match"
                    if row["Category"] == target_row["Category"]
                    else "Cross-Sell Ecosystem Companion"
                )
            })

        scored_items.sort(key=lambda x: x["Hybrid_Score"], reverse=True)
        return scored_items[:int(top_n)]

    # =========================================================================
    # MODULE 2: CUSTOMER BEHAVIOR PREDICTION (GRADIENT BOOSTING CLASSIFIER)
    # =========================================================================
    def _init_behavior_classifier(self):
        np.random.seed(42)
        n = 1500

        time_on_site = np.random.uniform(0.5, 25.0, n)
        pages_viewed = np.random.randint(1, 25, n)
        past_orders = np.random.randint(0, 12, n)
        discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], n)
        cart_value = np.random.uniform(1000, 150000, n)
        cart_additions = np.random.randint(1, 8, n)

        logit = (
            -3.8
            + 0.21 * time_on_site
            + 0.15 * pages_viewed
            + 0.38 * past_orders
            + 0.095 * discount_pct
            + 0.28 * cart_additions
            - 0.000009 * cart_value
            + 0.012 * (time_on_site * discount_pct)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        purchased = (np.random.rand(n) < prob).astype(int)

        self.feature_cols = [
            "time_on_site",
            "pages_viewed",
            "past_orders",
            "discount_pct",
            "cart_value",
            "cart_additions"
        ]
        self.feature_labels = [
            "Time on Site",
            "Pages Viewed",
            "Past Orders",
            "Coupon Discount",
            "Cart Value",
            "Cart Additions"
        ]

        X = pd.DataFrame({
            "time_on_site": time_on_site,
            "pages_viewed": pages_viewed,
            "past_orders": past_orders,
            "discount_pct": discount_pct,
            "cart_value": cart_value,
            "cart_additions": cart_additions
        })

        X_train, X_test, y_train, y_test = train_test_split(X, purchased, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.gb_clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
        self.gb_clf.fit(X_train_scaled, y_train)

        preds = self.gb_clf.predict(X_test_scaled)
        probs = self.gb_clf.predict_proba(X_test_scaled)[:, 1]

        importances = [round(float(imp) * 100, 1) for imp in self.gb_clf.feature_importances_]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, preds) * 100, 1),
            "precision": round(precision_score(y_test, preds) * 100, 1),
            "recall": round(recall_score(y_test, preds) * 100, 1),
            "f1_score": round(f1_score(y_test, preds) * 100, 1),
            "roc_auc": round(roc_auc_score(y_test, probs), 3),
            "training_samples": n,
            "feature_labels": self.feature_labels,
            "feature_importances": importances
        }

        # Persist Gradient Boosting Classifier + Scaler to disk
        joblib.dump(
            {"model": self.gb_clf, "scaler": self.scaler, "metrics": self.metrics},
            os.path.join(self.model_dir, "behavior_gb_model.joblib")
        )

    def predict_conversion(self, time_on_site, pages_viewed, past_orders, discount_pct, cart_value, cart_additions=2):
        raw_df = pd.DataFrame([{
            "time_on_site": float(time_on_site),
            "pages_viewed": int(pages_viewed),
            "past_orders": int(past_orders),
            "discount_pct": int(discount_pct),
            "cart_value": float(cart_value),
            "cart_additions": int(cart_additions)
        }])
        scaled_X = self.scaler.transform(raw_df)
        prob = round(float(self.gb_clf.predict_proba(scaled_X)[0][1]) * 100, 1)
        abandon_risk = round(100.0 - prob, 1)

        if prob >= 72.0:
            status = "High Conversion Intent"
            tier = "high"
            strategy = "Deploy 1-Click Express Checkout & high-margin accessory bundle. Suppress coupon pop-up to protect full profit margin."
        elif prob >= 42.0:
            status = "Moderate Intent (Hesitant Buyer)"
            tier = "moderate"
            strategy = "Trigger Exit-Intent Retention Modal with a 10% limited-time coupon & free next-day delivery badge."
        else:
            status = "Low Intent (Early Window Shopper)"
            tier = "low"
            strategy = "Serve personalized Top-6 Hybrid Recommendations, verified buyer reviews, and No-Cost EMI breakdown."

        return {
            "probability": prob,
            "abandon_risk": abandon_risk,
            "status": status,
            "tier": tier,
            "strategy": strategy,
            "feature_labels": self.metrics["feature_labels"],
            "feature_importances": self.metrics["feature_importances"]
        }

    # =========================================================================
    # MODULE 3: DYNAMIC PRICING & ELASTICITY OPTIMIZATION ENGINE
    # =========================================================================
    def calculate_dynamic_price(self, product_id, demand_index, competitor_price, time_slot="peak"):
        matches = self.products_df[self.products_df["Product_ID"] == int(product_id)]
        item = matches.iloc[0] if not matches.empty else self.products_df.iloc[0]

        base_price = float(item["Base_Price"])
        cost_price = float(item["Cost_Price"])
        stock = int(item["Stock"])
        demand = float(demand_index)
        comp_price = float(competitor_price)

        slot_mult = {"offpeak": 0.97, "normal": 1.00, "peak": 1.03, "flash": 1.06}.get(time_slot, 1.0)

        if stock <= 10:
            scarcity_mult = 1.07
            stock_status = "Low Stock Scarcity (+7%)"
        elif stock >= 35:
            scarcity_mult = 0.94
            stock_status = "High Inventory Clearance (-6%)"
        else:
            scarcity_mult = 1.00
            stock_status = "Balanced Inventory (Neutral)"

        demand_mult = 1.0 + (demand - 1.0) * 0.15
        model_price = base_price * scarcity_mult * demand_mult * slot_mult

        anchored_price = (0.65 * model_price) + (0.35 * comp_price)
        min_safe_price = max(cost_price * 1.12, base_price * 0.85)
        max_safe_price = base_price * 1.14
        optimal_price = int(round(np.clip(anchored_price, min_safe_price, max_safe_price), -1))

        elasticity = -1.4
        base_daily_units = 24.0 * demand
        price_ratio = optimal_price / base_price
        projected_units = max(1.0, base_daily_units * (price_ratio ** elasticity))

        static_revenue = base_price * base_daily_units
        dynamic_revenue = optimal_price * projected_units
        static_profit = (base_price - cost_price) * base_daily_units
        dynamic_profit = (optimal_price - cost_price) * projected_units
        profit_lift_pct = round(((dynamic_profit - static_profit) / max(static_profit, 1.0)) * 100, 1)
        delta_pct = round(((optimal_price - base_price) / base_price) * 100, 1)

        demand_grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        curve_prices = []
        curve_profits = []
        for d in demand_grid:
            p_raw = base_price * scarcity_mult * (1.0 + (d - 1.0) * 0.15) * slot_mult
            p_final = int(round(np.clip(0.65 * p_raw + 0.35 * comp_price, min_safe_price, max_safe_price), -1))
            u_est = max(1.0, (24.0 * d) * ((p_final / base_price) ** elasticity))
            prof_k = round(((p_final - cost_price) * u_est) / 1000.0, 1)
            curve_prices.append(p_final)
            curve_profits.append(prof_k)

        return {
            "product_name": item["Product_Name"],
            "base_price": int(base_price),
            "cost_price": int(cost_price),
            "stock": stock,
            "stock_status": stock_status,
            "optimal_price": optimal_price,
            "delta_pct": delta_pct,
            "projected_units": round(projected_units, 1),
            "static_revenue": int(round(static_revenue)),
            "dynamic_revenue": int(round(dynamic_revenue)),
            "static_profit": int(round(static_profit)),
            "daily_profit": int(round(dynamic_profit)),
            "profit_lift_pct": profit_lift_pct,
            "curve_labels": [f"{int(d * 100)}%" for d in demand_grid],
            "curve_prices": curve_prices,
            "curve_profits": curve_profits
        }

    # =========================================================================
    # MODULE 4: OMNI-QUERY SEMANTIC NLP SHOPPING ASSISTANT CHATBOT
    # =========================================================================
    def _find_mentioned_products(self, msg):
        matched = []
        tokens = set(re.findall(r"[a-z0-9]+", msg.lower()))
        for _, row in self.products_df.iterrows():
            name_tokens = set(re.findall(r"[a-z0-9]+", row["Product_Name"].lower()))
            brand_tok = row["Brand"].lower()
            id_str = str(row["Product_ID"])
            overlap = len(tokens.intersection(name_tokens - {"pro", "gen", "ultra", "5g", "2nd", "3rd"}))
            if id_str in tokens or overlap >= 2 or (brand_tok in tokens and overlap >= 1):
                matched.append((overlap + (2 if brand_tok in tokens else 0), row))
        matched.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matched]

    def chatbot_reply(self, message):
        raw_msg = message.strip()
        msg = raw_msg.lower()

        if not msg:
            return "Please type a product question, budget, comparison, or store policy query!"

        # 1. Greetings & Capabilities
        if any(g in msg for g in ["hi", "hello", "hey", "namaste", "good morning", "good evening"]) and len(msg.split()) <= 5:
            return (
                f"Hello! I am <strong>SmartCart AI Copilot</strong>, connected live to our <strong>{len(self.products_df)}-product</strong> catalog & ML engines.<br><br>"
                "I can answer <strong>any query</strong>:<br>"
                "• 🛍️ <strong>Product Search & Budget:</strong> <em>\"Best ANC headphones under 28k\"</em> or <em>\"Gaming laptops\"</em><br>"
                "• ⚖️ <strong>Side-by-Side Comparison:</strong> <em>\"Compare Sony WH-1000XM5 vs Bose\"</em> or <em>\"iPhone 16 Pro vs Galaxy S24 Ultra\"</em><br>"
                "• 📊 <strong>Live Catalog & AI Stats:</strong> <em>\"Catalog summary\"</em>, <em>\"Model accuracy\"</em>, <em>\"Low stock items\"</em><br>"
                "• 📦 <strong>Policies & EMI:</strong> <em>\"Shipping & return policy\"</em>, <em>\"No-Cost EMI options\"</em>, <em>\"Active coupons\"</em>"
            )

        if any(t in msg for t in ["thank you", "thanks", "awesome", "great job", "nice"]):
            return "You're very welcome! Let me know if you want to compare products, check EMI/discounts, or inspect live AI analytics."

        # 2. Store Policies, Shipping, Returns, EMI, Warranty, Discounts & Support
        if any(w in msg for w in ["shipping", "delivery", "deliver", "how fast", "dispatch"]):
            return (
                "🚚 <strong>SmartCart Express Shipping Policy:</strong><br>"
                "• <strong>Free Same-Day / Next-Day Delivery</strong> in Bengaluru, Mumbai, Delhi NCR, Hyderabad & Chennai on orders above ₹999.<br>"
                "• <strong>Pan-India Insured Courier:</strong> 2–4 business days with real-time GPS tracking.<br>"
                "• Every package includes tamper-evident packaging and open-box verification."
            )

        if any(w in msg for w in ["return", "refund", "replace", "exchange", "cancel"]):
            return (
                "🔄 <strong>7-Day Hassle-Free Return & Replacement Policy:</strong><br>"
                "• Instant 7-day replacement for manufacturing defects or transit damage.<br>"
                "• 100% original-mode refund processed within <strong>24–48 hours</strong> of doorstep pickup.<br>"
                "• Orders can be cancelled with zero fee anytime before dispatch."
            )

        if any(w in msg for w in ["emi", "installment", "payment", "upi", "credit card", "cod", "cash on delivery"]):
            return (
                "💳 <strong>Payments & No-Cost EMI Options:</strong><br>"
                "• <strong>No-Cost EMI (3, 6, 9, 12 Months):</strong> Available on HDFC, ICICI, SBI, Axis & Bajaj Finserv cards for orders above ₹8,000.<br>"
                "• <strong>Instant Methods:</strong> UPI (GPay/PhonePe/Paytm), NetBanking, Credit/Debit Cards & Cash on Delivery up to ₹50,000."
            )

        if any(w in msg for w in ["coupon", "promo", "offer", "deal", "voucher"]):
            return (
                "🎟️ <strong>Active SmartCart AI Promo Codes:</strong><br>"
                "• <code>SMART10</code> — Flat <strong>10% Instant Discount</strong> on Audio & Wearables.<br>"
                "• <code>TECHPRO15</code> — Up to <strong>₹7,500 Off</strong> on Laptops & Flagship Smartphones.<br>"
                "• <code>WELCOME500</code> — Flat ₹500 off on your first order above ₹4,999."
            )

        if any(w in msg for w in ["track", "order status", "where is my order", "customer care", "support", "contact"]):
            return (
                "🎧 <strong>24/7 AI & Live Concierge Support:</strong><br>"
                "• Share your 8-digit Order ID (e.g., <code>#SC-849201</code>) to check live courier telemetry.<br>"
                "• Priority Desk: <strong>support@smartcart-ai.in</strong> | Toll-Free: <strong>1800-419-2026</strong>."
            )

        # 3. AI/ML Architecture & Live Metrics Queries
        if any(w in msg for w in ["accuracy", "precision", "recall", "f1", "roc", "auc", "model", "algorithm", "gradient boosting", "tf-idf", "how do you work"]):
            m = self.metrics
            return (
                "🧠 <strong>SmartCart AI Live Model Telemetry:</strong><br>"
                f"• <strong>Conversion Classifier:</strong> Scikit-Learn <code>GradientBoostingClassifier</code> trained on {m['training_samples']} behavioral sessions.<br>"
                f"• <strong>Live Benchmark Scores:</strong> Accuracy: <strong>{m['accuracy']}%</strong> | Precision: <strong>{m['precision']}%</strong> | Recall: <strong>{m['recall']}%</strong> | F1-Score: <strong>{m['f1_score']}%</strong> | ROC-AUC: <strong>{m['roc_auc']}</strong>.<br>"
                "• <strong>Recommender & Chatbot:</strong> Bi-gram <code>TfidfVectorizer</code> + Cosine Similarity blended with Quality & Gaussian Price Affinity.<br>"
                "• <strong>Dynamic Pricing:</strong> Log-linear elasticity (<code>e = -1.4</code>) with inventory scarcity & competitor anchoring."
            )

        # 4. Catalog Overview, Categories, Inventory & Extremes
        if any(w in msg for w in ["how many product", "catalog summary", "overview", "all categories", "what categories", "total stock", "inventory"]):
            cats = self.products_df["Category"].value_counts().to_dict()
            cat_str = ", ".join([f"<strong>{k}</strong> ({v})" for k, v in cats.items()])
            total_stock = int(self.products_df["Stock"].sum())
            avg_price = int(self.products_df["Base_Price"].mean())
            return (
                f"📊 <strong>Live Catalog Intelligence ({len(self.products_df)} Products):</strong><br>"
                f"• <strong>Categories:</strong> {cat_str}<br>"
                f"• <strong>Total Warehouse Stock:</strong> {total_stock:,} units across {self.products_df['Brand'].nunique()} global brands.<br>"
                f"• <strong>Average Catalog Price:</strong> ₹{avg_price:,} | <strong>Avg Rating:</strong> ⭐ {self.products_df['Rating'].mean():.2f}/5.0."
            )

        if any(w in msg for w in ["low stock", "out of stock", "scarce", "running out"]):
            low_df = self.products_df[self.products_df["Stock"] <= 10].sort_values(by="Stock")
            items = "<br>".join([
                f"⚠️ <strong>{r['Product_Name']}</strong> — Only <strong>{r['Stock']} units</strong> left (₹{r['Base_Price']:,})"
                for _, r in low_df.iterrows()
            ])
            return f"🔥 <strong>High-Demand Low Stock Items (≤ 10 Units):</strong><br><br>{items}"

        # 5. Side-by-Side Product / Brand Comparison
        if any(w in msg for w in ["compare", " vs ", "versus", "difference between", "better between"]):
            mentioned = self._find_mentioned_products(msg)
            if len(mentioned) >= 2:
                p1, p2 = mentioned[0], mentioned[1]
                winner = p1 if (p1["Rating"] / max(p1["Base_Price"], 1)) >= (p2["Rating"] / max(p2["Base_Price"], 1)) else p2
                return (
                    f"⚖️ <strong>AI Head-to-Head Comparison:</strong><br><br>"
                    f"1️⃣ <strong>{p1['Product_Name']}</strong> ({p1['Brand']})<br>"
                    f"&nbsp;&nbsp;&nbsp;• Price: <strong>₹{p1['Base_Price']:,}</strong> | Rating: ⭐ {p1['Rating']} ({p1['Reviews_Count']:,} reviews)<br>"
                    f"&nbsp;&nbsp;&nbsp;• Stock: {p1['Stock']} units | Warranty: {p1['Warranty']}<br><br>"
                    f"2️⃣ <strong>{p2['Product_Name']}</strong> ({p2['Brand']})<br>"
                    f"&nbsp;&nbsp;&nbsp;• Price: <strong>₹{p2['Base_Price']:,}</strong> | Rating: ⭐ {p2['Rating']} ({p2['Reviews_Count']:,} reviews)<br>"
                    f"&nbsp;&nbsp;&nbsp;• Stock: {p2['Stock']} units | Warranty: {p2['Warranty']}<br><br>"
                    f"🏆 <strong>AI Verdict:</strong> <strong>{winner['Product_Name']}</strong> offers the stronger rating-to-price value score!"
                )

        # 6. Extract Budget Constraints (e.g., 25000, 30k, 1.2 lakh)
        budget = None
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", msg)
        lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", msg)
        num_match = re.search(r"(\d{4,6})", msg.replace(",", ""))

        if k_match:
            budget = int(float(k_match.group(1)) * 1000)
        elif lakh_match:
            budget = int(float(lakh_match.group(1)) * 100000)
        elif num_match:
            budget = int(num_match.group(1))

        # 7. Category & Use-Case Intent Filtering
        df = self.products_df.copy()
        category_filter = None
        if any(w in msg for w in ["laptop", "macbook", "coding", "ultrabook", "thinkpad", "spectre", "xps", "workstation"]):
            category_filter = "Laptops"
        elif any(w in msg for w in ["phone", "smartphone", "iphone", "galaxy", "oneplus", "pixel", "nothing", "mobile", "5g"]):
            category_filter = "Smartphones"
        elif any(w in msg for w in ["headphone", "earbud", "audio", "music", "speaker", "anc", "noise", "bose", "sennheiser", "airpods"]):
            category_filter = "Audio"
        elif any(w in msg for w in ["watch", "smartwatch", "fitness", "running", "garmin", "health", "whoop", "marathon"]):
            category_filter = "Wearables"
        elif any(w in msg for w in ["mouse", "keyboard", "charger", "power bank", "accessory", "accessories", "dock", "ssd", "hub"]):
            category_filter = "Accessories"
        elif any(w in msg for w in ["smart home", "vacuum", "dyson", "purifier", "alexa", "echo", "hue", "light", "roborock", "home"]):
            category_filter = "Smart Home"
        elif any(w in msg for w in ["gaming", "camera", "ps5", "playstation", "gopro", "drone", "dji", "monitor", "vlog", "razer"]):
            category_filter = "Gaming & Cameras"

        if category_filter:
            df = df[df["Category"] == category_filter]

        if budget:
            df = df[df["Base_Price"] <= budget]
            if df.empty:
                fallback = self.products_df[self.products_df["Category"] == category_filter] if category_filter else self.products_df
                cheapest = fallback.sort_values(by="Base_Price").iloc[0]
                return (
                    f"No matching items found within <strong>₹{budget:,}</strong>. "
                    f"The closest option in our catalog is <strong>{cheapest['Product_Name']}</strong> at <strong>₹{cheapest['Base_Price']:,}</strong> (⭐ {cheapest['Rating']})."
                )

        # 8. Rank Candidates via TF-IDF Semantic Similarity + Rating
        query_vec = self.tfidf.transform([msg])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        df = df.copy()
        df["query_sim"] = [sim_scores[idx] for idx in df.index]

        if any(w in msg for w in ["cheap", "cheapest", "lowest", "affordable"]):
            ranked = df.sort_values(by=["Base_Price", "Rating"], ascending=[True, False]).head(4)
        elif any(w in msg for w in ["expensive", "premium", "flagship", "costliest"]):
            ranked = df.sort_values(by=["Base_Price", "Rating"], ascending=[False, False]).head(4)
        elif any(w in msg for w in ["top rated", "best rated", "highest rated", "popular"]):
            ranked = df.sort_values(by=["Rating", "Reviews_Count"], ascending=[False, False]).head(4)
        else:
            ranked = df.sort_values(by=["query_sim", "Rating", "Reviews_Count"], ascending=[False, False, False]).head(4)

        header = f"Here are the top AI-matched <strong>{category_filter or 'Catalog'}</strong> recommendations"
        if budget:
            header += f" under <strong>₹{budget:,}</strong>"
        header += ":<br><br>"

        cards_html = "<br>".join([
            f"🔹 <strong>{r['Product_Name']}</strong> ({r['Brand']} • {r['Category']})<br>"
            f"&nbsp;&nbsp;&nbsp;Price: <strong>₹{r['Base_Price']:,}</strong> • Rating: ⭐ {r['Rating']} ({r['Reviews_Count']:,}) • Stock: {r['Stock']} units"
            for _, r in ranked.iterrows()
        ])
        return header + cards_html

    # =========================================================================
    # MODULE 5: ADMIN CUSTOM PRODUCT STUDIO (LIVE CSV + ML RETRAINING)
    # =========================================================================
    def get_catalog_stats(self):
        df = self.products_df
        cat_counts = df["Category"].value_counts().to_dict()
        cat_stock = df.groupby("Category")["Stock"].sum().to_dict()
        margins = ((df["Base_Price"] - df["Cost_Price"]) / df["Base_Price"]) * 100
        return {
            "total_products": int(len(df)),
            "total_stock": int(df["Stock"].sum()),
            "catalog_value": int((df["Base_Price"] * df["Stock"]).sum()),
            "avg_margin": round(float(margins.mean()), 1),
            "category_labels": list(cat_counts.keys()),
            "category_counts": [int(v) for v in cat_counts.values()],
            "category_stocks": [int(cat_stock[k]) for k in cat_counts.keys()]
        }

    def add_custom_product(self, name, category, brand, base_price, cost_price, stock, rating, tags, warranty="1 Year Standard"):
        next_id = int(self.products_df["Product_ID"].max()) + 1 if not self.products_df.empty else 101
        new_row = {
            "Product_ID": next_id,
            "Product_Name": str(name).strip(),
            "Category": str(category).strip(),
            "Brand": str(brand).strip(),
            "Base_Price": int(float(base_price)),
            "Cost_Price": int(float(cost_price)),
            "Stock": int(stock),
            "Rating": round(float(np.clip(float(rating), 1.0, 5.0)), 1),
            "Reviews_Count": 125,
            "Tags": str(tags).strip().lower(),
            "Warranty": str(warranty).strip() or "1 Year Standard"
        }
        self.products_df = pd.concat([self.products_df, pd.DataFrame([new_row])], ignore_index=True)
        self.products_df.to_csv(self.csv_path, index=False)

        # Immediately re-fit TF-IDF Recommender & Chatbot index
        self._init_hybrid_recommender()
        return new_row

    def delete_product(self, product_id):
        if len(self.products_df) <= 5:
            return False, "Minimum 5 products must remain in the catalog."
        pid = int(product_id)
        if pid not in self.products_df["Product_ID"].values:
            return False, "Product ID not found."
        self.products_df = self.products_df[self.products_df["Product_ID"] != pid].reset_index(drop=True)
        self.products_df.to_csv(self.csv_path, index=False)
        self._init_hybrid_recommender()
        return True, f"Product #{pid} deleted and AI models re-indexed."