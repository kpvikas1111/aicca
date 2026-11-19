import threading
import webbrowser
from flask import Flask, render_template, request, jsonify
from ml_engine import SmartCartAIEngine

app = Flask(__name__)
engine = SmartCartAIEngine("data/products.csv")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.route("/")
def index():
    products = engine.products_df.to_dict(orient="records")
    return render_template(
        "index.html",
        products=products,
        metrics=engine.metrics,
        stats=engine.get_catalog_stats()
    )


@app.route("/api/init", methods=["GET"])
def api_init():
    return jsonify({
        "products": engine.products_df.to_dict(orient="records"),
        "metrics": engine.metrics,
        "stats": engine.get_catalog_stats()
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    recs = engine.recommend_products(
        product_id=data.get("product_id", 101),
        top_n=int(data.get("top_n", 6)),
        diversity_weight=float(data.get("diversity_weight", 0.20)),
        category_filter=data.get("category_filter", "ALL")
    )
    return jsonify({"recommendations": recs})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    result = engine.predict_conversion(
        time_on_site=data.get("time_on_site", 9.5),
        pages_viewed=data.get("pages_viewed", 7),
        past_orders=data.get("past_orders", 2),
        discount_pct=data.get("discount_pct", 10),
        cart_value=data.get("cart_value", 29990),
        cart_additions=data.get("cart_additions", 2)
    )
    return jsonify(result)


@app.route("/api/pricing", methods=["POST"])
def api_pricing():
    data = request.get_json()
    result = engine.calculate_dynamic_price(
        product_id=data.get("product_id", 101),
        demand_index=data.get("demand_index", 1.25),
        competitor_price=data.get("competitor_price", 28990),
        time_slot=data.get("time_slot", "peak")
    )
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    reply = engine.chatbot_reply(data.get("message", ""))
    return jsonify({"reply": reply})


@app.route("/api/admin/add-product", methods=["POST"])
def api_admin_add_product():
    data = request.get_json()
    if not data.get("name") or not data.get("brand") or not data.get("base_price"):
        return jsonify({"success": False, "error": "Product Name, Brand, and Base Price are required."}), 400

    new_prod = engine.add_custom_product(
        name=data.get("name"),
        category=data.get("category", "Accessories"),
        brand=data.get("brand"),
        base_price=data.get("base_price", 9999),
        cost_price=data.get("cost_price", 6500),
        stock=data.get("stock", 20),
        rating=data.get("rating", 4.7),
        tags=data.get("tags", ""),
        warranty=data.get("warranty", "1 Year Standard")
    )
    return jsonify({
        "success": True,
        "product": new_prod,
        "products": engine.products_df.to_dict(orient="records"),
        "stats": engine.get_catalog_stats()
    })


@app.route("/api/admin/delete-product/<int:product_id>", methods=["DELETE"])
def api_admin_delete_product(product_id):
    ok, msg = engine.delete_product(product_id)
    if not ok:
        return jsonify({"success": False, "error": msg}), 400
    return jsonify({
        "success": True,
        "message": msg,
        "products": engine.products_df.to_dict(orient="records"),
        "stats": engine.get_catalog_stats()
    })


if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, port=5000)