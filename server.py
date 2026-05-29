from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from datetime import datetime

# Import everything from the existing app
from app import (
    process_customer_request,
    init_database,
    db_engine,
    get_all_inventory,
    get_cash_balance,
    generate_financial_report,
)

app = Flask(__name__, static_folder=".")
CORS(app)

# Initialize DB on startup
print("Initializing database...")
init_database(db_engine)
print("Database ready.")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/request", methods=["POST"])
def handle_request():
    data = request.json
    user_request = data.get("request", "").strip()
    current_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    if not user_request:
        return jsonify({"error": "No request provided"}), 400

    try:
        response = process_customer_request(user_request, current_date)
        cash = get_cash_balance(current_date)
        return jsonify({
            "response": response,
            "cash_balance": cash,
            "date": current_date,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        inventory = get_all_inventory(date)
        return jsonify({"inventory": inventory, "date": date})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/financials", methods=["GET"])
def get_financials():
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        report = generate_financial_report(date)
        # Convert to JSON-serializable format
        report["inventory_summary"] = [
            {k: float(v) if hasattr(v, "item") else v for k, v in item.items()}
            for item in report["inventory_summary"]
        ]
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
