# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 03:37:09 2025

@author: seid abdu
"""

import os
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define paths to your processed data
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, 'dashboard_historical_data.json')
CHANGE_POINT_RESULTS_PATH = os.path.join(DATA_DIR, 'change_point_results.json')
EVENTS_DATA_PATH = os.path.join(DATA_DIR, 'dashboard_events_data.json')

# Load data once when the app starts
historical_data = []
change_point_results = {}
events_data = []

def load_data_from_json():
    """Helper function to load all JSON data at startup."""
    global historical_data, change_point_results, events_data
    try:
        with open(HISTORICAL_DATA_PATH, 'r') as f:
            historical_data = json.load(f)
        print(f"Loaded historical data from {HISTORICAL_DATA_PATH}")

        with open(CHANGE_POINT_RESULTS_PATH, 'r') as f:
            change_point_results = json.load(f)
        print(f"Loaded change point results from {CHANGE_POINT_RESULTS_PATH}")

        with open(EVENTS_DATA_PATH, 'r') as f:
            events_data = json.load(f)
        print(f"Loaded events data from {EVENTS_DATA_PATH}")

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. Please run the Jupyter notebooks in order.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: A data file is not valid JSON. Please check the file for syntax errors or rerun the notebooks.")
        print(f"File: {e.doc}")
        print(f"JSON Decode Error: {e.msg} at line {e.lineno}, column {e.colno}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

# Call the function to load data before the first request
load_data_from_json()


@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Brent Oil Price Analysis API. Use /api/historical-prices, /api/change-point, /api/events."})

@app.route('/api/historical-prices')
def get_historical_prices():
    """
    API endpoint to serve historical Brent oil prices and log returns.
    """
    return jsonify(historical_data)

@app.route('/api/change-point')
def get_change_point_data():
    """
    API endpoint to serve change point analysis results.
    """
    return jsonify(change_point_results)

@app.route('/api/events')
def get_events_data():
    """
    API endpoint to serve the compiled key events data.
    """
    return jsonify(events_data)

if __name__ == '__main__':
    # For local development, run with `python app.py`
    # In production, use Gunicorn: `gunicorn -w 4 app:app`
    app.run(debug=True, port=5000, use_reloader=False)