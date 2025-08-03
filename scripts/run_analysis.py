# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 06:05:17 2025

@author: seid a.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import json
import os

from src.data_processing import load_brent_data, calculate_log_returns, load_event_data
from src.modeling import build_single_change_point_model, run_inference, get_change_point_summary, save_results

def main():
    print("--- Starting Brent Oil Price Analysis ---")

    # --- Task 1: Laying the Foundation for Analysis ---
    # Data Loading and Initial EDA
    brent_data_path = '../data/raw/BrentOilPrices.csv'
    events_data_path = '../data/raw/key_events.csv'
    processed_data_output_path = '../data/processed/processed_brent_data.csv'
    change_point_results_path = '../data/processed/change_point_results.json'

    print(f"\n1. Loading Brent Oil Price data from: {brent_data_path}")
    brent_df = load_brent_data(brent_data_path)
    if brent_df is None:
        print("Failed to load Brent oil price data. Exiting.")
        return

    print("Calculating log returns...")
    brent_df_processed = calculate_log_returns(brent_df.copy())
    if brent_df_processed is None:
        print("Failed to calculate log returns. Exiting.")
        return

    # Save processed data (optional, for dashboard prep later)
    brent_df_processed.to_csv(processed_data_output_path, index=False)
    print(f"Processed Brent data saved to: {processed_data_output_path}")

    # Plot raw prices and log returns for EDA
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(brent_df['Date'], brent_df['Price'])
    plt.title('Historical Brent Oil Prices (USD/barrel)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(brent_df_processed['Date'], brent_df_processed['Log_Return'])
    plt.title('Brent Oil Daily Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('notebooks/eda_plots.png') # Save EDA plots
    print("EDA plots saved to notebooks/eda_plots.png")
    # plt.show() # Uncomment to display plots during script execution

    print(f"\n2. Loading Key Event data from: {events_data_path}")
    events_df = load_event_data(events_data_path)
    if events_df is None:
        print("Failed to load event data. Exiting.")
        return
    print("Key Events Head:")
    print(events_df.head())