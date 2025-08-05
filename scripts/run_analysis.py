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
    brent_data_path = 'data/raw/BrentOilPrices.csv'
    events_data_path = 'data/raw/key_events.csv'
    processed_data_output_path = 'data/processed/processed_brent_data.csv'
    change_point_results_path = 'data/processed/change_point_results.json'

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


    # --- Task 2: Change Point Modeling and Insight Generation ---
    print("\n--- Starting Bayesian Change Point Modeling ---")

    log_returns_data = brent_df_processed['Log_Return'].values
    data_dates = brent_df_processed['Date']

    print(f"Modeling {len(log_returns_data)} data points for change point detection.")

    # Build the PyMC3 model
    model = build_single_change_point_model(log_returns_data)
    print("PyMC3 model built successfully.")

    # Run MCMC inference
    print("Running MCMC inference (this may take a few minutes)...")
    trace = run_inference(model)
    print("MCMC inference completed.")

    # Check model convergence
    print("\nChecking model convergence (R-hat values close to 1.0 are good):")
    model_summary = az.summary(trace, var_names=["tau", "mu_before", "sigma_before", "mu_after", "sigma_after"], round_to=2)
    print(model_summary)

    # Plot trace for visual inspection of convergence
    # fig, axes = az.plot_trace(trace, var_names=["tau", "mu_before", "sigma_before", "mu_after", "sigma_after"], compact=False)
    # plt.tight_layout()
    # plt.savefig('notebooks/model_trace_plots.png')
    # print("Model trace plots saved to notebooks/model_trace_plots.png")
    # plt.show() # Uncomment to display plots during script execution

    # Get and save change point summary
    print("\nExtracting change point summary and insights...")
    change_point_summary = get_change_point_summary(trace, data_dates)
    save_results(change_point_summary, change_point_results_path)
    print(f"Change point analysis results saved to: {change_point_results_path}")

    print("\n--- Interpreting Results and Associating with Events ---")
    most_probable_tau_idx = change_point_summary['most_probable_change_point_index']
    detected_change_date = pd.to_datetime(change_point_summary['most_probable_change_point_date'])

    print(f"Most probable change point detected at index: {most_probable_tau_idx}")
    print(f"Corresponding date: {detected_change_date.strftime('%Y-%m-%d')}")

    # Compare with compiled event data
    # Consider a window of +/- 60 days for association
    association_window_days = 60
    relevant_events = events_df[
        (events_df['Date'] >= detected_change_date - pd.Timedelta(days=association_window_days)) &
        (events_df['Date'] <= detected_change_date + pd.Timedelta(days=association_window_days))
    ]

    print(f"\nPotential relevant events within +/- {association_window_days} days of the detected change point:")
    if not relevant_events.empty:
        print(relevant_events.to_string(index=False))
        # Formulate hypothesis (example)
        print("\nHypothesis Formation:")
        print(f"The detected change point on {detected_change_date.strftime('%Y-%m-%d')} is highly likely associated with the following event(s) given their proximity:")
        for _, row in relevant_events.iterrows():
            print(f"- {row['Date'].strftime('%Y-%m-%d')}: {row['EventDescription']} ({row['EventType']})")
    else:
        print("No significant events found in the compiled list around this change point within the specified window.")

    # Quantify Impact
    mu_before = change_point_summary['mu_before_posterior_median']
    mu_after = change_point_summary['mu_after_posterior_median']
    sigma_before = change_point_summary['sigma_before_posterior_median']
    sigma_after = change_point_summary['sigma_after_posterior_median']

    # Calculate percentage change in geometric mean daily price
    # A log return of `r` means price changes by factor `exp(r)`.
    # So, exp(mu_after) / exp(mu_before) = exp(mu_after - mu_before) is the factor change in geometric mean daily price.
    price_change_factor = np.exp(mu_after - mu_before)
    percent_change_in_geometric_mean = (price_change_factor - 1) * 100

    volatility_change_percent = ((sigma_after - sigma_before) / sigma_before) * 100

    print("\nQuantified Impact of the Detected Change Point:")
    print(f"  Average daily log return BEFORE: {mu_before:.4f} (Posterior HDI: [{change_point_summary['mu_before_posterior_hdi_lower']:.4f}, {change_point_summary['mu_before_posterior_hdi_upper']:.4f}])")
    print(f"  Average daily log return AFTER:  {mu_after:.4f} (Posterior HDI: [{change_point_summary['mu_after_posterior_hdi_lower']:.4f}, {change_point_summary['mu_after_posterior_hdi_upper']:.4f}])")
    print(f"  Implied geometric mean daily price change: {percent_change_in_geometric_mean:.2f}%")
    print(f"  Daily volatility (Std Dev of Log Return) BEFORE: {sigma_before:.4f} (Posterior HDI: [{change_point_summary['sigma_before_posterior_hdi_lower']:.4f}, {change_point_summary['sigma_before_posterior_hdi_upper']:.4f}])")
    print(f"  Daily volatility (Std Dev of Log Return) AFTER:  {sigma_after:.4f} (Posterior HDI: [{change_point_summary['sigma_after_posterior_hdi_lower']:.4f}, {change_point_summary['sigma_after_posterior_hdi_upper']:.4f}])")
    print(f"  Volatility change: {volatility_change_percent:.2f}%")

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    # Ensure current working directory is the project root when running this script
    # This assumes `run_analysis.py` is in the `scripts/` folder
    # Adjust if run from a different directory
    os.chdir(os.path.dirname(os.path.abspath('scripts'))) # Change to scripts directory
    os.chdir('../') # Move up to project root

    main()
