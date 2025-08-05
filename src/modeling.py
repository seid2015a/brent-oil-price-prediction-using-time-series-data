# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 04:41:37 2025

@author: Seid A.
"""
import pymc as pm
import numpy as np
import pandas as pd
import json
import arviz as az
from scipy.stats import mode


def build_single_change_point_model(data):
    """
    Builds a Bayesian single change point model for the given data (log returns).
    The model assumes a change in both mean and standard deviation.

    Args:
        data (pd.Series or np.array): The time series data (e.g., log returns) to model.

    Returns:
        pymc3.Model: The PyMC3 model object.
    """
    n_data = len(data)
    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_data-1)
        
        # Define parameters for the 'before' period
        mu_before = pm.Normal("mu_before", mu=0, sigma=1)
        sigma_before = pm.HalfNormal("sigma_before", sigma=1)
        
        # Define parameters for the 'after' period
        mu_after = pm.Normal("mu_after", mu=0, sigma=1)
        sigma_after = pm.HalfNormal("sigma_after", sigma=1)
        
        # Create indices for the switch function
        idx = np.arange(n_data)
        
        # Use pm.math.switch to select parameters based on tau
        mu = pm.math.switch(idx < tau, mu_before, mu_after)
        sigm = pm.math.switch(idx < tau, sigma_before, sigma_after)
        
        observation = pm.Normal("observation",mu = mu,sigma = sigm, observed = data)
        
    return model

def run_inference(model,draws = 2000, tune = 1000, chains = 4, target_accept=0.9):
    # Runs the MCMC inference for the PyMC3 model.
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                          return_inferencedata=True, random_seed=42)
    return trace
def get_change_point_summary(trace, data_dates):
    """
    Extracts and summarizes change point information from the trace.

    Args:
        trace (arviz.InferenceData): The inference data object.
        data_dates (pd.Series): The dates corresponding to the data points.

    Returns:
        dict: A dictionary containing change point summary.
    """
    # Get posterior samples for tau
    tau_samples = trace.posterior["tau"].values.flatten()
    
    # Use the mode to find the most probable change point index.
    mode_result = mode(tau_samples)
    if np.isscalar(mode_result.mode):
        most_probable_tau_idx = int(mode_result.mode)
    else:
        most_probable_tau_idx = int(mode_result.mode[0])

    # Get the corresponding date
    change_point_date = data_dates.iloc[most_probable_tau_idx]

    # Get parameter summaries from arviz.summary()
    summary_df = az.summary(trace, var_names=["mu_before", "sigma_before", "mu_after", "sigma_after"], round_to=4)
    
    # Convert the DataFrame to a dictionary and then manually convert any NumPy types
    # to native Python types.
    summary_dict = summary_df.to_dict('index')
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_summary_dict = convert_to_serializable(summary_dict)

    results = {
        "most_probable_change_point_index": most_probable_tau_idx,
        "most_probable_change_point_date": change_point_date.strftime('%Y-%m-%d'),
        "parameter_summary": serializable_summary_dict
    }

    # Add posterior quantiles for mu and sigma, handling array outputs from az.hdi
    for param in ["mu_before", "sigma_before", "mu_after", "sigma_after"]:
        # Median is always a single value, so this is safe.
        results[f"{param}_posterior_median"] = float(np.median(trace.posterior[param].values))
        
        # arviz.hdi might return an array if the posterior is multi-dimensional.
        hdi_result = az.hdi(trace.posterior[param].values, hdi_prob=0.94)

        if np.isscalar(hdi_result):
            # If hdi_result is a scalar, it's a single value (unlikely, but safe to check)
            results[f"{param}_posterior_hdi_lower"] = float(hdi_result)
            results[f"{param}_posterior_hdi_upper"] = float(hdi_result)
        elif hdi_result.ndim == 1:
            # If hdi_result is a 1-D array of length 2, it's the standard HDI interval.
            results[f"{param}_posterior_hdi_lower"] = float(hdi_result[0])
            results[f"{param}_posterior_hdi_upper"] = float(hdi_result[1])
        else:
            # For other array shapes, we convert the entire array to a list
            # and store it, ensuring it's JSON serializable.
            # This is a robust fallback for unexpected output.
            results[f"{param}_posterior_hdi"] = hdi_result.tolist()
    
    return results

def save_results(results, filepath):
    """
    Saves the analysis results to a JSON file.
    """
    with open(filepath, 'w') as f:
        # The 'results' object is now fully serializable
        json.dump(results, f, indent=4)

        
def load_results(filepath):
    with open(filepath,'r') as f:
        return json.load(f)

if __name__ == '__main__':
    
    from data_processing import load_brent_data, calculate_log_returns, load_event_data
    
    # Load and process data
    brent_df = load_brent_data('../data/raw/BrentOilPrices.csv')
    if brent_df is None:
        exit()
    brent_df_processed = calculate_log_returns(brent_df.copy())
    if brent_df_processed is None:
        exit()
        
    log_returns = brent_df_processed['Log_Return'].values
    data_dates = brent_df_processed['Date']
    
    print(f"Data points for modeling: {len(log_returns)}")

    # Build and run model
    model = build_single_change_point_model(log_returns)
    print("\nStarting MCMC sampling...")
    trace = run_inference(model)
    print("MCMC sampling complete.")

    # Check diagnostics
    print("\nModel Summary (R-hat values should be close to 1.0):")
    print(az.summary(trace, var_names=["tau", "mu_before", "sigma_before", "mu_after", "sigma_after"], round_to=2))
    


    # Get change point summary
    change_point_summary = get_change_point_summary(trace, data_dates)
    print("\nChange Point Analysis Summary:")
    print(json.dumps(change_point_summary, indent=4))

    # Save results
    save_results(change_point_summary, '../data/processed/change_point_results.json')
    print("\nChange point results saved to data/processed/change_point_results.json")

    # Load events for association
    events_df = load_event_data('../data/raw/key_events.csv')
    if events_df is not None:
        change_date = pd.to_datetime(change_point_summary['most_probable_change_point_date'])
        
        # Find events around the change point (e.g., +/- 90 days)
        relevant_events = events_df[
            (events_df['Date'] >= change_date - pd.Timedelta(days=90)) &
            (events_df['Date'] <= change_date + pd.Timedelta(days=90))
        ]
        
        print(f"\nEvents around the detected change point ({change_date.strftime('%Y-%m-%d')}):")
        if not relevant_events.empty:
            print(relevant_events)
        else:
            print("No significant events found around the detected change point within 90 days.")

        # Quantify impact example
        mu_before = change_point_summary['mu_before_posterior_median']
        mu_after = change_point_summary['mu_after_posterior_median']
        
        # For log returns, a change in mean log return translates to a geometric mean shift
        # exp(mu) is the geometric mean growth factor
        impact_percent_change = (np.exp(mu_after) - np.exp(mu_before)) / np.exp(mu_before) * 100
        print(f"\nQuantified Impact:")
        print(f"Average daily log return before change: {mu_before:.4f}")
        print(f"Average daily log return after change: {mu_after:.4f}")
        print(f"Implied geometric mean daily price change: {impact_percent_change:.2f}%")
        print(f"This indicates a relative daily price shift of approximately {impact_percent_change:.2f}% around the change point.")



    
        
