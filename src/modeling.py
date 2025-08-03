# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 04:41:37 2025

@author: Seid A.
"""
import pymc3 as pm
import numpy as np
import pandas as pd
import json
import arviz as az


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
        tau = pm.DiscretUniform("tau", lower=0, upper=n_data-1)
        
        # Define parameters for the 'before' period
        mu_before = pm.Normal("mu_before", mu=0, sd=1)
        sigma_before = pm.HalfNormal("sigma_before", sd=1)
        
        # Define parameters for the 'after' period
        mu_after = pm.Normal("mu_after", mu=0, sd=1)
        sigma_after = pm.HalfNormal("sigma_after", sd=1)
        
        # Create indices for the switch function
        idx = np.arange(n_data)
        
        # Use pm.math.switch to select parameters based on tau
        mu = pm.math.switch(idx < tau, mu_before, mu_after)
        sigma = pm.math.switch(idx < tau, sigma_before, sigma_after)
        
        observation = pm.Normal("observation",mu = mu,sd = sigma, observed = data)
        
    return model

def run_inference(model,draws = 2000, tune = 1000, chains = 4, target_accept=0.9):
    # Runs the MCMC inference for the PyMC3 model.
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                          return_inferencedata=True, random_seed=42)
    return trace
def get_change_point_summary(trace, data_dates):
    # Get posterior samples for tau
    tau_samples = trace.posterior["tau"].values.flatten()
    
    from scipy.stats import mode
    most_probable_tau_idx = int(mode(tau_samples)[0][0])
    change_point_date = data_dates.iloc[most_probable_tau_idx]
    
    # Get parameter summaries
    summary = az.summery(trace, var_names=["mu_before","mu_after","sigma_before","sigma_after"],round_to=4)
    
    results = {
        "most_probable_change_point_index": most_probable_tau_idx,
        "most_probable_change_point_date": change_point_date.strftime('%Y-%m-%d'),
        "parameter_summary": summary.to_dict('index')
    }
    
    # Add posterior quantiles for mu and sigma
    for param in ["mu_before", "sigma_before", "mu_after", "sigma_after"]:
        results[f"{param}_posterior_median"] = np.median(trace.posterior[param].values)
        results[f"{param}_posterior_hdi_lower"] = az.hdi(trace.posterior[param].values, hdi_prob=0.94)[0]
        results[f"{param}_posterior_hdi_upper"] = az.hdi(trace.posterior[param].values, hdi_prob=0.94)[1]

    return results

def save_results(results,filepath):
    with open(filepath,'w') as f:
        json.dump(results,f,indent=4)
        
def load_results(filepath):
    with open(filepath,'r') as f:
        return json.load(f)

if __name__ == '__main__':
    
    from src.data_processing import load_brent_data, calculate_log_returns, load_event_data
    
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



    
        
