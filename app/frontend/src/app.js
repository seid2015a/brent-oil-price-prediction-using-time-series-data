# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 04:20:47 2025

@author: seid a
"""
import React, { useState, useEffect } from 'react';
import PriceChart from './components/PriceChart';
import EventTable from './components/EventTable';
import { getHistoricalPrices, getChangePointData, getEventsData } from './api';
import './App.css'; // Assuming you'll add some basic styling

function App() {
  const [historicalData, setHistoricalData] = useState([]);
  const [changePoint, setChangePoint] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const prices = await getHistoricalPrices();
        const cp = await getChangePointData();
        const evts = await getEventsData();

        setHistoricalData(prices);
        setChangePoint(cp);
        setEvents(evts);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch data. Make sure the Flask backend is running.");
        setLoading(false);
        console.error("Error fetching data:", err);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return <div className="loading-container">Loading dashboard data...</div>;
  }

  if (error) {
    return <div className="error-container">{error}</div>;
  }

  const detectedChangeDate = changePoint?.most_probable_change_point_date;
  const muBefore = changePoint?.mu_before_posterior_median;
  const muAfter = changePoint?.mu_after_posterior_median;
  const sigmaBefore = changePoint?.sigma_before_posterior_median;
  const sigmaAfter = changePoint?.sigma_after_posterior_median;

  const percentChangeInGeometricMean = muBefore !== undefined && muAfter !== undefined
    ? ((Math.exp(muAfter - muBefore) - 1) * 100).toFixed(2)
    : 'N/A';

  const volatilityChangePercent = sigmaBefore !== undefined && sigmaAfter !== undefined
    ? (((sigmaAfter - sigmaBefore) / sigmaBefore) * 100).toFixed(2)
    : 'N/A';

  return (
    <div className="App">
      <header className="App-header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <p>Impact of Global Events on Oil Prices</p>
      </header>

      <section className="dashboard-summary">
        <h2>Analysis Summary</h2>
        {changePoint && (
          <div className="summary-cards">
            <div className="card">
              <h3>Detected Change Point</h3>
              <p>Date: <strong>{detectedChangeDate || 'N/A'}</strong></p>
              {changePoint.price_at_change_point && <p>Price on this date: ${changePoint.price_at_change_point?.toFixed(2)}</p>}
            </div>
            <div className="card">
              <h3>Impact on Mean Daily Log Return</h3>
              <p>Before: {muBefore?.toFixed(4) || 'N/A'}</p>
              <p>After: {muAfter?.toFixed(4) || 'N/A'}</p>
              <p>Implied Daily Price Change: <strong>{percentChangeInGeometricMean}%</strong></p>
            </div>
            <div className="card">
              <h3>Impact on Daily Volatility</h3>
              <p>Before: {sigmaBefore?.toFixed(4) || 'N/A'}</p>
              <p>After: {sigmaAfter?.toFixed(4) || 'N/A'}</p>
              <p>Volatility Change: <strong>{volatilityChangePercent}%</strong></p>
            </div>
          </div>
        )}
      </section>

      <section className="price-chart-section">
        <h2>Historical Brent Oil Prices & Log Returns</h2>
        {historicalData.length > 0 ? (
          <PriceChart data={historicalData} changePointDate={detectedChangeDate} events={events} />
        ) : (
          <p>No historical price data available.</p>
        )}
      </section>

      <section className="events-table-section">
        <h2>Key Geopolitical and Economic Events</h2>
        {events.length > 0 ? (
          <EventTable events={events} changePointDate={detectedChangeDate} />
        ) : (
          <p>No event data available.</p>
        )}
      </section>

      <footer>
        <p>&copy; 2025 Birhan Energies. Data-driven insights for the energy sector.</p>
      </footer>
    </div>
  );
}

export default App;
