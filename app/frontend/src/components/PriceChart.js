# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 04:31:00 2025

@author: seid a.
"""
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const PriceChart = ({ data, changePointDate, events }) => {
  // Ensure data has 'Date' as string (YYYY-MM-DD) and 'Price', 'Log_Return' as numbers
  // This is handled in 03_dashboard_data_prep.ipynb

  // Filter events to only show relevant ones on the chart (e.g., major impacts)
  // For simplicity, we'll just plot all events for now.
  // You might want to add a filter or only include major events for annotation.

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="Date" angle={-45} textAnchor="end" height={80} interval="preserveStartEnd" />
        <YAxis yAxisId="left" domain={['auto', 'auto']} label={{ value: 'Price (USD)', angle: -90, position: 'insideLeft' }} />
        <YAxis yAxisId="right" orientation="right" label={{ value: 'Log Return', angle: 90, position: 'insideRight' }} />
        <Tooltip
          labelFormatter={(label) => `Date: ${label}`}
          formatter={(value, name, props) => {
            if (name === 'Price') return [`$${value.toFixed(2)}`, 'Price'];
            if (name === 'Log_Return') return [`${value.toFixed(4)}`, 'Log Return'];
            return [value, name];
          }}
        />
        <Legend />

        <Line yAxisId="left" type="monotone" dataKey="Price" stroke="#8884d8" name="Brent Price" dot={false} />
        <Line yAxisId="right" type="monotone" dataKey="Log_Return" stroke="#82ca9d" name="Log Return" dot={false} />

        {/* Highlight Change Point */}
        {changePointDate && (
          <ReferenceLine x={changePointDate} stroke="red" strokeDasharray="3 3" label={{ value: "Change Point", position: "insideTopRight", fill: "red", angle: 0 }} />
        )}

        {/* Annotate Major Events (optional, depends on density of events) */}
        {events.map((event, index) => {
          // You might want to filter events or strategically place labels to avoid clutter
          // For now, let's add a vertical line for each event
          return (
            <ReferenceLine
              key={`event-${index}`}
              x={event.Date}
              stroke="orange"
              strokeDasharray="5 5"
              // Add a label for significant events if not too many
              // label={{ value: event.EventType, position: "insideBottomRight", fill: "orange", angle: -90 }}
            />
          );
        })}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;
