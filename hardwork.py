#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import io

# Define the start and end dates
start_date = datetime(2020, 10, 1)
end_date = datetime(2023, 6, 21)  # Assuming today's date as the end date

# Get the Nifty 50 stock symbols
nifty50_symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'HDFC.NS', 'TCS.NS', 'ITC.NS', 'KOTAKBANK.NS',
                   'HINDUNILVR.NS', 'LT.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 'ASIANPAINT.NS',
                   'M&M.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'BAJAJFINSV.NS', 'HCLTECH.NS', 'ADANIENT.NS',
                   'TATASTEEL.NS', 'INDUSINDBK.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS', 'ULTRACEMCO.NS',
                   'NESTLEIND.NS', 'TECHM.NS', 'GRASIM.NS', 'CIPLA.NS', 'JSWSTEEL.NS', 'ADANIPORTS.NS', 'WIPRO.NS',
                   'HINDALCO.NS', 'SBILIFE.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'HDFCLIFE.NS', 'ONGC.NS', 'TATACONSUM.NS',
                   'DIVISLAB.NS', 'BAJAJ-AUTO.NS', 'BRITANNIA.NS', 'APOLLOHOSP.NS', 'COALINDIA.NS', 'UPL.NS',
                   'HEROMOTOCO.NS', 'BPCL.NS']  # List all 50 stock symbols here

# Fetch historical stock prices
data = yf.download(nifty50_symbols, start=start_date, end=end_date)

# Extract the 'Close' prices from the data
close_prices = data['Close']

# Calculate the number of shares for each stock
allocation_per_stock = 1000000 / 50
shares_per_stock = allocation_per_stock / close_prices.iloc[0]

# Calculate the portfolio value for each day
portfolio_value1 = (close_prices * shares_per_stock).sum(axis=1)


# Streamlit app
def main():
    # Set page title
    st.title("Nifty Index Equity Curve Simulation")

    # User inputs
    simulation_start_date = st.date_input("Simulation Start Date", value=pd.to_datetime(start_date))
    simulation_end_date = st.date_input("Simulation End Date", value=pd.to_datetime(end_date))
    performance_days = st.number_input("Number of Days for Stock Selection Performance", min_value=1, value=100)
    top_stocks = st.number_input("Number of Top Stocks to Select for Sample Strategy", min_value=1, value=10)
    initial_equity = st.number_input("Initial Equity", min_value=1, value=1000000)

    # Calculate equity curves
    benchmark_equity_curve = calculate_benchmark_equity_curve(simulation_start_date, simulation_end_date, initial_equity)
    sample_strategy_equity_curve, selected_stocks = calculate_sample_strategy_equity_curve(simulation_start_date,
                                                                                         simulation_end_date,
                                                                                         initial_equity, performance_days,
                                                                                         top_stocks, close_prices)

     # Plot the equity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portfolio_value1.index, portfolio_value1, label='Nifty 50 Portfolio')
    ax.plot(benchmark_equity_curve.index, benchmark_equity_curve, label='Benchmark Allocation')
    ax.plot(sample_strategy_equity_curve.index, sample_strategy_equity_curve, label='Sample Strategy')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity Value')
    ax.set_title('Equity Curves')
    ax.legend()
    ax.grid(True)
    
 # Save the figure to a file-like object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the figure using st.image()
    st.image(buffer)
    
    
    # Display selected stocks
    st.subheader("Selected Stocks for Sample Strategy")
    st.write(selected_stocks)

    # Calculate performance metrics
    benchmark_cagr, benchmark_volatility, benchmark_sharpe_ratio = calculate_performance_metrics(portfolio_value1,
                                                                                               portfolio_value1.pct_change().fillna(0))
    sample_strategy_cagr, sample_strategy_volatility, sample_strategy_sharpe_ratio = calculate_performance_metrics(
        sample_strategy_equity_curve, sample_strategy_equity_curve.pct_change().fillna(0))

    # Display performance metrics
    st.subheader("Performance Metrics")
    st.write("Benchmark Allocation:")
    st.write(f"CAGR: {benchmark_cagr:.2f}%")
    st.write(f"Volatility: {benchmark_volatility:.2f}%")
    st.write(f"Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")
    st.write("Sample Strategy:")
    st.write(f"CAGR: {sample_strategy_cagr:.2f}%")
    st.write(f"Volatility: {sample_strategy_volatility:.2f}%")
    st.write(f"Sharpe Ratio: {sample_strategy_sharpe_ratio:.2f}")


# Calculate the benchmark equity curve
def calculate_benchmark_equity_curve(start_date, end_date, initial_equity):
    # Select the data within the specified start and end dates
    benchmark_prices = portfolio_value1.loc[start_date:end_date]

    # Normalize the benchmark prices based on the initial equity
    benchmark_equity_curve = initial_equity * (benchmark_prices / benchmark_prices.iloc[0])

    return benchmark_equity_curve


# Calculate the sample strategy equity curve
def calculate_sample_strategy_equity_curve(start_date, end_date, initial_equity, performance_days, top_stocks,
                                           close_prices):
    # Convert start_date to datetime object
    start_date = np.datetime64(start_date)

    # Check if the specified start date is present in the index
    if start_date not in close_prices.index:
        # Find the nearest available date before the specified start date
        nearest_date = close_prices.index[close_prices.index < start_date][-1]
        # Update the start date to the nearest available date
        start_date = nearest_date

    # Select the data within the specified start and end dates
    sample_strategy_prices = close_prices.loc[start_date:end_date]

    # Calculate the percentage returns for the latest performance_days
    returns = sample_strategy_prices.pct_change(performance_days)

    # Get the top stocks based on the average returns
    selected_stocks = returns.iloc[-1].nlargest(top_stocks).index

    # Calculate the number of shares for each stock
    allocation_per_stock = initial_equity / top_stocks
    shares_per_stock = allocation_per_stock / sample_strategy_prices.loc[start_date, selected_stocks]

    # Calculate the portfolio value for each day
    sample_strategy_equity_curve = (sample_strategy_prices[selected_stocks] * shares_per_stock).sum(axis=1)

    return sample_strategy_equity_curve, selected_stocks


# Calculate performance metrics
def calculate_performance_metrics(equity_curve, returns):
    total_days = len(equity_curve)
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (365 / total_days) - 1) * 100
    volatility = returns.std() * (365 ** 0.5) * 100
    sharpe_ratio = (cagr - 5) / volatility  # Assuming risk-free rate of 5%

    return cagr, volatility, sharpe_ratio


if __name__ == '__main__':
    main()

