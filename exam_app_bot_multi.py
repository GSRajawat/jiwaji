
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta
import time as time_module
from supabase import create_client, Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Add the parent directory to the path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy



# --- Flattrade API Credentials ---
USER_SESSION = "b6ded0e7caa5f7203ad542b9b89a19d73b85dc34037705e4f37ff70ac4200904"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2"  # Note: Session is usually preferred over password/2FA for live trading
# SUPABASE credentials are noted but not used in the Flattrade trading logic
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

# --- Strategy Parameters ---
PROFIT_TARGET = 0.04  # 4%
# Using a 20-period EMA as a proxy for the custom VWAP bands
# This is a major assumption as the custom VWAP bands are not standard API data.
EMA_PERIOD = 20

# --- Global State Variables for Trading Logic ---
# These mimic the state tracking in the Backtest class
entry_price = None
target_hit_today = False
last_exit_date = None
last_exit_direction = None # 'long' or 'short'
position_size = 0 # 0 for no position, positive for long, negative for short

def initialize_flattrade():
    """Initializes and logs into the Flattrade API."""
    try:
        ft = Flattrade(USER_ID)
        # Assuming the API requires login, though a session ID is provided.
        # Often a session ID (like USER_SESSION) is sufficient to instantiate and use.
        # If the API needs explicit login, this block is used:
        # login_response = ft.login(USER_ID, FLATTRADE_PASSWORD, 'OTP_OR_2FA_TOKEN')
        # print("Flattrade Login Response:", login_response)
        
        # Set the session ID provided in the credentials
        ft.set_session_id(USER_SESSION)
        
        # Check connection/session validity (optional but recommended)
        user_details = ft.get_user_details()
        if user_details and 'user_id' in user_details:
             print(f"✅ Flattrade API initialized successfully for User: {user_details['user_id']}")
             return ft
        else:
             print(f"❌ Failed to validate Flattrade session. Check USER_SESSION.")
             sys.exit(1)
             
    except Exception as e:
        print(f"❌ Error initializing Flattrade API: {e}")
        sys.exit(1)

def get_user_inputs():
    """Gets and validates user inputs for the trade."""
    print("\n--- Trade Setup ---")
    
    stock = input("Enter Stock/Instrument symbol (e.g., 'NTPC', 'RELIANCE'): ").upper().strip()
    
    while True:
        try:
            quantity = int(input("Enter Quantity (integer > 0): "))
            if quantity > 0:
                break
            print("Quantity must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    order_type = input("Enter Order Type (MIS or CNC, default is MIS): ").upper().strip() or "MIS"
    if order_type not in ["MIS", "CNC"]:
        print("Invalid Order Type. Defaulting to MIS.")
        order_type = "MIS"
        
    while True:
        try:
            capital = float(input("Enter Max Capital to risk (default 500.0): ")) or 500.0
            if capital > 0:
                break
            print("Capital must be a positive number.")
        except ValueError:
            print("Invalid input. Defaulting to 500.0.")
            capital = 500.0
            break

    print(f"\nTrade Parameters: Stock={stock}, Qty={quantity}, Type={order_type}, Capital Limit={capital}")
    return stock, quantity, order_type, capital

def get_instrument_token(ft: Flattrade, symbol: str):
    """Fetches the instrument token for a given symbol."""
    # Flattrade API usually needs the exchange and symbol. Assuming 'NSE' for equity.
    try:
        # This is a placeholder as the exact search API may vary.
        # Real-world APIs often require a mapping file or a search endpoint.
        # For a known stock like 'NTPC', the token might be looked up from a local file.
        print(f"Searching for instrument token for {symbol} on NSE...")
        
        # Mocking a search function to find the token
        search_result = ft.get_instrument_by_symbol('NSE', symbol) 
        
        if search_result and 'token' in search_result[0]:
            token = search_result[0]['token']
            print(f"Found Token: {token}")
            return token
        else:
            print(f"❌ Could not find instrument token for {symbol}. Check symbol and exchange.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error fetching instrument token: {e}")
        sys.exit(1)
        
def fetch_and_prepare_data(ft: Flattrade, instrument_token: str, symbol: str) -> pd.DataFrame:
    """
    Fetches the latest minute-level data and calculates the required indicators (EMA).
    
    NOTE: This is a placeholder. Flattrade API usually provides historic candles.
    We are mocking the data fetching for demonstration.
    
    In a real scenario, you'd fetch the last N minutes of data (e.g., 50 bars for a 5-min EMA)
    and then append the real-time tick/quote data to calculate the latest value.
    """
    print(f"Fetching latest minute data for {symbol} (Token: {instrument_token})...")
    
    # --- MOCK DATA FOR DEMO ---
    # You must replace this with actual API calls (e.g., historical data endpoint)
    # The API might be `ft.get_historical_data(instrument_token, 'MINUTE', count=50)`
    data = {
        'Open': [100, 101, 102, 103, 104],
        'High': [102, 103, 104, 105, 106],
        'Low': [99, 100, 101, 102, 103],
        'Close': [101, 102, 103, 104, 105],
        'Volume': [1000, 1200, 1100, 1300, 1400],
        # Replace the custom VWAP bands with an EMA for this demo
        'EMA': [99.5, 100.5, 101.5, 102.5, 103.5] 
    }
    index = [
        dt.datetime.now() - dt.timedelta(minutes=i) for i in range(5, 0, -1)
    ]
    df = pd.DataFrame(data, index=index)
    df.index.name = 'Date'
    # --- END MOCK DATA ---
    
    # In a real scenario, calculate the EMA on the fetched data
    # df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    # The strategy needs the last two candles (at least)
    if len(df) < 2:
        print("❌ Not enough data points to run strategy.")
        return None
        
    print(f"Data prepared. Last close: {df['Close'].iloc[-1]}, Last EMA: {df['EMA'].iloc[-1]}")
    return df

def execute_trade(ft: Flattrade, instrument_token: str, symbol: str, quantity: int, order_type: str, price: float, transaction_type: str):
    """Places an order using the Flattrade API."""
    
    # Flattrade API uses 'BUY' or 'SELL' for transaction_type
    # order_type: 'MIS' or 'CNC'
    # product_type: 'MARKET', 'LIMIT', etc.
    
    print(f"Attempting to place {transaction_type} order for {quantity} of {symbol} at Market price...")

    try:
        order_response = ft.place_order(
            instrument_token=instrument_token,
            transaction_type=transaction_type,  # 'BUY' or 'SELL'
            quantity=quantity,
            product_type=order_type,            # 'MIS' or 'CNC'
            order_type='MARKET',                # Placing Market Order
            price=0.0                           # Market price, so price is 0
        )
        
        if order_response and 'order_id' in order_response:
            print(f"✅ Order placed successfully! ID: {order_response['order_id']}")
            # In a real system, you would check for order execution status here
            return True
        else:
            print(f"❌ Order failed: {order_response}")
            return False
            
    except Exception as e:
        print(f"❌ Error executing trade: {e}")
        return False

def check_and_trade(ft: Flattrade, instrument_token: str, symbol: str, base_quantity: int, order_type: str, df: pd.DataFrame):
    """
    Applies the strategy logic to the latest data and executes trades.
    
    NOTE: Replaced custom VWAP bands with a 20-period EMA (df['EMA']).
    Buy/Sell conditions are adapted: Crossover of Close and EMA.
    """
    global entry_price, target_hit_today, last_exit_date, last_exit_direction, position_size
    
    current_datetime = df.index[-1]
    current_time = current_datetime.time()
    current_date = current_datetime.date()
    current_close = df['Close'].iloc[-1]
    
    # --- Daily Reset (Mimics Backtest init) ---
    # This should ideally be outside and check against a persistent state
    # For a simple script, we assume a new run or a timed loop handles this.
    
    # --- Time Check ---
    if current_time < dt.time(9, 30):
        print("Market not open yet (before 9:30 AM). Waiting...")
        return
        
    # --- Intraday Square Off (3:20 PM) ---
    if current_time >= dt.time(15, 20):
        if position_size != 0:
            print(f"🕒 Squaring off position at 3:20 PM...")
            transaction_type = 'BUY' if position_size < 0 else 'SELL'
            success = execute_trade(ft, instrument_token, symbol, abs(position_size), 'MIS', current_close, transaction_type)
            if success:
                position_size = 0
                entry_price = None
                target_hit_today = True # No re-entry after final square-off
        return
        
    # --- Exit Logic (Profit Target) ---
    if position_size != 0 and entry_price is not None:
        profit_pct = 0
        if position_size > 0: # Long
            profit_pct = (current_close - entry_price) / entry_price
        elif position_size < 0: # Short
            profit_pct = (entry_price - current_close) / entry_price
            
        if profit_pct >= PROFIT_TARGET:
            print(f"💰 Profit target ({PROFIT_TARGET*100}%) hit at {current_close}. Exiting position.")
            transaction_type = 'BUY' if position_size < 0 else 'SELL'
            success = execute_trade(ft, instrument_token, symbol, abs(position_size), 'MIS', current_close, transaction_type)
            
            if success:
                # Update state after successful exit
                last_exit_date = current_date
                last_exit_direction = 'long' if position_size > 0 else 'short'
                target_hit_today = True
                position_size = 0
                entry_price = None
                return # Stop further logic for this bar
    
    # --- Target Hit Restriction ---
    if target_hit_today:
        print(f"🚫 Target hit today. No further entries.")
        return

    # --- Strategy Entry Logic (Adapted to Close vs EMA) ---
    # Get previous two candles' data and EMA values
    candle_minus_1_close = df['Close'].iloc[-1]
    candle_minus_2_close = df['Close'].iloc[-2]
    ema_val_1 = df['EMA'].iloc[-1]
    ema_val_2 = df['EMA'].iloc[-2]

    # SELL condition (Simplified): Close crosses below EMA and stays below.
    # Original logic: close < open AND close < vwap_minus (for two consecutive bars)
    sell_condition_1 = candle_minus_2_close < ema_val_2
    sell_condition_2 = candle_minus_1_close < ema_val_1
    
    # BUY condition (Simplified): Close crosses above EMA and stays above.
    # Original logic: close > open AND close > vwap_plus (for two consecutive bars)
    buy_condition_1 = candle_minus_2_close > ema_val_2
    buy_condition_2 = candle_minus_1_close > ema_val_1

    if position_size == 0:
        # No position - enter fresh
        if sell_condition_1 and sell_condition_2:
            if last_exit_direction != 'short' or last_exit_date != current_date:
                print(f"⬇️ SELL signal detected. Entering fresh SHORT position.")
                success = execute_trade(ft, instrument_token, symbol, base_quantity, order_type, current_close, 'SELL')
                if success:
                    position_size = -base_quantity
                    entry_price = current_close
            else:
                print("🚫 SELL signal ignored: Restricted due to previous same-direction exit today.")
                
        elif buy_condition_1 and buy_condition_2:
            if last_exit_direction != 'long' or last_exit_date != current_date:
                print(f"⬆️ BUY signal detected. Entering fresh LONG position.")
                success = execute_trade(ft, instrument_token, symbol, base_quantity, order_type, current_close, 'BUY')
                if success:
                    position_size = base_quantity
                    entry_price = current_close
            else:
                print("🚫 BUY signal ignored: Restricted due to previous same-direction exit today.")
    
    else:
        # Have position - check for opposite signal (Reverse and Triple)
        current_abs_size = abs(position_size)
        triple_size = current_abs_size * 3
        
        if position_size > 0 and sell_condition_1 and sell_condition_2: # Long position + Short signal
            print(f"🔄 Reverse signal (Long -> Short) detected. Exiting Long and entering {triple_size} Short.")
            # Execute a SELL order for triple_size: This closes the 'current_abs_size' long
            # and opens a short for 'triple_size - current_abs_size'
            success = execute_trade(ft, instrument_token, symbol, triple_size, order_type, current_close, 'SELL')
            if success:
                position_size = -triple_size
                entry_price = current_close
                
        elif position_size < 0 and buy_condition_1 and buy_condition_2: # Short position + Long signal
            print(f"🔄 Reverse signal (Short -> Long) detected. Exiting Short and entering {triple_size} Long.")
            # Execute a BUY order for triple_size: This closes the 'current_abs_size' short
            # and opens a long for 'triple_size - current_abs_size'
            success = execute_trade(ft, instrument_token, symbol, triple_size, order_type, current_close, 'BUY')
            if success:
                position_size = triple_size
                entry_price = current_close
                
def main():
    """Main function to run the live trading script."""
    
    # 1. Initialize API and get user inputs
    ft = initialize_flattrade()
    symbol, quantity, order_type, capital = get_user_inputs()
    
    # 2. Get Instrument Token
    instrument_token = get_instrument_token(ft, symbol)
    
    # --- Live Trading Loop ---
    # In a real scenario, this loop runs every minute (or 5 minutes, matching the bar size)
    # The total capital limit is for risk management outside this script.
    
    # For a demonstration, we will just run the logic once with mock data
    print("\n--- Starting Trading Simulation (Once-off run with mock data) ---")
    
    try:
        # 3. Fetch and prepare data
        df = fetch_and_prepare_data(ft, instrument_token, symbol)
        
        if df is not None:
            # 4. Apply strategy and trade
            check_and_trade(ft, instrument_token, symbol, quantity, order_type, df)
            
    except KeyboardInterrupt:
        print("\n👋 Trading loop manually stopped.")
    except Exception as e:
        print(f"\n❌ An unhandled error occurred in the trading loop: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Trading session finished. ---")
    print(f"Final Position Size: {position_size} (Positive for Long, Negative for Short)")
    
if __name__ == "__main__":
    main()
