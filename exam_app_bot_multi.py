import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta
import time as time_module
from supabase import create_client, Client
import json
import requests
import io

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming api_helper.py is in the parent directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api_helper import NorenApiPy
except ImportError:
    st.error("Could not import NorenApiPy. Please ensure api_helper.py is in the correct path.")
    st.stop()

# --- Flattrade API Credentials ---
# IMPORTANT: Replace these with your actual credentials or use Streamlit secrets.
USER_SESSION = "YOUR_FLATTRADE_SESSION_TOKEN_HERE"
USER_ID = "YOUR_FLATTRADE_USER_ID_HERE"
FLATTRADE_PASSWORD = "YOUR_FLATTRADE_PASSWORD_HERE"

# --- Supabase Credentials ---
# IMPORTANT: Replace these with your actual Supabase credentials.
SUPABASE_URL = "YOUR_SUPABASE_URL_HERE"
SUPABASE_KEY = "YOUR_SUPABASE_KEY_HERE"

# Global variables for API and Supabase clients
api = None
supabase_client: Client = None

# --- API and DB Initialization Functions ---
def initialize_flattrade_api():
    global api
    if api is None:
        try:
            api = NorenApiPy()
            # Attempt a login to get session
            login_result = api.login(userid=USER_ID, password=FLATTRADE_PASSWORD, twofa="YOUR_2FA_TOKEN_HERE", vendor_code="YOUR_VENDOR_CODE_HERE")
            if login_result['stat'] == 'Ok':
                api.set_session(USER_ID, USER_SESSION)
                logging.info(f"Flattrade API initialized and session set for {USER_ID}")
                return api
            else:
                st.error(f"Failed to log in to Flattrade API. Error: {login_result.get('emsg', 'Unknown')}")
                return None
        except Exception as e:
            st.error(f"Failed to initialize Flattrade API: {e}")
            return None
    return api

def initialize_supabase_client():
    global supabase_client
    if supabase_client is None:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logging.info("Supabase client initialized successfully.")
            return supabase_client
        except Exception as e:
            st.error(f"Failed to initialize Supabase client: {e}")
            return None
    return supabase_client

# --- Data Loading and Processing ---
def load_nse_equity_stocks():
    """Simulates loading a list of stocks from nse_equity.csv."""
    # In a real scenario, you would load this from a file.
    # For demonstration, we'll use a small, hardcoded list.
    data = """\
symbol,token
RELIANCE,2885
TCS,11536
HDFCBANK,1333
INFY,1594
HINDUNILVR,1394"""
    df = pd.read_csv(io.StringIO(data))
    return df

def calculate_sdvwap(df, sd_multiplier=1):
    """
    Calculates VWAP and a standard deviation band around it.
    Args:
        df (pd.DataFrame): DataFrame with 'time', 'vwap', 'c' (close), 'v' (volume) columns.
        sd_multiplier (float): Multiplier for the standard deviation band.
    Returns:
        tuple: (vwap_series, upper_band, lower_band)
    """
    df = df.copy()
    if 'v' not in df.columns or 'c' not in df.columns:
        return None, None, None
    
    # Calculate VWAP
    df['cum_volume'] = df['v'].cumsum()
    df['cum_price_volume'] = (df['c'] * df['v']).cumsum()
    df['vwap'] = df['cum_price_volume'] / df['cum_volume']

    # Calculate Standard Deviation of price from VWAP
    df['vwap_deviation'] = (df['c'] - df['vwap'])**2 * df['v']
    df['vwap_variance'] = df['vwap_deviation'].cumsum() / df['cum_volume']
    df['sd'] = np.sqrt(df['vwap_variance'])

    upper_band = df['vwap'] + (df['sd'] * sd_multiplier)
    lower_band = df['vwap'] - (df['sd'] * sd_multiplier)
    
    return df['vwap'], upper_band, lower_band

# --- Supabase Functions ---
def upsert_trade_to_supabase(trade_data):
    """Inserts or updates a trade record in Supabase."""
    try:
        response = supabase_client.table('trades').upsert(trade_data).execute()
        return response
    except Exception as e:
        logging.error(f"Failed to upsert trade data to Supabase: {e}")
        return None

def get_tracked_trades_from_supabase():
    """Fetches all tracked trades from Supabase."""
    try:
        response = supabase_client.table('trades').select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to fetch trades from Supabase: {e}")
        return pd.DataFrame()

# --- Core Trading Logic ---
def check_trading_conditions(symbol, token, api_client, supa_client, tracked_trades, sd_multiplier=1):
    """
    Checks for the "two green or red candles" condition and manages trades.
    """
    st.info(f"Scanning {symbol}...")
    try:
        # Get historical data (1-minute candles)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=15) # Fetch 15 minutes of data
        history = api_client.get_history(exchange='NSE', token=token, starttime=start_time.timestamp(), interval='1')
        
        if not history or history['stat'] != 'Ok' or not history['values']:
            logging.warning(f"No history data found for {symbol}")
            return
        
        df = pd.DataFrame(history['values'])
        
        # Calculate SDVWAP bands
        vwap, upper_band, lower_band = calculate_sdvwap(df, sd_multiplier)
        if vwap is None:
            logging.warning(f"Could not calculate SDVWAP for {symbol}")
            return
        
        # Get the last two candles
        last_two_candles = df.iloc[-2:]
        if len(last_two_candles) < 2:
            return

        last_candle = last_two_candles.iloc[1]
        second_last_candle = last_two_candles.iloc[0]

        is_last_green = last_candle['c'] > last_candle['o']
        is_second_last_green = second_last_candle['c'] > second_last_candle['o']
        is_last_red = last_candle['c'] < last_candle['o']
        is_second_last_red = second_last_candle['c'] < second_last_candle['o']

        # Check conditions
        is_above_band = last_candle['c'] > upper_band.iloc[-1] and second_last_candle['c'] > upper_band.iloc[-2]
        is_below_band = last_candle['c'] < lower_band.iloc[-1] and second_last_candle['c'] < lower_band.iloc[-2]

        is_two_green = is_last_green and is_second_last_green
        is_two_red = is_last_red and is_second_last_red

        current_price = last_candle['c']
        
        # Check if the stock is already being tracked
        tracked_trade = tracked_trades[tracked_trades['tsym'] == symbol]
        is_tracked = not tracked_trade.empty
        
        # --- Trading Logic ---
        if is_two_red and is_below_band and not is_tracked:
            # Condition met to initiate a long trade (two red candles below band)
            logging.info(f"LONG signal found for {symbol} at {current_price}")
            # Define trade details
            trade_data = {
                'tsym': symbol,
                'token': token,
                'status': 'Open',
                'entry_price': current_price,
                'quantity': 1, # Initial quantity
                'side': 'BUY',
                'entry_time': datetime.now().isoformat(),
            }
            upsert_trade_to_supabase(trade_data)
            st.success(f"Trade opened for {symbol}. Side: LONG")
            
        elif is_two_green and is_above_band and not is_tracked:
            # Condition met to initiate a short trade (two green candles above band)
            logging.info(f"SHORT signal found for {symbol} at {current_price}")
            trade_data = {
                'tsym': symbol,
                'token': token,
                'status': 'Open',
                'entry_price': current_price,
                'quantity': 1,
                'side': 'SELL',
                'entry_time': datetime.now().isoformat(),
            }
            upsert_trade_to_supabase(trade_data)
            st.success(f"Trade opened for {symbol}. Side: SHORT")

        # --- Reversal & Increasing Quantity ---
        elif is_tracked:
            trade = tracked_trade.iloc[0]
            current_side = trade['side']
            
            # Reversal check for open trades
            if current_side == 'BUY' and is_two_green and is_above_band:
                # Reversal from long to short, or increasing quantity
                new_quantity = trade['quantity'] + 1
                logging.info(f"Increasing quantity for {symbol} at {current_price}. New quantity: {new_quantity}")
                upsert_trade_to_supabase({
                    'tsym': symbol,
                    'quantity': new_quantity
                })
                st.warning(f"Increasing quantity for {symbol}. New quantity: {new_quantity}")
                
            elif current_side == 'SELL' and is_two_red and is_below_band:
                # Reversal from short to long, or increasing quantity
                new_quantity = trade['quantity'] + 1
                logging.info(f"Increasing quantity for {symbol} at {current_price}. New quantity: {new_quantity}")
                upsert_trade_to_supabase({
                    'tsym': symbol,
                    'quantity': new_quantity
                })
                st.warning(f"Increasing quantity for {symbol}. New quantity: {new_quantity}")

        # --- Close Trade at Target ---
        if is_tracked:
            trade = tracked_trade.iloc[0]
            entry_price = trade['entry_price']
            side = trade['side']
            quantity = trade['quantity']
            
            # Simple target price: 1% from entry
            target_price = entry_price * (1.01) if side == 'BUY' else entry_price * (0.99)
            
            if (side == 'BUY' and current_price >= target_price) or \
               (side == 'SELL' and current_price <= target_price):
                
                logging.info(f"Target hit for {symbol}. Closing trade at {current_price}.")
                # Here you would place a square-off order with the Flattrade API
                # api_client.place_order(buy_or_sell="SELL" if side == "BUY" else "BUY", ...)
                
                # Update Supabase to close the trade
                upsert_trade_to_supabase({
                    'tsym': symbol,
                    'status': 'Closed',
                    'exit_price': current_price,
                    'exit_time': datetime.now().isoformat()
                })
                st.success(f"Trade for {symbol} closed. Target hit.")
    
    except Exception as e:
        logging.error(f"Error checking trading conditions for {symbol}: {e}")

# --- Streamlit App UI ---
def main():
    st.set_page_config(
        page_title="Flattrade SDVWAP Scanner",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Flattrade Automated SDVWAP Scanner")
    st.markdown("This app scans NSE stocks for trading signals and manages positions using the Flattrade API.")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Settings")
        scan_interval = st.slider("Scan Interval (seconds)", 30, 300, 60, 30)
        sd_multiplier_input = st.number_input("SDVWAP Band Multiplier", 0.5, 3.0, 1.0, 0.1)

        st.markdown("---")
        run_screener = st.button("▶️ Start Screener", type="primary")
        stop_screener = st.button("⏹️ Stop Screener", type="secondary")
        
        st.markdown("---")
        st.write("For this app to work, you must set up a `trades` table in your Supabase database with columns like `tsym`, `status`, `entry_price`, `quantity`, etc.")

    # Initialize API and DB clients
    flattrade_api = initialize_flattrade_api()
    supabase_db = initialize_supabase_client()

    if not flattrade_api or not supabase_db:
        st.warning("API or Supabase initialization failed. Please check your credentials.")
        st.stop()

    # --- Main App Body ---
    st.subheader("🔥 Live Screener Status")
    status_placeholder = st.empty()
    
    if run_screener:
        status_placeholder.info("Screener is running...")
        
        # Load stocks
        stocks_df = load_nse_equity_stocks()
        if stocks_df.empty:
            st.error("No stocks found in nse_equity.csv. Please upload a valid file.")
        else:
            # Main screening loop
            while run_screener:
                start_time = time_module.time()
                tracked_trades = get_tracked_trades_from_supabase()

                for index, row in stocks_df.iterrows():
                    symbol = row['symbol']
                    token = str(row['token'])
                    check_trading_conditions(symbol, token, flattrade_api, supabase_db, tracked_trades, sd_multiplier_input)

                end_time = time_module.time()
                time_taken = end_time - start_time
                remaining_time = max(0, scan_interval - time_taken)
                
                status_placeholder.info(f"Scan cycle complete. Next scan in {int(remaining_time)} seconds...")
                time_module.sleep(remaining_time)
                
    elif stop_screener:
        status_placeholder.warning("Screener stopped.")
        
    # --- Display Tracked Trades ---
    st.subheader("📋 Tracked Positions")
    tracked_trades_df = get_tracked_trades_from_supabase()
    
    if not tracked_trades_df.empty:
        # Filter for open trades
        open_trades = tracked_trades_df[tracked_trades_df['status'] == 'Open']
        if not open_trades.empty:
            st.dataframe(open_trades, use_container_width=True)
        else:
            st.info("No open trades being tracked.")
    else:
        st.info("No trades in the database.")
        
    st.write("---")
    st.subheader("🚀 Account Details")
    try:
        limits = flattrade_api.get_limits()
        if limits and limits.get('stat') == 'Ok':
            cash = limits.get('cash', 'N/A')
            margin_used = limits.get('marginused', 'N/A')
            st.info(f"**Cash:** ₹{cash} | **Margin Used:** ₹{margin_used}")
        else:
            st.warning("Could not fetch account details.")
    except Exception as e:
        st.warning(f"Could not fetch account details: {e}")
        
    if st.button("🔄 Refresh Data", type="secondary"):
        st.rerun()

if __name__ == "__main__":
    # Check if the required API helper is available
    if 'NorenApiPy' not in sys.modules:
        st.error("Please ensure you have api_helper.py from the NorenApiPy library in the correct path.")
    else:
        main()
