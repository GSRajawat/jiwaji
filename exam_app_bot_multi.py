import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time, timedelta, timezone
import time
import json
import os
from supabase import create_client, Client
import yfinance as yf

# ==========================================
# TIMEZONE CONFIGURATION (IST)
# ==========================================
IST = timezone(timedelta(hours=5, minutes=30))

# ==========================================
# SUPABASE DATABASE CREDENTIALS
# ==========================================
# Load secrets from Streamlit environment
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except KeyError:
    st.error("❌ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in Streamlit's secrets.")
    supabase = None
except Exception as e:
    st.error(f"❌ Failed to connect to Supabase: {e}")
    supabase = None

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NSE Enhanced Trading Strategy",
    page_icon="📊",
    layout="wide"
)

# ==================== CSV FALLBACK PATH ====================
LOCAL_CSV_FILE = "trades_database.csv"

# ==================== SUPABASE DB FUNCTIONS ====================

def save_trade_to_db(trade_data):
    """Save a new trade to the Supabase database"""
    if supabase is None:
        return {'status': 'error', 'message': "Supabase client not initialized."}

    # Ensure nullable fields are present, even if None
    trade_data.setdefault('exit_order_id', None)
    trade_data.setdefault('exit_price', None)
    trade_data.setdefault('exit_time', None)
    trade_data.setdefault('pnl', None)
    
    try:
        response = supabase.table("trades").insert(trade_data).execute()
        return {'status': 'success', 'message': f"Trade saved with ID: {trade_data['entry_order_id']}"}
    except Exception as e:
        return {'status': 'error', 'message': f"DB Save Error: {e}"}

def update_trade_exit(entry_order_id, exit_price, exit_order_id, pnl):
    """Update an existing trade with exit details and set status to CLOSED"""
    if supabase is None:
        return {'status': 'error', 'message': "Supabase client not initialized."}

    update_data = {
        'status': 'CLOSED',
        'exit_price': float(exit_price),
        'exit_order_id': exit_order_id,
        'exit_time': datetime.now(IST).isoformat(),
        'pnl': float(pnl)
    }
    
    try:
        response = supabase.table("trades").update(update_data).eq("entry_order_id", entry_order_id).execute()
        if response.data:
            return {'status': 'success', 'message': f"Trade {entry_order_id} closed."}
        else:
            return {'status': 'error', 'message': f"Trade {entry_order_id} not found."}
    except Exception as e:
        return {'status': 'error', 'message': f"DB Update Error: {e}"}

def get_open_trades():
    """Fetch all OPEN trades from the database"""
    if supabase is None:
        return pd.DataFrame()
    try:
        response = supabase.table("trades").select("*").eq("status", "OPEN").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"DB Fetch Error (Open Trades): {e}")
        return pd.DataFrame()

def get_all_trades():
    """Fetch all trades, ordered by entry_time descending"""
    if supabase is None:
        return pd.DataFrame()
    try:
        response = supabase.table("trades").select("*").order("entry_time", desc=True).execute()
        df = pd.DataFrame(response.data)
        
        # Clean data types for display
        if not df.empty:
            for col in ['entry_price', 'target_price', 'stop_loss', 'exit_price', 'pnl']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"DB Fetch Error (All Trades): {e}")
        return pd.DataFrame()

def reset_trades_db():
    """Delete all records from the trades table (DANGEROUS)"""
    if supabase is None:
        return {'status': 'error', 'message': "Supabase client not initialized."}
    try:
        # Note: Supabase policy should allow deletion
        response = supabase.table("trades").delete().gt("entry_time", "1970-01-01T00:00:00+00:00").execute()
        return {'status': 'success', 'message': f"Deleted {len(response.data)} trade records."}
    except Exception as e:
        return {'status': 'error', 'message': f"DB Reset Error: {e}"}

# ==================== MARKET DATA FETCHER (YFINANCE) ====================
class MarketDataFetcher:
    """Fetch broad market data using yfinance to replace the unreliable direct NSE calls."""
    
    def __init__(self):
        # A list of NSE F&O stocks with the required ".NS" suffix for Yahoo Finance.
        self.fno_symbols = [
            "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", 
            "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
            "SBIN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "WIPRO.NS", "TITAN.NS",
            "MARUTI.NS", "ADANIPORTS.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "POWERGRID.NS",
            "NESTLEIND.NS", "SUNPHARMA.NS", "HCLTECH.NS", "TECHM.NS", "NTPC.NS",
            "TATAMOTORS.NS", "ULTRACEMCO.NS", "GRASIM.NS", "JSWSTEEL.NS", "LTIM.NS"
        ]
        
    def fetch_all_securities(self):
        st.info(f"⏳ Fetching data for {len(self.fno_symbols)} F&O symbols using Yahoo Finance API...")
        
        try:
            # Fetch data for all symbols for the "1 day" period
            data = yf.download(self.fno_symbols, period="1d", progress=False, show_errors=False)
            
            if data.empty:
                st.error("❌ Yahoo Finance returned empty data.")
                return None

            st.success(f"✅ Successfully fetched data from Yahoo Finance.")
            return data
            
        except Exception as e:
            st.error(f"Error during data fetch from yfinance: {str(e)}")
            return None
    
    def parse_market_data(self, yf_data):
        """Converts yfinance multi-index data to a single DataFrame with required columns."""
        
        if yf_data is None or yf_data.empty:
            return pd.DataFrame()
        
        try:
            # Stack the data to get a single row per symbol/date
            df = yf_data.stack().reset_index()
            # Select only the latest date's data
            df = df[df['Date'] == df['Date'].max()]

            # Rename columns to match strategy expectations
            column_mapping = {
                'level_1': 'symbol', 
                'Close': 'ltp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'tradedValue', # Using Volume as proxy
            }
            
            df = df.rename(columns=column_mapping)
            
            # Clean symbols (remove .NS suffix)
            df['symbol'] = df['symbol'].str.replace('.NS', '', regex=False)
            
            # Calculate Pct Change (LTP vs Open)
            df['pChange'] = ((df['ltp'] - df['open']) / df['open']) * 100
            
            # Select and clean final columns
            df = df[['symbol', 'ltp', 'open', 'high', 'low', 'tradedValue', 'pChange']].copy()
            df = df.dropna(subset=['ltp', 'open', 'high', 'low'])
            df = df[df['ltp'] > 0]
            df = df[df['tradedValue'] > 0]
            
            # tradedValue (Volume) is used as a proxy for filtering, scale it for visibility
            df['tradedValue'] = df['tradedValue'] * df['ltp'] / 100 
            
            st.success(f"✅ Parsed {len(df)} stock prices for analysis.")
            return df
            
        except Exception as e:
            st.error(f"❌ Parsing error in market data: {str(e)}")
            return pd.DataFrame()

# ==================== DEFINEDGE API STUBS (PLACEHOLDERS) ====================
class DefinedgeAPI:
    """Handles authentication and order placement with the Definedge API."""
    
    BASE_AUTH_URL = "https://api.definedge.com/auth"
    BASE_API_URL = "https://api.definedge.com/api/v1"

    def __init__(self, token=None):
        self.api_token = None
        self.api_secret = None
        self.access_token = token
        self.client_id = None

    def set_credentials(self, api_token, api_secret, client_id):
        self.api_token = api_token
        self.api_secret = api_secret
        self.client_id = client_id

    def generate_otp(self):
        try:
            # Placeholder: Simulate OTP request
            response = requests.post(f"{self.BASE_AUTH_URL}/generateotp", 
                                     json={"api_token": self.api_token, "api_secret": self.api_secret})
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return data.get("otp_token"), data.get("message")
            return None, data.get("message", "OTP generation failed.")
        except Exception as e:
            return None, f"API Error: {e}"

    def verify_otp_and_authenticate(self, otp, otp_token):
        try:
            # Placeholder: Simulate authentication
            response = requests.post(f"{self.BASE_AUTH_URL}/gettoken", 
                                     json={"otp": otp, "otp_token": otp_token, "client_id": self.client_id})
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                self.access_token = data.get("api_session_key")
                return True, "Authentication Successful."
            return False, data.get("message", "Authentication failed.")
        except Exception as e:
            return False, f"API Error: {e}"
            
    def get_quotes(self, symbols):
        """Fetches live quotes for a list of symbols."""
        if not self.access_token: return []
        formatted_symbols = [self._format_symbol(s, 'NSE') for s in symbols]
        try:
            # Placeholder: Simulate quote fetching
            response = requests.post(f"{self.BASE_API_URL}/quotes", 
                                     headers={"Authorization": f"Bearer {self.access_token}"},
                                     json={"symbols": formatted_symbols})
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                # In a real scenario, this returns a list of quote objects
                return data.get("quotes", [])
            return []
        except Exception:
            # Return a generic failure, allowing the strategy to fall back to yfinance data
            return [] 

    def place_order(self, symbol, side, quantity, product_type, order_type="MARKET", price=0, trigger_price=0):
        if not self.access_token: return None, "Not authenticated."
        try:
            # Placeholder: Simulate order placement
            order_data = {
                "tradingsymbol": self._format_symbol(symbol, 'NSE'),
                "exchange": "NSE",
                "transactiontype": side, # 'BUY' or 'SELL'
                "quantity": int(quantity),
                "producttype": product_type, # e.g., 'MIS', 'CNC'
                "ordertype": order_type,
                "price": price,
                "triggerprice": trigger_price,
            }
            response = requests.post(f"{self.BASE_API_URL}/placeorder",
                                     headers={"Authorization": f"Bearer {self.access_token}"},
                                     json=order_data)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                order_id = data.get("order_id", f"MOCK-{datetime.now().timestamp()}")
                return order_id, "Order placed successfully."
            return None, data.get("message", "Order placement failed.")
        except Exception as e:
            return None, f"Order placement API Error: {e}"
            
    def _format_symbol(self, symbol, exchange):
        """Helper to format symbols as required by the broker (e.g., RELIANCE-EQ)"""
        if exchange == 'NSE' and not symbol.endswith('-EQ'):
            return f"{symbol}-EQ"
        return symbol

class DefinedgeReports:
    """Fetches various reports from the Definedge API."""
    BASE_API_URL = DefinedgeAPI.BASE_API_URL

    def __init__(self, access_token):
        self.access_token = access_token
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def orders(self):
        # Placeholder for fetching order book
        return [{"order_id": "O123", "symbol": "MOCK_ORDER", "status": "FILLED"}]

    def trades(self):
        # Placeholder for fetching trade book
        return [{"order_id": "MOCK-1", "trade_id": "T1", "symbol": "MOCK_TRADE", "price": 100.50}]

    def positions(self):
        # Placeholder for fetching current open positions
        return [{"symbol": "RELIANCE", "netQty": 10, "ltp": 2500, "buyAvg": 2490}]

    def limits(self):
        # Placeholder for fetching margin/limits
        return {"equity_margin": 500000.00, "exposure": 20000.00}

# ==================== CORE TRADING LOGIC ====================

def monitor_and_exit_positions(api, reports, open_trades, positions_df):
    """Checks open trades against live price and places exit orders if SL/Target is hit."""
    
    if not st.session_state.get('auto_exit_enabled', False) or open_trades.empty:
        return
        
    st.sidebar.caption("Auto Exit: ACTIVE")
    
    for _, trade in open_trades.iterrows():
        symbol = trade['symbol']
        entry_order_id = trade['entry_order_id']
        side = trade['side']
        
        # Find position to get live LTP
        matching_pos = positions_df[positions_df['symbol'] == symbol]
        if matching_pos.empty:
            continue # Skip if no position found
            
        ltp = matching_pos['ltp'].iloc[0]
        
        is_target_hit = (side == 'BUY' and ltp >= trade['target_price']) or \
                       (side == 'SELL' and ltp <= trade['target_price'])
                       
        is_sl_hit = (side == 'BUY' and ltp <= trade['current_stop_loss']) or \
                    (side == 'SELL' and ltp >= trade['current_stop_loss'])

        if is_target_hit or is_sl_hit:
            exit_side = 'SELL' if side == 'BUY' else 'BUY'
            
            # Use net quantity from broker position
            quantity_to_exit = abs(matching_pos['netQty'].iloc[0]) 

            st.sidebar.warning(f"🚨 Auto-Exit Triggered for {symbol}: {'TARGET' if is_target_hit else 'SL'}")
            
            # Place exit order (Using MARKET order for guaranteed exit)
            exit_order_id, msg = api.place_order(
                symbol=symbol, 
                side=exit_side, 
                quantity=quantity_to_exit, 
                product_type='MIS', 
                order_type='MARKET'
            )
            
            if exit_order_id:
                # Assuming immediate execution for simplicity (in a real app, this would be asynchronous)
                exit_price = ltp # Use current ltp as the exit price placeholder
                pnl = (exit_price - trade['entry_price']) * trade['quantity'] * (1 if side == 'BUY' else -1)
                
                update_result = update_trade_exit(
                    entry_order_id=entry_order_id,
                    exit_price=exit_price,
                    exit_order_id=exit_order_id,
                    pnl=pnl
                )
                if update_result['status'] == 'success':
                    st.success(f"✅ Auto-Exit complete for {symbol}. P&L: ₹{pnl:.2f}")
                    st.rerun()
                else:
                    st.error(update_result['message'])
            else:
                st.error(f"❌ Failed to place exit order for {symbol}: {msg}")

def check_order_executed(reports, order_id):
    """Check if the given order ID is present in the trade book (i.e., executed)."""
    # In a real API, this would poll the trade book or order book status
    time.sleep(1) # Simulate network delay
    # MOCK implementation: Assume successful placement means execution
    fill_price = 100.0 # Placeholder
    return True, fill_price

# ==================== TRADING STRATEGY ====================
class EnhancedTradingStrategy:
    
    def is_trading_time(self):
        now = datetime.now(IST)
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=0, microsecond=0)
        return start <= now <= end

    def filter_by_traded_value(self, df, min_value_cr):
        if df.empty:
            return df
        # Note: TradedValue is Volume*LTP/100 as proxy (scaled to match crore expectation)
        min_value = min_value_cr * 1e5 
        return df[df['tradedValue'] > min_value].copy()
    
    def generate_signals(self, df_market, api, min_traded_value_cr):
        """Generate signals using Market data for filtering and Definedge for live quotes"""
        signals = []
        
        if df_market.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        # 1. Filter using the market data (Volume/LTP proxy)
        df_filtered = self.filter_by_traded_value(df_market, min_traded_value_cr)
        
        if df_filtered.empty:
            st.warning(f"⚠️ No stocks with traded value > {min_traded_value_cr} crore")
            return signals, pd.DataFrame(), pd.DataFrame()
        
        symbols = df_filtered['symbol'].tolist()
        
        # 2. Get the absolutely latest LTP and price data from the Definedge API
        quotes = api.get_quotes(symbols)
        
        # If live quotes fail, use the yfinance data directly (df_filtered contains this)
        if quotes:
            for quote in quotes:
                symbol = quote.get('tradingsymbol', '').replace('-EQ', '')
                if symbol in df_filtered['symbol'].values:
                    idx = df_filtered[df_filtered['symbol'] == symbol].index[0]
                    ltp = float(quote.get('ltp', 0))
                    open_price = float(quote.get('open', 0))
                    
                    if ltp > 0 and open_price > 0:
                        df_filtered.loc[idx, 'ltp'] = ltp
                        df_filtered.loc[idx, 'open'] = open_price
                        df_filtered.loc[idx, 'high'] = float(quote.get('high', df_filtered.loc[idx, 'high']))
                        df_filtered.loc[idx, 'low'] = float(quote.get('low', df_filtered.loc[idx, 'low']))
                        df_filtered.loc[idx, 'pChange'] = ((ltp - open_price) / open_price) * 100

        # 3. Generate Signals based on updated prices
        df_gainers = df_filtered[df_filtered['pChange'] > 0].copy().sort_values('tradedValue', ascending=False)
        df_losers = df_filtered[df_filtered['pChange'] < 0].copy().sort_values('tradedValue', ascending=False)
        
        is_trading_time = self.is_trading_time()
        
        for _, stock in df_gainers.iterrows():
            if stock['ltp'] >= stock['high']: # Day High Breakout
                signals.append({
                    'symbol': stock['symbol'], 'type': 'BUY', 'ltp': float(stock['ltp']), 'open': float(stock['open']),
                    'target': float(stock['open']), 'initial_stop_loss': float(stock['low']),
                    'current_stop_loss': float(stock['low']), 'sl_distance': float(stock['ltp'] - stock['low']),
                    'day_low': float(stock['low']), 'day_high': float(stock['high']), 'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']), 'can_trade': is_trading_time,
                    'breakout_type': 'Day High Breakout'
                })
        
        for _, stock in df_losers.iterrows():
            if stock['ltp'] <= stock['low']: # Day Low Breakdown
                signals.append({
                    'symbol': stock['symbol'], 'type': 'SELL', 'ltp': float(stock['ltp']), 'open': float(stock['open']),
                    'target': float(stock['open']), 'initial_stop_loss': float(stock['high']),
                    'current_stop_loss': float(stock['high']), 'sl_distance': float(stock['high'] - stock['ltp']),
                    'day_low': float(stock['low']), 'day_high': float(stock['high']), 'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']), 'can_trade': is_trading_time,
                    'breakout_type': 'Day Low Breakdown'
                })
        
        return signals, df_gainers, df_losers


# ==================== UI HELPERS ====================

def display_signals_table(signals, trade_side, api, reports, open_trades_df, positions_df):
    """Renders the signal table and handles trade entry/exit logic."""
    
    if not signals:
        st.info(f"No {trade_side} signals generated yet.")
        return

    # Check for total risk and available cash
    risk_per_trade = st.session_state.get('risk_per_trade_percent', 1) / 100
    reports = DefinedgeReports(api.access_token)
    limits = reports.limits()
    available_margin = limits.get('equity_margin', 0)
    total_risk_limit = available_margin * risk_per_trade
    
    st.markdown(f"**Risk Profile:** Max Margin: **₹{available_margin:,.0f}**, Risk/Trade: **{st.session_state.get('risk_per_trade_percent', 1)}%** (₹{total_risk_limit:,.0f})")
    
    # Filter signals based on Max Signals setting
    max_signals = st.session_state.get('max_signals', 5)
    signals = signals[:max_signals]
    
    auto_entry_enabled = st.session_state.get('auto_entry_enabled', False)
    if auto_entry_enabled:
        st.warning(f"🤖 Auto Entry is **ON**. Will attempt to place {trade_side} orders.")

    for i, signal in enumerate(signals):
        # Calculate quantity based on risk management formula
        sl_distance = signal['sl_distance']
        if sl_distance <= 0:
            calc_qty = 0
        else:
            calc_qty = int(total_risk_limit / sl_distance)

        # Check if trade is already open in DB
        is_open = (open_trades_df['symbol'] == signal['symbol']) & (open_trades_df['side'] == trade_side)
        trade_status = "OPEN" if is_open.any() else "CLOSED"
        
        # Determine appropriate icon and color
        header_color = 'green' if trade_side == 'BUY' else 'red'
        status_icon = "✅" if trade_status == "CLOSED" else "⚠️"
        
        with st.expander(f"**{status_icon} {signal['symbol']} - {signal['breakout_type']}** | LTP: ₹{signal['ltp']:.2f}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            col1.metric("LTP", f"₹{signal['ltp']:.2f}", f"{signal['pct_change']:.2f}%")
            col1.metric("Day High", f"₹{signal['day_high']:.2f}")
            col1.metric("Day Low", f"₹{signal['day_low']:.2f}")
            
            col2.metric("Target Price (Open)", f"₹{signal['target']:.2f}")
            col2.metric("Initial Stop Loss", f"₹{signal['initial_stop_loss']:.2f}")
            col2.metric("Risk Distance", f"₹{sl_distance:.2f}")

            # --- Trade Entry Section ---
            
            with col3:
                st.markdown(f"**Trading Parameters ({trade_side})**")
                
                # Default values for inputs
                default_qty = calc_qty
                
                # Check if a position already exists for manual exit control
                matching_pos = positions_df[positions_df['symbol'] == signal['symbol']]
                has_position = not matching_pos.empty
                
                if trade_status == "OPEN":
                    st.success(f"Trade is currently **OPEN** in the database (Qty: {open_trades_df[is_open]['quantity'].iloc[0]})")
                    st.caption("Use Positions tab for live MTM tracking.")
                elif has_position:
                    st.warning("Position open at broker, but not tracked in DB.")
                
                if not is_open.any():
                    # Manual Override Form
                    with st.form(key=f'manual_entry_{signal["symbol"]}_{trade_side}'):
                        override_qty = st.number_input("Quantity Override", min_value=1, value=max(1, default_qty), step=1)
                        override_sl = st.number_input("Stop Loss Override", value=signal['initial_stop_loss'], step=0.05)
                        override_target = st.number_input("Target Override", value=signal['target'], step=0.05)
                        
                        trade_button = st.form_submit_button(f"🚀 Place Entry Order ({trade_side})", disabled=not signal['can_trade'])
                        
                        if trade_button:
                            entry_order_id, msg = api.place_order(
                                symbol=signal['symbol'],
                                side=trade_side,
                                quantity=override_qty,
                                product_type='MIS', # Day trade
                                order_type='LIMIT' if override_sl else 'MARKET', # Use limit if SL is defined, otherwise market
                                price=signal['ltp'], # Limit price near LTP for breakout trade
                                trigger_price=0 # N/A for Limit/Market
                            )
                            
                            if entry_order_id:
                                executed, fill_price = check_order_executed(reports, entry_order_id)
                                
                                if executed:
                                    trade_data = {
                                        'entry_order_id': entry_order_id,
                                        'symbol': signal['symbol'],
                                        'side': trade_side,
                                        'quantity': override_qty,
                                        'entry_price': fill_price,
                                        'target_price': override_target,
                                        'stop_loss': override_sl,
                                        'entry_time': datetime.now(IST).isoformat(),
                                        'status': 'OPEN'
                                    }
                                    save_result = save_trade_to_db(trade_data)
                                    if save_result['status'] == 'success':
                                        st.success(f"✅ Trade executed & saved! Qty: {override_qty}, Price: {fill_price:.2f}")
                                        st.rerun()
                                    else:
                                        st.error(f"DB Error: {save_result['message']}")
                                else:
                                    st.warning("Order placed but not yet executed.")
                            else:
                                st.error(f"❌ Order failed: {msg}")

                    # Auto Entry Logic (Outside Manual Form)
                    if auto_entry_enabled and signal['can_trade']:
                        # Only place if the current signal is NOT already in the DB (checked above)
                        if not is_open.any():
                            st.caption(f"Attempting Auto Entry for {signal['symbol']}...")
                            entry_order_id, msg = api.place_order(
                                symbol=signal['symbol'],
                                side=trade_side,
                                quantity=default_qty,
                                product_type='MIS',
                                order_type='MARKET' 
                            )
                            
                            if entry_order_id:
                                executed, fill_price = check_order_executed(reports, entry_order_id)
                                if executed:
                                    trade_data = {
                                        'entry_order_id': entry_order_id,
                                        'symbol': signal['symbol'],
                                        'side': trade_side,
                                        'quantity': default_qty,
                                        'entry_price': fill_price,
                                        'target_price': signal['target'],
                                        'stop_loss': signal['initial_stop_loss'],
                                        'entry_time': datetime.now(IST).isoformat(),
                                        'status': 'OPEN'
                                    }
                                    save_trade_to_db(trade_data)
                                    st.toast(f"🤖 Auto-Entry Success: {signal['symbol']}")
                                    st.rerun()
                                else:
                                    st.warning(f"Auto Order placed ({entry_order_id}), pending execution.")
                            else:
                                st.error(f"❌ Auto-Order Failed for {signal['symbol']}: {msg}")
                        else:
                            st.caption("Auto-entry skipped: Trade already open.")
                
                # --- Manual Exit Section ---
                if is_open.any():
                    # For manual exit, we check broker position
                    if has_position:
                        trade_record = open_trades_df[is_open].iloc[0]
                        exit_side = 'SELL' if trade_side == 'BUY' else 'BUY'
                        net_qty = abs(matching_pos['netQty'].iloc[0])

                        if st.button(f"🛑 Exit {net_qty} Qty Market", key=f"exit_{signal['symbol']}_{i}"):
                            exit_order_id, msg = api.place_order(
                                symbol=signal['symbol'],
                                side=exit_side,
                                quantity=net_qty, 
                                product_type='MIS', 
                                order_type='MARKET'
                            )
                            
                            if exit_order_id:
                                executed, exit_price = check_order_executed(reports, exit_order_id)
                                if executed:
                                    pnl = (exit_price - trade_record['entry_price']) * net_qty * (1 if trade_side == 'BUY' else -1)
                                    update_trade_exit(
                                        entry_order_id=trade_record['entry_order_id'],
                                        exit_price=exit_price,
                                        exit_order_id=exit_order_id,
                                        pnl=pnl
                                    )
                                    st.success(f"✅ Manual Exit complete! P&L: ₹{pnl:.2f}")
                                    st.rerun()
                                else:
                                    st.warning("Exit order placed, pending execution.")
                            else:
                                st.error(f"❌ Exit failed: {msg}")
                    else:
                        st.warning("Trade tracked in DB, but no open position found at broker.")

def show_authentication_page():
    """Renders the login/authentication interface."""
    
    st.title("Definedge API Authentication")
    
    if st.session_state.get('access_token'):
        st.success("You are already authenticated.")
        if st.button("Proceed to Dashboard"):
            st.session_state.authenticated = True
            st.rerun()
        return

    # Initialize API instance if not present
    if 'api_instance' not in st.session_state:
        st.session_state.api_instance = DefinedgeAPI()

    api = st.session_state.api_instance

    if 'otp_sent' not in st.session_state:
        st.session_state.otp_sent = False
        
    st.markdown("Enter your Definedge API credentials to start the session.")

    with st.form("credentials_form"):
        api_token = st.text_input("API Token (Key)")
        api_secret = st.text_input("API Secret", type="password")
        client_id = st.text_input("Client ID")
        
        submit_creds = st.form_submit_button("Save Credentials & Generate OTP")

        if submit_creds:
            if api_token and api_secret and client_id:
                api.set_credentials(api_token, api_secret, client_id)
                st.session_state.api_credentials = (api_token, api_secret, client_id)
                
                otp_token, msg = api.generate_otp()
                if otp_token:
                    st.session_state.otp_sent = True
                    st.session_state.temp_otp_token = otp_token
                    st.success(f"OTP generation successful. {msg}")
                    st.rerun()
                else:
                    st.error(f"OTP generation failed: {msg}")
            else:
                st.error("Please enter all credentials.")

    if st.session_state.otp_sent:
        with st.form("otp_form"):
            otp_input = st.text_input("Enter OTP Received on Email/SMS")
            verify_otp = st.form_submit_button("Verify OTP & Login")
            
            if verify_otp:
                if otp_input:
                    success, msg = api.verify_otp_and_authenticate(otp_input, st.session_state.temp_otp_token)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.access_token = api.access_token
                        st.session_state.api_instance = api # Store authenticated instance
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Please enter the OTP.")

def show_trading_dashboard():
    """Renders the main trading dashboard."""
    
    api = st.session_state.api_instance
    reports = DefinedgeReports(api.access_token)
    
    # --- Sidebar Setup ---
    st.sidebar.header("Strategy Settings")
    
    # Initialize session state for settings
    if 'min_traded_value_cr' not in st.session_state:
        st.session_state.min_traded_value_cr = 100 
    if 'auto_entry_enabled' not in st.session_state:
        st.session_state.auto_entry_enabled = False
    if 'auto_exit_enabled' not in st.session_state:
        st.session_state.auto_exit_enabled = False
    if 'max_signals' not in st.session_state:
        st.session_state.max_signals = 5
    if 'risk_per_trade_percent' not in st.session_state:
        st.session_state.risk_per_trade_percent = 1.0
    if 'last_data' not in st.session_state:
        st.session_state.last_data = pd.DataFrame()
        
    st.session_state.min_traded_value_cr = st.sidebar.slider(
        "Min Traded Value (Cr)", min_value=10, max_value=500, 
        value=st.session_state.min_traded_value_cr, step=10
    )
    st.session_state.max_signals = st.sidebar.number_input(
        "Max Signals to Show", min_value=1, max_value=20, 
        value=st.session_state.max_signals, step=1
    )
    st.session_state.risk_per_trade_percent = st.sidebar.number_input(
        "Risk/Trade (%) of Capital", min_value=0.1, max_value=5.0, 
        value=st.session_state.risk_per_trade_percent, step=0.1
    )
    
    st.sidebar.markdown("---")
    st.session_state.auto_entry_enabled = st.sidebar.checkbox("Enable Auto Entry", st.session_state.auto_entry_enabled)
    st.session_state.auto_exit_enabled = st.sidebar.checkbox("Enable Auto Exit", st.session_state.auto_exit_enabled)
    st.sidebar.markdown("---")

    # Fetch live data, positions, and open trades
    df_all = st.session_state.last_data
    open_trades_df = get_open_trades()
    positions_list = reports.positions()
    positions_df = pd.DataFrame(positions_list)

    scan = st.sidebar.button("🔍 Fetch Live", use_container_width=True, type="primary")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")

    if scan or auto_refresh:
        with st.spinner("Fetching market data (yfinance) and broker positions..."):
            fetcher = MarketDataFetcher()
            data = fetcher.fetch_all_securities()
            if data is None:
                new_df_all = pd.DataFrame()
            else:
                new_df_all = fetcher.parse_market_data(data)
            
            # Re-fetch positions on refresh
            positions_list = reports.positions()
            positions_df = pd.DataFrame(positions_list)
        
        if not new_df_all.empty:
            st.session_state.last_data = new_df_all
            df_all = new_df_all
        else:
            st.warning("New fetch returned no data.")

    if df_all.empty:
        if not (scan or auto_refresh):
            st.info("👆 Click 'Fetch Live' to start the analysis.")
        return

    # Check and exit positions first
    monitor_and_exit_positions(api, reports, open_trades_df, positions_df)

    # --- Signal Generation ---
    strategy = EnhancedTradingStrategy()
    signals, gainers, losers = strategy.generate_signals(df_all, api, st.session_state.min_traded_value_cr)

    # --- Header Metrics ---
    metrics_cols = st.columns(6)
    metrics_cols[0].metric("Total F&O Stocks", len(fetcher.fno_symbols))
    metrics_cols[1].metric("Filtered by Value", len(df_all))
    metrics_cols[2].metric("Total Gainers", len(gainers))
    metrics_cols[3].metric("Total Losers", len(losers))
    metrics_cols[4].metric("BUY Signals", len([s for s in signals if s['type'] == 'BUY']))
    metrics_cols[5].metric("SELL Signals", len([s for s in signals if s['type'] == 'SELL']))

    # --- Main Tabs ---
    buy_tab, sell_tab, overview_tab, orders_tab, positions_tab, db_tab = st.tabs(
        ["BUY Signals", "SELL Signals", "Market Overview", "Order Book", "Positions", "Trade Database"]
    )

    with buy_tab:
        st.subheader("Day High Breakouts (BUY)")
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        display_signals_table(buy_signals, 'BUY', api, reports, open_trades_df, positions_df)

    with sell_tab:
        st.subheader("Day Low Breakdowns (SELL)")
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        display_signals_table(sell_signals, 'SELL', api, reports, open_trades_df, positions_df)

    with overview_tab:
        st.subheader("Top Performers")
        st.markdown("### Top Gainers (Filtered)")
        st.dataframe(gainers[['symbol', 'ltp', 'open', 'high', 'low', 'pChange', 'tradedValue']], use_container_width=True, hide_index=True, column_config={"tradedValue": st.column_config.NumberColumn("Traded Value (Proxy)", format="₹%.2f")})
        st.markdown("### Top Losers (Filtered)")
        st.dataframe(losers[['symbol', 'ltp', 'open', 'high', 'low', 'pChange', 'tradedValue']], use_container_width=True, hide_index=True, column_config={"tradedValue": st.column_config.NumberColumn("Traded Value (Proxy)", format="₹%.2f")})

    with orders_tab:
        st.subheader("Broker Order Book")
        st.dataframe(reports.orders(), use_container_width=True, hide_index=True)

    with positions_tab:
        st.subheader("Open Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            # Calculate total MTM (Mock calculation)
            total_mtm = positions_df.apply(lambda x: (x['ltp'] - x['buyAvg']) * x['netQty'], axis=1).sum() if 'buyAvg' in positions_df.columns else 0
            st.metric("Total MTM (Mock)", f"₹{total_mtm:,.2f}", delta=f"{total_mtm/1000:,.2f}k")
        else:
            st.info("No open positions found.")

    with db_tab:
        st.subheader("Trade History Database (Supabase)")
        df_history = get_all_trades()
        
        if not df_history.empty:
            
            # P&L Metrics
            total_pnl = df_history['pnl'].sum()
            open_trades_count = df_history[df_history['status'] == 'OPEN'].shape[0]
            closed_trades_count = df_history[df_history['status'] == 'CLOSED'].shape[0]
            
            db_metrics = st.columns(4)
            db_metrics[0].metric("Total Trades", len(df_history))
            db_metrics[1].metric("Open Trades", open_trades_count)
            db_metrics[2].metric("Closed Trades", closed_trades_count)
            db_metrics[3].metric("Total Realized P&L", f"₹{total_pnl:,.2f}", delta_color=("inverse" if total_pnl < 0 else "normal"))

            # Filtering and display
            filter_option = st.selectbox("Filter Trades:", ['All', 'OPEN', 'CLOSED'])
            if filter_option != 'All':
                df_display = df_history[df_history['status'] == filter_option]
            else:
                df_display = df_history

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.markdown("### Database Management")
            
            # CSV Download/Upload (For backup/migration)
            col_csv_download, col_csv_upload = st.columns(2)
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            col_csv_download.download_button(
                "📥 Download History to CSV",
                csv_data,
                LOCAL_CSV_FILE,
                "text/csv",
                use_container_width=True
            )
            
            # --- Manual Exit DB Tool ---
            st.markdown("---")
            st.markdown("### Manual Database Exit Tool")
            
            open_db_trades = df_history[df_history['status'] == 'OPEN']
            if not open_db_trades.empty:
                exit_symbol_db = st.selectbox("Select OPEN Trade to Manually Exit (DB Record Only):", open_db_trades['symbol'].tolist(), key='db_exit_symbol')
                
                trade_to_exit = open_db_trades[open_db_trades['symbol'] == exit_symbol_db].iloc[0]
                live_price = api.get_quotes([exit_symbol_db]) # Get latest quote for accurate P&L
                exit_price_manual = live_price[0]['ltp'] if live_price else trade_to_exit['entry_price'] * 1.01 # Fallback
                
                st.info(f"Entry: ₹{trade_to_exit['entry_price']:.2f} | Quantity: {trade_to_exit['quantity']} | Side: {trade_to_exit['side']}")
                
                with st.form("manual_db_exit_form"):
                    manual_exit_price = st.number_input("Final Exit Price", value=exit_price_manual, step=0.01)
                    confirm_db_exit = st.form_submit_button("Force Close DB Record")
                    
                    if confirm_db_exit:
                        # Calculate PnL based on manual price
                        pnl = (manual_exit_price - trade_to_exit['entry_price']) * trade_to_exit['quantity'] * (1 if trade_to_exit['side'] == 'BUY' else -1)
                        
                        update_result = update_trade_exit(
                            entry_order_id=trade_to_exit['entry_order_id'],
                            exit_price=manual_exit_price,
                            exit_order_id='MANUAL_DB_EXIT',
                            pnl=pnl
                        )
                        if update_result['status'] == 'success':
                            st.success(f"✅ DB record for {exit_symbol_db} closed. P&L: ₹{pnl:.2f}")
                            st.rerun()
                        else:
                            st.error(update_result['message'])
            else:
                st.info("No open trades to manually close in the database.")
            
            st.markdown("---")
            # --- Reset DB Tool ---
            st.warning("DANGER ZONE: Completely wipe the Supabase trade history.")
            if st.checkbox("Confirm Database Reset", key='confirm_reset'):
                if st.button("🔴 RESET ALL TRADES DB", use_container_width=True):
                    reset_result = reset_trades_db()
                    if reset_result['status'] == 'success':
                        st.success(f"Database reset successful: {reset_result['message']}")
                        st.session_state.last_data = pd.DataFrame() # Clear session cache too
                        st.rerun()
                    else:
                        st.error(reset_result['message'])
        else:
            st.info("🔭 No trades in database yet. Place your first trade to start tracking!")


    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} | Market Data: YFinance | Broker: Definedge Securities")
    st.caption("⚠️ Educational purposes only. Not financial advice. Trade execution is mocked.")

# ==================== MAIN ====================
def main():
    if not st.session_state.get('authenticated', False):
        show_authentication_page()
    else:
        show_trading_dashboard()

if __name__ == "__main__":
    main()
