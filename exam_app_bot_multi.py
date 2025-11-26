import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv  # pip install python-dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to the path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from api_helper import NorenApiPy
except ImportError:
    st.error("api_helper.py not found. Please ensure it is in the parent directory.")

# --- Configuration & Security ---
load_dotenv() # Load credentials from .env file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Credentials
USER_ID = os.getenv("FLATTRADE_USER_ID")
FLATTRADE_PASSWORD = os.getenv("FLATTRADE_PASSWORD")
USER_SESSION = os.getenv("FLATTRADE_SESSION")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Flattrade API Initialization ---
@st.cache_resource
def init_flattrade_api():
    if not USER_ID or not FLATTRADE_PASSWORD:
        st.error("Missing Credentials! Please check your .env file.")
        return None
    
    api = NorenApiPy()
    # Attempt login
    try:
        ret = api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=FLATTRADE_PASSWORD)
        if ret:
            logging.info("API Session Active")
            return api
        else:
            st.error("Failed to set API session.")
            return None
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

class FnOBreakoutStrategy:
    def __init__(self):
        self.stop_loss_pct = 0.01
        self.target_pct = 0.03
        self.trailing_stop_pct = 0.01
        self.position_symbol = None
        self.product_type = 'C'
        self.screened_stocks = []
        self.stock_bias = {}
        self.entry_price = None
        self.current_position = None
        self.position_size = 0
        self.stop_loss = None
        self.target = None
        self.day_high = None
        self.day_low = None
        self.trailing_stop = None
        self.current_date = None
        
    def reset_daily_data(self, current_date):
        if self.current_date != current_date:
            self.current_date = current_date
            self.screened_stocks = []
            self.stock_bias = {}
            self.day_high = None
            self.day_low = None

    def check_breakout_entry(self, df, current_candle, day_high, day_low, symbol):
        """
        Updated Logic:
        1. Heavy Volume: > 5x of 20-period MA
        2. Big Move: Current Candle Range (High-Low) > 2x of 20-period Avg Range
        3. Breakout: Price breaks previous Day High/Low
        """
        if len(df) < 22: return None
        
        bias = self.stock_bias.get(symbol)
        if not bias: return None

        # --- DATA PREPARATION ---
        prev_20_candles = df.iloc[-21:-1]
        
        # 1. Volume Check
        avg_volume_20 = prev_20_candles['Volume'].mean()
        if avg_volume_20 == 0: avg_volume_20 = 1
        current_volume = current_candle['Volume']
        
        if current_volume < (5 * avg_volume_20):
            return None
        
        # 2. Volatility Check (Using RANGE, not Body)
        # We use Range (High - Low) because it is stable during a live candle
        prev_ranges = prev_20_candles['High'] - prev_20_candles['Low']
        avg_range_20 = prev_ranges.mean()
        
        current_range = current_candle['High'] - current_candle['Low']
        
        # Avoid division by zero or extremely small ranges
        min_range_threshold = current_candle['Close'] * 0.0005 
        effective_avg_range = max(avg_range_20, min_range_threshold)
        
        if current_range < (2 * effective_avg_range):
            return None
            
        # 3. Breakout Check
        # Recalculate Day High/Low excluding current candle
        today_data = df[df.index.date == current_candle.name.date()]
        if len(today_data) > 1:
            prior_data = today_data[:-1]
            prior_day_high = prior_data['High'].max()
            prior_day_low = prior_data['Low'].min()
        else:
            # Fallback if it's the first candle (risky, but uses provided H/L)
            prior_day_high = day_high
            prior_day_low = day_low
        
        # --- BUY LOGIC ---
        if bias == 'BUY':
            if current_candle['Close'] > current_candle['Open']: # Still want a green candle
                if current_candle['High'] > prior_day_high:
                    return 'BUY'
        
        # --- SELL LOGIC ---
        elif bias == 'SELL':
            if current_candle['Close'] < current_candle['Open']: # Still want a red candle
                if current_candle['Low'] < prior_day_low:
                    return 'SELL'
        
        return None
    
    def update_trailing_stop(self, current_price, current_day_high, current_day_low):
        if self.current_position == 'long':
            new_trailing = current_day_high * (1 - self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing
        elif self.current_position == 'short':
            new_trailing = current_day_low * (1 + self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing < self.trailing_stop:
                self.trailing_stop = new_trailing
    
    def enter_position(self, signal_type, entry_price, day_high, day_low, quantity, symbol):
        self.current_position = 'long' if signal_type == 'BUY' else 'short'
        self.entry_price = entry_price
        self.position_size = quantity
        self.day_high = day_high
        self.day_low = day_low
        self.position_symbol = symbol
        self.trailing_stop = None
        
        if self.current_position == 'long':
            stop_loss_val = entry_price * (1 - self.stop_loss_pct)
            self.stop_loss = max(stop_loss_val, day_low)
            self.target = entry_price * (1 + self.target_pct)
        else:
            stop_loss_val = entry_price * (1 + self.stop_loss_pct)
            self.stop_loss = min(stop_loss_val, day_high)
            self.target = entry_price * (1 - self.target_pct)

# --- Helper Functions ---

@st.cache_data
def get_fno_stocks_list():
    """
    Tries to load stocks from 'NSE_Equity.csv'. 
    Falls back to a default list if the file is missing.
    """
    default_list = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 
        'KOTAKBANK', 'LT', 'AXISBANK', 'ITC', 'BAJFINANCE', 'MARUTI', 'TATAMOTORS'
    ]
    
    csv_path = 'NSE_Equity.csv'
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names to handle variations like 'Symbol', 'symbol', 'TRADINGSYMBOL'
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Look for a valid symbol column
            target_col = None
            for col in ['symbol', 'tradingsymbol', 'trading symbol']:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                # Return the top 50 to avoid hitting API rate limits during scanning
                return df[target_col].dropna().unique().tolist()[:50]
        except Exception as e:
            logging.error(f"Error reading CSV: {e}")
    
    return default_list

def get_token_from_isin(api, symbol, exchange="NSE"):
    try:
        # Optimization: Just use searchscrip directly
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result:
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ":
                    return item.get('token')
        return None
    except:
        return None

def load_market_data(api, symbol, exchange="NSE", days=5):
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if not token: return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_time = start_date.strftime("%d-%m-%Y") + " 09:15:00"
        
        # Use simple end time
        hist_data = api.get_time_price_series(exchange=exchange, token=token, starttime=start_time, interval='1')
        
        if not hist_data: return None
        
        df = pd.DataFrame(hist_data)
        df['Date'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
        df = df.rename(columns={'into':'Open','inth':'High','intl':'Low','intc':'Close','intv':'Volume'})
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype({'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':int})
        df.set_index('Date', inplace=True)
        return df.sort_index()
    except Exception as e:
        return None

def execute_trade(api, signal_type, quantity, symbol, product_type='C'):
    try:
        buy_or_sell = 'B' if signal_type == 'BUY' else 'S'
        return api.place_order(buy_or_sell=buy_or_sell, product_type=product_type, exchange='NSE',
                             tradingsymbol=f"{symbol}-EQ", quantity=quantity, discloseqty=0,
                             price_type='MKT', retention='DAY')
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

def check_and_sync_positions(api, strategy):
    """Syncs API positions with Strategy State"""
    try:
        positions = api.get_positions()
        if positions:
            for pos in positions:
                netqty = int(pos.get('netqty', 0))
                if netqty != 0:
                    symbol = pos.get('tsym', '').replace('-EQ', '')
                    avg_price = float(pos.get('netavgprc', 0))
                    
                    if strategy.position_symbol != symbol:
                        # Logic to restore state from an existing position
                        strategy.position_symbol = symbol
                        strategy.current_position = 'long' if netqty > 0 else 'short'
                        strategy.entry_price = avg_price
                        strategy.position_size = abs(netqty)
                        
                        # Set default stops since we missed the entry
                        if strategy.current_position == 'long':
                            strategy.target = avg_price * 1.03
                            strategy.stop_loss = avg_price * 0.99
                        else:
                            strategy.target = avg_price * 0.97
                            strategy.stop_loss = avg_price * 1.01
                    return
        # If loop finishes and no active position found
        strategy.current_position = None
        strategy.position_symbol = None
    except:
        pass

# --- Optimized Parallel Scanner ---

def scan_single_stock(api, symbol, rvol_threshold, ma_window, min_consecutive):
    """Worker function for threading"""
    try:
        df = load_market_data(api, symbol, days=2) # Only need 2 days for scanning
        if df is None or len(df) == 0: return None
        
        current_date = datetime.now().date()
        today_df = df[df.index.date == current_date]
        
        if len(today_df) < ma_window: return None
        
        # RVOL Logic
        vol_ma = today_df['Volume'].rolling(window=ma_window, min_periods=ma_window).mean()
        heavy_mask = today_df['Volume'] > (rvol_threshold * vol_ma)
        
        # Calculate max consecutive streak
        longest = 0
        curr = 0
        for is_heavy in heavy_mask.fillna(False):
            if is_heavy:
                curr += 1
                longest = max(longest, curr)
            else:
                curr = 0
                
        if longest >= min_consecutive:
            last_row = today_df.iloc[-1]
            last_ma = vol_ma.iloc[-1]
            rvol = (last_row['Volume'] / last_ma) if last_ma > 0 else 0
            
            open_p = today_df['Open'].iloc[0]
            ltp = last_row['Close']
            chg = ((ltp - open_p) / open_p) * 100
            
            return {
                'Symbol': symbol,
                'LTP': ltp,
                'Change': chg,
                'RVOL': rvol,
                'Streak': longest,
                'TotalVol': int(today_df['Volume'].sum()),
                'Bias': 'BUY' if chg >= 0 else 'SELL'
            }
    except Exception:
        return None

def get_heavy_volume_stocks_threaded(api, rvol_threshold=3.0, ma_window=20, min_consecutive=10, limit=10):
    stock_list = get_fno_stocks_list()
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {executor.submit(scan_single_stock, api, sym, rvol_threshold, ma_window, min_consecutive): sym for sym in stock_list}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    
    results.sort(key=lambda x: (x['RVOL'], x['Streak']), reverse=True)
    return results[:limit]

# --- Charting ---

def create_chart(df, entry=None, sl=None, tgt=None, trail=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Price', 'Volume'), 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # Volume
    colors = ['red' if r['Open'] > r['Close'] else 'green' for i, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    # Lines
    if entry: fig.add_hline(y=entry, line_dash="dash", line_color="blue", annotation_text="Entry", row=1, col=1)
    if sl: fig.add_hline(y=sl, line_dash="solid", line_color="red", annotation_text="SL", row=1, col=1)
    if tgt: fig.add_hline(y=tgt, line_dash="solid", line_color="green", annotation_text="TGT", row=1, col=1)
    if trail: fig.add_hline(y=trail, line_dash="dot", line_color="orange", annotation_text="Trail", row=1, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="FnO Scanner Pro", layout="wide", initial_sidebar_state="expanded")
    
    api = init_flattrade_api()
    if not api: return

    if 'strategy' not in st.session_state: 
        st.session_state.strategy = FnOBreakoutStrategy()
    strategy = st.session_state.strategy
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("Strategy Settings", expanded=True):
            stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 5.0, 1.0, 0.1)
            target_pct = st.slider("Target (%)", 0.1, 10.0, 3.0, 0.1)
            trailing_stop_pct = st.slider("Trailing Stop (%)", 0.5, 3.0, 1.0, 0.1)
            
            # Update strategy object
            strategy.stop_loss_pct = stop_loss_pct / 100
            strategy.target_pct = target_pct / 100
            strategy.trailing_stop_pct = trailing_stop_pct / 100
            
        with st.expander("Scanner Settings"):
            rvol_thresh = st.slider("Min RVOL", 1.5, 10.0, 3.0)
            min_cons = st.slider("Min Streak (Min)", 5, 60, 10)
            
        with st.expander("Execution"):
            auto_trading = st.toggle("Enable Auto Trading", False)
            qty = st.number_input("Quantity", 1, 1000, 1)
            
        if st.button("🚀 Run Scanner", type="primary"):
            st.session_state.run_scanning = True

    # --- Header ---
    st.title("⚡ FnO Volatility Breakout")
    
    # Limits Check
    try:
        limits = api.get_limits()
        if limits and limits.get('stat') == 'Ok':
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash Available", f"₹{float(limits.get('cash', 0)):,.2f}")
            c2.metric("Margin Used", f"₹{float(limits.get('marginused', 0)):,.2f}")
            c3.metric("API Status", "🟢 Online")
    except:
        st.warning("Could not fetch limits.")

    # --- Sync & Date Reset ---
    current_date = datetime.now().date()
    strategy.reset_daily_data(current_date)
    check_and_sync_positions(api, strategy)

    # --- Scanning Logic ---
    if st.session_state.get('run_scanning'):
        with st.spinner("Scanning Market (Parallel)..."):
            heavy_results = get_heavy_volume_stocks_threaded(api, rvol_thresh, 20, min_cons)
            
            strategy.screened_stocks = [x['Symbol'] for x in heavy_results]
            strategy.stock_bias = {x['Symbol']: x['Bias'] for x in heavy_results}
            st.session_state.heavy_results = {x['Symbol']: x for x in heavy_results}
            
            st.session_state.run_scanning = False
            if not heavy_results:
                st.toast("No stocks found matching criteria.")

    # --- Layout ---
    col_dash, col_chart = st.columns([1, 2])
    
    with col_dash:
        st.subheader("📡 Watchlist")
        if strategy.screened_stocks:
            res_list = list(st.session_state.heavy_results.values())
            st.dataframe(pd.DataFrame(res_list)[['Symbol', 'Bias', 'LTP', 'RVOL']], hide_index=True)
        else:
            st.info("Scanner results will appear here.")
            
        st.subheader("📍 Active Position")
        if strategy.current_position:
            pos_df = pd.DataFrame([{
                'Symbol': strategy.position_symbol,
                'Type': strategy.current_position.upper(),
                'Entry': strategy.entry_price,
                'Target': f"{strategy.target:.2f}",
                'SL': f"{strategy.stop_loss:.2f}"
            }])
            st.dataframe(pos_df, hide_index=True)
            
            # Manual Exit Button
            if st.button("🚨 Emergency Exit"):
                # Simplified Exit Logic
                try:
                    positions = api.get_positions()
                    for p in positions:
                        if p['tsym'] == f"{strategy.position_symbol}-EQ" and int(p['netqty']) != 0:
                            qty_exit = abs(int(p['netqty']))
                            side = 'S' if int(p['netqty']) > 0 else 'B'
                            api.place_order(buy_or_sell=side, product_type='C', exchange='NSE',
                                          tradingsymbol=p['tsym'], quantity=qty_exit, discloseqty=0,
                                          price_type='MKT', retention='DAY')
                            st.success("Exit Order Placed")
                            time_module.sleep(1)
                            st.rerun()
                except Exception as e:
                    st.error(f"Exit failed: {e}")

    with col_chart:
        sel_stock = None
        if strategy.current_position:
            sel_stock = strategy.position_symbol
        elif strategy.screened_stocks:
            sel_stock = st.selectbox("View Chart", strategy.screened_stocks)
            
        if sel_stock:
            df = load_market_data(api, sel_stock)
            if df is not None:
                # Live Check Logic
                curr_candle = df.iloc[-1]
                day_h = df[df.index.date == current_date]['High'].max()
                day_l = df[df.index.date == current_date]['Low'].min()
                
                # Check Signal
                if not strategy.current_position:
                    sig = strategy.check_breakout_entry(df, curr_candle, day_h, day_l, sel_stock)
                    if sig:
                        st.toast(f"SIGNAL: {sig} on {sel_stock}", icon="🔥")
                        if auto_trading:
                            resp = execute_trade(api, sig, qty, sel_stock)
                            if resp and resp.get('stat') == 'Ok':
                                strategy.enter_position(sig, curr_candle['Close'], day_h, day_l, qty, sel_stock)
                                st.rerun()
                
                # Check Exit
                elif strategy.current_position:
                    ltp = curr_candle['Close']
                    strategy.update_trailing_stop(ltp, day_h, day_l)
                    
                    exit_needed = False
                    if strategy.current_position == 'long':
                        if ltp >= strategy.target or ltp <= strategy.stop_loss: exit_needed = True
                        if strategy.trailing_stop and ltp <= strategy.trailing_stop: exit_needed = True
                    else:
                        if ltp <= strategy.target or ltp >= strategy.stop_loss: exit_needed = True
                        if strategy.trailing_stop and ltp >= strategy.trailing_stop: exit_needed = True
                        
                    if exit_needed and auto_trading:
                        # Perform Exit
                        st.toast(f"Auto-Exiting {sel_stock}")
                        # (Reuse exit logic or call specific function)
                
                # Plot
                st.plotly_chart(create_chart(df.tail(60), strategy.entry_price, strategy.stop_loss, strategy.target, strategy.trailing_stop), use_container_width=True)

    # Auto Refresh
    time_module.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
