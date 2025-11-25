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

# --- Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = "df24a9522687879f168b54077701691ecf23e1020703248688e90324fcba2d6b"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@3"

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flattrade API
@st.cache_resource
def init_flattrade_api():
    api = NorenApiPy()
    api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=FLATTRADE_PASSWORD)
    return api

class FnOBreakoutStrategy:
    def __init__(self):
        self.stop_loss_pct = 0.01  # 1% stop loss
        self.target_pct = 0.03  # 3% target
        self.trailing_stop_pct = 0.01  # 1% trailing stop
        self.position_symbol = None
        self.product_type = 'C'  # Default to C
        self.screened_stocks = [] # List of stock symbols
        self.stock_bias = {} # Dictionary to store bias: {'INFY': 'BUY', 'TCS': 'SELL'}
        self.entry_price = None
        self.current_position = None  # 'long' or 'short'
        self.position_size = 0
        self.stop_loss = None
        self.target = None
        self.day_high = None
        self.day_low = None
        self.trailing_stop = None
        self.current_date = None
        
    def reset_daily_data(self, current_date):
        """Reset daily data for new trading day"""
        if self.current_date != current_date:
            self.current_date = current_date
            self.screened_stocks = []
            self.stock_bias = {}
            self.day_high = None
            self.day_low = None

    def calculate_opening_candle_metrics(self, df):
        """
        Helper for UI display - keeps track of basic candle data
        """
        if len(df) == 0:
            return None, None, None, None
            
        opening_data = df[(df.index.time >= time(9, 15)) & (df.index.time <= time(9, 29))]
        avg_volume_opening = 0
        if len(opening_data) > 0:
            avg_volume_opening = opening_data['Volume'].mean()
        
        return avg_volume_opening, 0, False, None
    
    def check_breakout_entry(self, df, current_candle, day_high, day_low, symbol):
        """
        Check for breakout entry signals based on Gainers (High) and Losers (Low)
        Updated Logic:
        1. Heavy Volume: > 5x of 20-period Moving Average
        2. Big Move: Candle Body > 2x of 20-period Average Body
        3. Breakout: Price breaks previous Day High/Low
        """
        # We need at least 21 candles (20 previous + 1 current) for MA calculation
        if len(df) < 22:
            return None
        
        # Determine Bias (Direction) based on screening (Gainer vs Loser)
        bias = self.stock_bias.get(symbol)
        if not bias:
            return None

        # --- DATA PREPARATION ---
        # Get the previous 20 candles (excluding the current one)
        prev_20_candles = df.iloc[-21:-1]
        
        # Calculate Averages (Volume & Body)
        avg_volume_20 = prev_20_candles['Volume'].mean()
        
        # Body size = abs(Close - Open)
        prev_bodies = (prev_20_candles['Close'] - prev_20_candles['Open']).abs()
        avg_body_20 = prev_bodies.mean()
        
        # --- CURRENT CANDLE METRICS ---
        current_volume = current_candle['Volume']
        current_body = abs(current_candle['Close'] - current_candle['Open'])
        current_high = current_candle['High']
        current_low = current_candle['Low']
        
        # --- 1. HEAVY VOLUME CHECK ---
        # Logic: Volume must be > 5 times the 20-period Moving Average
        if avg_volume_20 == 0: avg_volume_20 = 1 # Prevent div/0
        
        volume_confirmed = current_volume >= (5 * avg_volume_20)
        
        if not volume_confirmed:
            return None
        
        # --- 2. BIG MOVE (CANDLE) CHECK ---
        # Logic: Current Body must be > 2 times the 20-period Average Body
        # We add a small minimum threshold to avoid triggering on tiny candles during flat markets
        min_body_threshold = current_candle['Close'] * 0.0005 # 0.05% minimum move
        effective_avg_body = max(avg_body_20, min_body_threshold)
        
        big_move_confirmed = current_body >= (2 * effective_avg_body)
        
        if not big_move_confirmed:
            return None
            
        # --- 3. BREAKOUT DIRECTION CHECK ---
        
        # Recalculate Day High/Low excluding the current candle to identify a TRUE breakout
        # (The 'day_high' passed in might already include the current candle's high)
        today_data = df[df.index.date == current_candle.name.date()]
        if len(today_data) > 1:
            prior_data = today_data[:-1] # Exclude current
            prior_day_high = prior_data['High'].max()
            prior_day_low = prior_data['Low'].min()
        else:
            # If it's the very first candle, we can't really breakout, but let's use the passed values
            prior_day_high = day_high
            prior_day_low = day_low
        
        # --- BUY LOGIC (For Top Gainers) ---
        if bias == 'BUY':
            # Must be a green candle
            if current_candle['Close'] > current_candle['Open']:
                # Entry: Price Breaks Previous Day High
                if current_high > prior_day_high:
                    return 'BUY'
        
        # --- SELL LOGIC (For Top Losers) ---
        elif bias == 'SELL':
            # Must be a red candle
            if current_candle['Close'] < current_candle['Open']:
                # Entry: Price Breaks Previous Day Low
                if current_low < prior_day_low:
                    return 'SELL'
        
        return None
    
    def update_trailing_stop(self, current_price, current_day_high, current_day_low):
        """Update trailing stop loss based on day high/low"""
        if self.current_position == 'long':
            # Trailing stop: 1% down from day high
            new_trailing = current_day_high * (1 - self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing
        
        elif self.current_position == 'short':
            # Trailing stop: 1% up from day low
            new_trailing = current_day_low * (1 + self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing < self.trailing_stop:
                self.trailing_stop = new_trailing
    
    def check_exit_conditions(self, current_price, current_day_high, current_day_low):
        """Check exit conditions (Target, SL, Trailing)"""
        if not self.current_position:
            return False, None
        
        # Update trailing stop
        self.update_trailing_stop(current_price, current_day_high, current_day_low)
        
        if self.current_position == 'long':
            if current_price >= self.target: return True, "Target hit"
            if current_price <= self.stop_loss: return True, "Stop loss hit"
            if self.trailing_stop and current_price <= self.trailing_stop: return True, "Trailing stop hit"
        
        elif self.current_position == 'short':
            if current_price <= self.target: return True, "Target hit"
            if current_price >= self.stop_loss: return True, "Stop loss hit"
            if self.trailing_stop and current_price >= self.trailing_stop: return True, "Trailing stop hit"
        
        return False, None
    
    def enter_position(self, signal_type, entry_price, day_high, day_low, quantity, symbol):
        """Enter a position"""
        self.current_position = 'long' if signal_type == 'BUY' else 'short'
        self.entry_price = entry_price
        self.position_size = quantity
        self.day_high = day_high
        self.day_low = day_low
        self.trailing_stop = None
        self.position_symbol = symbol
        
        if self.current_position == 'long':
            # Stop loss: max of 1% or day low
            stop_loss_pct = entry_price * (1 - self.stop_loss_pct)
            self.stop_loss = max(stop_loss_pct, day_low)
            self.target = entry_price * (1 + self.target_pct)
        
        else:  # short
            # Stop loss: min of 1% or day high
            stop_loss_pct = entry_price * (1 + self.stop_loss_pct)
            self.stop_loss = min(stop_loss_pct, day_high)
            self.target = entry_price * (1 - self.target_pct)
    
    def exit_position(self):
        """Exit current position"""
        self.current_position = None
        self.entry_price = None
        self.position_size = 0
        self.stop_loss = None
        self.target = None
        self.trailing_stop = None
        self.position_symbol = None

# --- Helper Functions ---

def get_fno_stocks_list():
    """Get list of FnO stocks from CSV"""
    try:
        equity_df = load_nse_equity_data()
        if equity_df is not None:
            possible_cols = ['Symbol', 'Tradingsymbol', 'Trading Symbol', 'symbol', 'tradingsymbol']
            symbol_col = None
            for col in possible_cols:
                if col in equity_df.columns:
                    symbol_col = col
                    break
            if symbol_col:
                return equity_df[symbol_col].dropna().unique().tolist()[:100] # Scan more for top g/l
        return ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL']
    except Exception as e:
        return ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']

def load_nse_equity_data():
    try:
        return pd.read_csv('NSE_Equity.csv')
    except Exception as e:
        return None

def get_token_from_isin(api, symbol, exchange="NSE"):
    try:
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result and len(result['values']) > 0:
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ":
                    return item.get('token')
            return result['values'][0].get('token')
        return None
    except:
        return None

def load_market_data(api, symbol, exchange="NSE"):
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if not token: return None
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        start_time = start_date.strftime("%d-%m-%Y") + " 09:15:00"
        end_time = end_date.strftime("%d-%m-%Y") + " " + end_date.strftime("%H:%M:%S")
        
        hist_data = api.get_time_price_series(exchange=exchange, token=token, starttime=start_time, endtime=end_time, interval='1')
        
        if not hist_data: return None
        
        data_list = []
        for item in hist_data:
            data_list.append({
                'Date': pd.to_datetime(item['time'], format='%d-%m-%Y %H:%M:%S'),
                'Open': float(item['into']), 'High': float(item['inth']),
                'Low': float(item['intl']), 'Close': float(item['intc']),
                'Volume': int(item.get('intv', 0))
            })
        
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        return df.sort_index()
    except:
        return None

def get_live_price(api, symbol, exchange="NSE"):
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if token:
            live_data = api.get_quotes(exchange=exchange, token=token)
            if live_data and live_data.get('stat') == 'Ok':
                return float(live_data.get('lp', 0))
        return None
    except:
        return None

def execute_trade(api, signal_type, quantity, symbol, product_type='C'):
    try:
        buy_or_sell = 'B' if signal_type == 'BUY' else 'S'
        # Added discloseqty=0 here as well for consistency
        return api.place_order(buy_or_sell=buy_or_sell, product_type=product_type, exchange='NSE',
                             tradingsymbol=f"{symbol}-EQ", quantity=quantity, discloseqty=0,
                             price_type='MKT', retention='DAY')
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

def check_and_sync_positions(api, strategy):
    try:
        positions = api.get_positions()
        if positions:
            for pos in positions:
                netqty = int(pos.get('netqty', 0))
                if netqty != 0:
                    symbol = pos.get('tsym', '').replace('-EQ', '')
                    avg_price = float(pos.get('netavgprc', 0))
                    product_type = pos.get('prd', 'C')
                    position_type = 'long' if netqty > 0 else 'short'
                    
                    if strategy.position_symbol != symbol:
                        # Only sync to strategy if it's the one we want to track actively on the chart
                        strategy.position_symbol = symbol
                        strategy.product_type = product_type
                        strategy.current_position = position_type
                        strategy.entry_price = avg_price
                        strategy.position_size = abs(netqty)
                        
                        # Load data to set stops roughly
                        df = load_market_data(api, symbol)
                        day_high = avg_price
                        day_low = avg_price
                        if df is not None:
                            today_data = df[df.index.date == datetime.now().date()]
                            if len(today_data) > 0:
                                day_high = today_data['High'].max()
                                day_low = today_data['Low'].min()
                        
                        if position_type == 'long':
                            strategy.stop_loss = max(avg_price * (1 - strategy.stop_loss_pct), day_low)
                            strategy.target = avg_price * (1 + strategy.target_pct)
                        else:
                            strategy.stop_loss = min(avg_price * (1 + strategy.stop_loss_pct), day_high)
                            strategy.target = avg_price * (1 - strategy.target_pct)
                    
                    return True, {'symbol': symbol, 'type': position_type, 'qty': abs(netqty)}
        return False, None
    except:
        return False, None

def get_heavy_volume_stocks(api, rvol_threshold=3.0, ma_window=20, min_consecutive=10, limit=10):
    try:
        stock_list = get_fno_stocks_list()
        current_date = datetime.now().date()
        results = []
        for symbol in stock_list:
            df = load_market_data(api, symbol)
            if df is None or len(df) == 0:
                continue
            today_df = df[df.index.date == current_date]
            if len(today_df) < ma_window:
                continue
            vol_ma = today_df['Volume'].rolling(window=ma_window, min_periods=ma_window).mean()
            heavy = today_df['Volume'] > (rvol_threshold * vol_ma)
            longest = 0
            curr = 0
            for flag in heavy.fillna(False).tolist():
                if flag:
                    curr += 1
                    if curr > longest:
                        longest = curr
                else:
                    curr = 0
            if longest >= min_consecutive:
                last_row = today_df.iloc[-1]
                last_ma = vol_ma.iloc[-1]
                rvol = (last_row['Volume'] / last_ma) if last_ma and not np.isnan(last_ma) and last_ma != 0 else 0
                open_p = today_df['Open'].iloc[0]
                ltp = last_row['Close']
                chg = ((ltp - open_p) / open_p) * 100 if open_p else 0
                bias = 'BUY' if chg >= 0 else 'SELL'
                results.append({
                    'Symbol': symbol,
                    'LTP': ltp,
                    'Change': chg,
                    'RVOL': rvol,
                    'Streak': longest,
                    'TotalVol': int(today_df['Volume'].sum()),
                    'Bias': bias
                })
        results.sort(key=lambda x: (x['RVOL'], x['Streak']), reverse=True)
        return results[:limit]
    except Exception as e:
        logging.error(f"Error fetching heavy volume: {e}")
        return []

def create_chart(df, entry_point=None, stop_loss=None, target=None, trailing_stop=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Price', 'Volume'), row_heights=[0.8, 0.2])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    if entry_point: fig.add_hline(y=entry_point, line_dash="dash", line_color="blue", annotation_text="Entry", row=1, col=1)
    if stop_loss: fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="Stop Loss", row=1, col=1)
    if target: fig.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Target", row=1, col=1)
    if trailing_stop: fig.add_hline(y=trailing_stop, line_dash="dot", line_color="orange", annotation_text="Trailing", row=1, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="FnO Heavy Volume Screener", layout="wide")
    
    try:
        api = init_flattrade_api()
        if 'strategy' not in st.session_state: st.session_state.strategy = FnOBreakoutStrategy()
        strategy = st.session_state.strategy
        
        # Test Connection & Get Limits
        limits_resp = api.get_limits()
        if limits_resp.get('stat') != 'Ok':
            st.error("API Connection Failed")
            return
            
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Bot Configuration")
        st.subheader("Strategy Parameters")
        stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 5.0, 1.0, 0.1)
        target_pct = st.slider("Target (%)", 0.1, 10.0, 3.0, 0.1)
        trailing_stop_pct = st.slider("Trailing Stop (%)", 0.5, 3.0, 1.0, 0.1)
        strategy.stop_loss_pct = stop_loss_pct / 100
        strategy.target_pct = target_pct / 100
        strategy.trailing_stop_pct = trailing_stop_pct / 100
        st.subheader("Position Sizing")
        sizing_mode = st.radio("Mode", ["Fixed Quantity", "Fixed Amount"])
        if sizing_mode == "Fixed Quantity":
            quantity = st.number_input("Qty", 1, value=1)
            trade_amount = None
        else:
            trade_amount = st.number_input("Amount (₹)", 1000, value=10000)
            quantity = None
        st.subheader("Heavy Volume Scan")
        rvol_threshold = st.slider("RVOL threshold", 1.0, 10.0, 3.0, 0.5)
        min_consecutive = st.slider("Min consecutive minutes", 5, 60, 10, 1)
        scan_limit = st.number_input("Max results", 1, 50, value=10)
        st.divider()
        auto_trading = st.toggle("Enable Auto Trading", False)
        auto_refresh = st.toggle("🔄 Auto Refresh (5s)", value=False)
        if st.button("🔍 Scan Heavy Volume", type="primary"):
            st.session_state.run_scanning = True
    
    # --- Top Dashboard (Funds) ---
    st.title("⚡ FnO Heavy Volume Screener")
    
    # Account Summary
    cash_avail = float(limits_resp.get('cash', 0))
    margin_used = float(limits_resp.get('marginused', 0))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Available Cash", f"₹{cash_avail:,.2f}")
    c2.metric("🔒 Margin Used", f"₹{margin_used:,.2f}")
    c3.metric("📊 Auto-Trading", "Active" if auto_trading else "Inactive", delta_color="normal")
    st.divider()

    # --- Sync & Update ---
    current_date = datetime.now().date()
    strategy.reset_daily_data(current_date)
    check_and_sync_positions(api, strategy)
    
    # --- Scanning Logic ---
    if st.session_state.get('run_scanning'):
        with st.spinner("Scanning heavy volume since market open..."):
            heavy_results = get_heavy_volume_stocks(api, rvol_threshold=rvol_threshold, ma_window=20, min_consecutive=min_consecutive, limit=int(scan_limit))
            strategy.screened_stocks = [x['Symbol'] for x in heavy_results]
            strategy.stock_bias = {x['Symbol']: x['Bias'] for x in heavy_results}
            st.session_state.heavy_results = {x['Symbol']: x for x in heavy_results}
            st.session_state.run_scanning = False
            st.success(f"Screened {len(strategy.screened_stocks)} stocks.")

    # --- Tabs for Main Interface ---
    tab_monitor, tab_positions, tab_orders, tab_trades = st.tabs([
        "📈 Strategy Monitor", "📍 Open Positions & Exit", "📋 Order Book", "📒 Trade Book"
    ])
    
    # TAB 1: Strategy Monitor
    with tab_monitor:
        col_scan, col_live = st.columns([1, 2])
        
        with col_scan:
            st.subheader("Watchlist")
            if strategy.screened_stocks:
                w_data = []
                heavy_map = st.session_state.get('heavy_results', {})
                for sym in strategy.screened_stocks:
                    m = heavy_map.get(sym)
                    if m:
                        w_data.append({
                            "Symbol": sym,
                            "Bias": "🟢 BUY" if strategy.stock_bias[sym] == 'BUY' else "🔴 SELL",
                            "LTP": f"{m['LTP']:.2f}",
                            "% Change": f"{m['Change']:.2f}",
                            "RVOL": f"{m['RVOL']:.2f}",
                            "Streak (min)": m['Streak'],
                            "Total Vol": f"{m['TotalVol']:,}"
                        })
                    else:
                        token = get_token_from_isin(api, sym)
                        quote_data = {}
                        if token:
                            res = api.get_quotes(exchange='NSE', token=token)
                            if res and res.get('stat') == 'Ok':
                                quote_data = res
                        ltp = float(quote_data.get('lp', 0))
                        w_data.append({
                            "Symbol": sym,
                            "Bias": "🟢 BUY" if strategy.stock_bias[sym] == 'BUY' else "🔴 SELL",
                            "LTP": f"{ltp:.2f}"
                        })
                st.dataframe(pd.DataFrame(w_data), hide_index=True, use_container_width=True)
            else:
                st.info("Run scan to populate.")
                
        with col_live:
            st.subheader("Live Chart")
            # Select logic
            if strategy.current_position:
                sel_stock = strategy.position_symbol
                st.info(f"Tracking Active: {sel_stock}")
            elif strategy.screened_stocks:
                sel_stock = st.selectbox("Select Stock", strategy.screened_stocks)
            else:
                sel_stock = None
                
            if sel_stock:
                df = load_market_data(api, sel_stock)
                if df is not None and len(df) > 0:
                    today_data = df[df.index.date == current_date]
                    if len(today_data) > 0:
                        day_h = today_data['High'].max()
                        day_l = today_data['Low'].min()
                        curr = df.iloc[-1]
                        
                        # Check Entry
                        if not strategy.current_position:
                            signal = strategy.check_breakout_entry(df, curr, day_h, day_l, sel_stock)
                            if signal:
                                st.toast(f"{signal} Signal on {sel_stock}!")
                                if auto_trading:
                                    qty = int(trade_amount / curr['Close']) if sizing_mode == "Fixed Amount" else quantity
                                    res = execute_trade(api, signal, qty, sel_stock)
                                    if res and res.get('stat') == 'Ok':
                                        strategy.enter_position(signal, curr['Close'], day_h, day_l, qty, sel_stock)
                                        st.rerun()
                        
                        st.plotly_chart(create_chart(df.tail(100), strategy.entry_price, strategy.stop_loss, strategy.target, strategy.trailing_stop), use_container_width=True)
                    else:
                        st.warning("No intraday data.")
    
    # TAB 2: Positions & Auto-Exit
    with tab_positions:
        st.subheader("📍 Open Positions Manager")
        
        # Fetch all positions from API
        positions = api.get_positions()
        open_positions = [p for p in positions if int(p.get('netqty', 0)) != 0] if positions else []
        
        if open_positions:
            pos_data = []
            
            for p in open_positions:
                sym = p['tsym']
                netqty = int(p['netqty'])
                avg_prc = float(p['netavgprc'])
                prd = p['prd']
                
                # Get LTP
                ltp = float(p.get('lp', 0))
                if ltp == 0:
                    ltp = get_live_price(api, sym.replace('-EQ', '')) or avg_prc
                
                # Calculate P&L
                if netqty > 0: # Long
                    pnl = (ltp - avg_prc) * netqty
                    bias = "LONG"
                    # Exit Conditions based on sliders
                    calc_target = avg_prc * (1 + strategy.target_pct)
                    calc_sl = avg_prc * (1 - strategy.stop_loss_pct)
                    exit_hit = (ltp >= calc_target) or (ltp <= calc_sl)
                else: # Short
                    pnl = (avg_prc - ltp) * abs(netqty)
                    bias = "SHORT"
                    calc_target = avg_prc * (1 - strategy.target_pct)
                    calc_sl = avg_prc * (1 + strategy.stop_loss_pct)
                    exit_hit = (ltp <= calc_target) or (ltp >= calc_sl)
                
                pos_data.append({
                    "Symbol": sym,
                    "Side": bias,
                    "Qty": netqty,
                    "Avg": f"{avg_prc:.2f}",
                    "LTP": f"{ltp:.2f}",
                    "Target": f"{calc_target:.2f}",
                    "SL": f"{calc_sl:.2f}",
                    "P&L": f"{pnl:.2f}",
                    "Exit Hit?": "⚠️ YES" if exit_hit else "No"
                })
                
                # AUTO EXIT LOGIC
                if auto_trading and exit_hit:
                    st.toast(f"⚡ Auto-Exiting {sym}...")
                    trantype = 'S' if netqty > 0 else 'B'
                    
                    # --- FIXED LINE BELOW (Added discloseqty=0) ---
                    api.place_order(buy_or_sell=trantype, product_type=prd, exchange='NSE', 
                                    tradingsymbol=sym, quantity=abs(netqty), discloseqty=0,
                                    price_type='MKT', retention='DAY')
                    time_module.sleep(1)
                    st.rerun()
            
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
            
            if st.button("Refresh Positions"):
                st.rerun()
        else:
            st.info("No open positions.")

    # TAB 3: Order Book
    with tab_orders:
        st.subheader("📋 Today's Orders")
        orders = api.get_order_book()
        if orders:
            ord_df = pd.DataFrame(orders)
            # Filter relevant columns for display
            cols = ['norenordno', 'tsym', 'trantype', 'qty', 'prc', 'status', 'rejreason', 'ordertimestamp']
            show_cols = [c for c in cols if c in ord_df.columns]
            st.dataframe(ord_df[show_cols], use_container_width=True)
        else:
            st.info("No orders placed today.")

    # TAB 4: Trade Book
    with tab_trades:
        st.subheader("📒 Executed Trades")
        trades = api.get_trade_book()
        if trades:
            trd_df = pd.DataFrame(trades)
            cols = ['norentm', 'tsym', 'trantype', 'qty', 'flprc', 'exchangetimestamp']
            show_cols = [c for c in cols if c in trd_df.columns]
            st.dataframe(trd_df[show_cols], use_container_width=True)
        else:
            st.info("No trades executed today.")

    # --- AUTO REFRESH LOGIC ---
    if auto_refresh:
        time_module.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
