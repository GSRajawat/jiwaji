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

# --- Flattrade API Credentials (Using credentials from the previously modified file) ---
USER_SESSION = "09add3026f8c427c3dc3fbe0b53e48c09cda332a6bb09ca3bca58625b407559d"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@3"

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouh_OiMjwqGU"

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
        self.stop_loss_pct = 0.01
        self.target_pct = 0.03
        self.trailing_stop_pct = 0.01
        self.open_positions = {}
        self.product_type = 'C'
        self.screened_stocks = []
        self.stock_bias = {}
        self.current_date = None
        self.rejected_symbols = set()
        
    def reset_daily_data(self, current_date):
        """Reset daily data for new trading day"""
        if self.current_date != current_date:
            self.current_date = current_date
            self.screened_stocks = []
            self.stock_bias = {}
            self.rejected_symbols = set()

    def update_position_parameters(self):
        """Recalculate SL and Target for open positions using current strategy percentages."""
        for symbol, pos in self.open_positions.items():
            entry_price = pos['entry_price']
            
            if pos['type'] == 'long':
                pos['stop_loss'] = entry_price * (1 - self.stop_loss_pct)
                pos['target'] = entry_price * (1 + self.target_pct)
            
            elif pos['type'] == 'short':
                pos['stop_loss'] = entry_price * (1 + self.stop_loss_pct)
                pos['target'] = entry_price * (1 - self.target_pct)

    def check_breakout_entry(self, df, current_ltp, open_price, day_high, day_low, symbol):
        """
        Logic: Buy/Sell if LTP confirms the candle color AND breaks the day range midpoint.
        This remains the existing trade execution signal.
        """
        current_day_high = day_high
        current_day_low = day_low
        
        price_range = current_day_high - current_day_low
        
        if price_range <= 0 or open_price == 0 or current_ltp == 0:
            return None
            
        midpoint = current_day_low + (price_range / 2)
        
        # BUY condition (LTP above open AND above midpoint)
        if (current_ltp > open_price) and (current_ltp > midpoint):
            return 'BUY'
        
        # SELL condition (LTP below open AND below midpoint)
        elif (current_ltp < open_price) and (current_ltp < midpoint):
            return 'SELL'
            
        return None
    
    def update_trailing_stop(self, symbol, current_price, current_day_high, current_day_low):
        """Update trailing stop loss based on day high/low"""
        if symbol not in self.open_positions: return
        
        pos = self.open_positions[symbol]
        
        if pos['type'] == 'long':
            new_trailing = current_day_high * (1 - self.trailing_stop_pct)
            if pos['trailing_stop'] is None or new_trailing > pos['trailing_stop']:
                pos['trailing_stop'] = new_trailing
        
        elif pos['type'] == 'short':
            new_trailing = current_day_low * (1 + self.trailing_stop_pct)
            if pos['trailing_stop'] is None or new_trailing < pos['trailing_stop']:
                pos['trailing_stop'] = new_trailing
    
    def check_exit_conditions(self, symbol, current_price, current_day_high, current_day_low):
        """Check exit conditions (Target, SL, Trailing) for a specific symbol"""
        if symbol not in self.open_positions:
            return False, None
        
        pos = self.open_positions[symbol]
        
        self.update_trailing_stop(symbol, current_price, current_day_high, current_day_low)
        
        if pos['type'] == 'long':
            if current_price >= pos['target']: return True, "Target hit"
            if current_price <= pos['stop_loss']: return True, "Stop loss hit"
            if pos['trailing_stop'] and current_price <= pos['trailing_stop']: return True, "Trailing stop hit"
        
        elif pos['type'] == 'short':
            if current_price <= pos['target']: return True, "Target hit"
            if current_price >= pos['stop_loss']: return True, "Stop loss hit"
            if pos['trailing_stop'] and current_price >= pos['trailing_stop']: return True, "Trailing stop hit"
        
        return False, None
    
    def enter_position(self, signal_type, entry_price, day_high, day_low, quantity, symbol, product_type):
        """Enter a position."""
        position_type = 'long' if signal_type == 'BUY' else 'short'
        
        if position_type == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            target = entry_price * (1 + self.target_pct)
        
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            target = entry_price * (1 - self.target_pct)
            
        self.open_positions[symbol] = {
            'type': position_type,
            'entry_price': entry_price,
            'size': quantity,
            'stop_loss': stop_loss,
            'target': target,
            'trailing_stop': None,
            'day_high': day_high,
            'day_low': day_low,
            'product_type': product_type
        }
    
    def exit_position(self, symbol):
        """Exit current position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

# --- Helper Functions (Updated) ---

def load_nse_equity_data():
    try:
        return pd.read_csv('NSE_Equity.csv')
    except Exception as e:
        return None

def get_fno_stocks_list():
    """Get list of F&O stocks."""
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
                return equity_df[symbol_col].dropna().unique().tolist()
        
        return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    except Exception as e:
        logging.error(f"Critical error during stock list generation: {e}")
        return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']


def get_token_from_isin(api, symbol, exchange="NSE"):
    try:
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result and len(result['values']) > 0:
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ" or item.get('symname') == symbol:
                    return item.get('token')
            return result['values'][0].get('token')
        return None
    except:
        return None

def get_historical_data_for_volume(api, symbol, days_back=5, interval='1'):
    """
    Fetches historical 1-minute data for volume analysis.
    The days_back is flexible to allow for both 3-day average (for screening)
    and 1-day/recent history (for smart bias check).
    """
    try:
        token = get_token_from_isin(api, symbol)
        if not token: return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back) 
        
        start_time_str = start_date.strftime("%d-%m-%Y") + " 09:00:00"
        end_time_str = end_date.strftime("%d-%m-%Y") + " " + end_date.strftime("%H:%M:%S")

        hist_data = api.get_time_price_series(exchange='NSE', token=token, starttime=start_time_str, endtime=end_time_str, interval=interval)
        
        if not hist_data or (isinstance(hist_data, dict) and hist_data.get('stat') != 'Ok'):
            return None
        
        data_list = []
        for item in hist_data:
            data_list.append({
                'Date': pd.to_datetime(item['time'], format='%d-%m-%Y %H:%M:%S'),
                'Volume': int(item.get('intv', 0)),
                'Open': float(item['into']),
                'Close': float(item['intc']),
                'High': float(item['inth']),
                'Low': float(item['intl']),
            })
        
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        return df.sort_index()
    except Exception as e:
        return None

# --- NEW COMPLEX BIAS CHECK FUNCTION ---
def check_smart_bias_filter(api, symbol, min_candle_trade_value):
    """
    Applies the complex filtering logic based on the latest 1-minute candle data
    and average metrics from past candles.
    
    Returns True if the candle meets the volatility/volume criteria, False otherwise.
    The directional bias (LTP vs Open) is handled by the caller.
    """
    try:
        # Fetch the last 15 minutes of 1-minute data (days_back=1 is enough for intraday)
        hist_df = get_historical_data_for_volume(api, symbol, days_back=1, interval='1') 
        
        if hist_df is None or hist_df.empty: return False

        # Ensure we have enough data (at least 10 candles for volume avg, 2 for range avg)
        if len(hist_df) < 10: return False 

        # --- 1. Current Candle Metrics (last completed 1-min candle) ---
        current_candle = hist_df.iloc[-1]
        
        current_vol = current_candle['Volume']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        current_range = current_high - current_low
        current_mid_price = (current_high + current_low) / 2
        
        # Current Trade Value for the latest COMPLETED 1-minute candle
        current_trade_value = current_vol * current_mid_price
        
        # --- 2. Historical Averages ---
        
        # Avg Volume of last 10 candles (excluding current one for clean average)
        last_10_candles_vol = hist_df.iloc[-11:-1]['Volume']
        avg_vol_10 = last_10_candles_vol.mean()
        
        # Avg Range of last 2 candles (excluding current one)
        last_2_range_candles = hist_df.iloc[-3:-1] 
        avg_range_2 = (last_2_range_candles['High'] - last_2_range_candles['Low']).mean()
        
        if avg_vol_10 <= 0 or avg_range_2 <= 0: return False
        
        # --- 3. APPLY RULES ---
        
        # Rule 2 (Common Filter): Trade value of current candle > 5 crore (or configurable amount)
        if current_trade_value < min_candle_trade_value:
            return False
            
        # Rule 2 (Volume Condition OR Range Condition)
        
        # Volume Condition: current candle volume > 3 * average volume of last 10 candles
        volume_condition = current_vol > (3 * avg_vol_10)
        
        # Range Condition: High - Low range of current candle > 2 * average of High - Low range of last 2 candles
        range_condition = current_range > (2 * avg_range_2)
        
        # The complex bias is TRUE if either condition is met
        return volume_condition or range_condition
            
    except Exception as e:
        logging.error(f"Error in complex bias check for {symbol}: {e}")
        return False


def get_mid_price(quote_data):
    """Calculates the mid-price (Average of Best Bid and Best Ask)."""
    try:
        bp1 = float(quote_data.get('bp1', 0.0))
        sp1 = float(quote_data.get('sp1', 0.0))
        
        if bp1 > 0 and sp1 > 0:
            return round((bp1 + sp1) / 2, 2)
        
        ltp = float(quote_data.get('lp', 0.0))
        return ltp
    except Exception as e:
        return 0.0

def get_live_price(api, symbol, exchange="NSE"):
    """Helper to get a single live price and critical day metrics from quote data."""
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if token:
            live_data = api.get_quotes(exchange=exchange, token=token)
            if live_data and live_data.get('stat') == 'Ok':
                ltp = float(live_data.get('lp', 0.0))
                day_h = float(live_data.get('h', 0.0))
                day_l = float(live_data.get('l', 0.0))
                open_p = float(live_data.get('o', 0.0))
                volume = float(live_data.get('v', 0.0))
                return ltp, day_h, day_l, open_p, volume, live_data
        return None, None, None, None, None, None
    except:
        return None, None, None, None, None, None

def execute_trade(api, buy_or_sell, quantity, symbol, product_type, limit_price=0.0):
    """Executes a trade using a Limit Order based on the calculated limit_price."""
    try:
        if not isinstance(quantity, int) or quantity <= 0:
            return {'stat': 'Not_Ok', 'emsg': f'Invalid quantity: {quantity}'}
            
        formatted_price = f"{limit_price:.2f}"
        
        return api.place_order(buy_or_sell=buy_or_sell, product_type=product_type, exchange='NSE',
                             tradingsymbol=f"{symbol}-EQ", quantity=quantity, discloseqty=0,
                             price_type='LMT', price=formatted_price, retention='DAY')
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return {'stat': 'Not_Ok', 'emsg': f"API Exception: {e}"}


# --- UPDATED VOLUME SCREENING FUNCTION (3-Day Average) ---
def get_heavy_volume_stocks(api, volume_multiplier, max_stocks, max_price, min_trade_value):
    """
    Screens for stocks with current cumulative volume significantly higher than the 3-day 
    average volume for the same time of day.
    """
    current_date = datetime.now().date()
    
    last_minute = datetime.now().replace(second=0, microsecond=0)
    if datetime.now().second < 30 and datetime.now().minute != 15 and datetime.now().hour >= 9:
         last_minute -= timedelta(minutes=1)
         
    time_filter = last_minute.time()
    
    stock_list = get_fno_stocks_list()
    results = []
    
    scan_limit = 100 # Internal performance limit
    
    for symbol in stock_list[:scan_limit]: 
        try:
            # 1. Get Live Data 
            ltp, _, _, open_p, current_volume, quote_data = get_live_price(api, symbol)

            if ltp is None or ltp == 0 or open_p == 0: continue
            
            # Basic Filters
            if ltp > max_price: continue
            trade_value = current_volume * ltp
            if trade_value < min_trade_value: continue

            # 2. Get Historical Volume Data (Need minimum 5 days back to ensure 3 trading days)
            hist_df = get_historical_data_for_volume(api, symbol, days_back=5) 
            if hist_df is None or hist_df.empty: continue
            
            # 3. Calculate Average Volume for Time of Day (using last 3 days)
            
            past_days_df = hist_df[hist_df.index.date != current_date]
            
            # Get the last 3 trading days with data
            historical_dates = past_days_df.index.normalize().unique()[-3:] 
            
            if len(historical_dates) < 3: continue 

            daily_volumes = []
            for date in historical_dates:
                day_data = past_days_df.loc[past_days_df.index.date == date]
                
                # Filter data for this day up to the current time of day
                time_filtered_data = day_data[day_data.index.time <= time_filter]
                
                if not time_filtered_data.empty:
                    daily_volumes.append(time_filtered_data['Volume'].sum())

            if not daily_volumes: continue
            
            average_volume = np.mean(daily_volumes)
            
            if average_volume == 0: continue
            
            # 4. Apply Volume Filter
            volume_ratio = current_volume / average_volume
            
            if volume_ratio >= volume_multiplier:
                
                # 5. Simple Bias Detection (LTP vs Day Open)
                chg = ((ltp - open_p) / open_p) * 100
                bias = 'BUY' if ltp >= open_p else 'SELL'
                
                results.append({
                    'Symbol': symbol,
                    'LTP': ltp,
                    'Change': chg,
                    'Bias': bias,
                    'CurrentVolume': int(current_volume),
                    'AvgVolume': int(average_volume),
                    'VolumeRatio': volume_ratio,
                    'TradeValue': trade_value
                })

        except Exception as e:
            continue

    # Sort by Volume Ratio
    results.sort(key=lambda x: x['VolumeRatio'], reverse=True)
    
    return results[:max_stocks]


# (Omitted: check_and_sync_positions, exit_all_positions, create_chart - they remain the same)
def check_and_sync_positions(api, strategy):
    try:
        positions = api.get_positions()
        if not positions or (isinstance(positions, dict) and positions.get('stat') != 'Ok'):
            logging.warning("Failed to fetch positions from API. Keeping internal state unchanged.")
            return False, set(strategy.open_positions.keys())
        api_open_symbols = set()
        for pos in positions if positions else []:
            netqty = int(pos.get('netqty', 0))
            symbol = pos.get('tsym', '').replace('-EQ', '')
            if netqty != 0:
                api_open_symbols.add(symbol)
                if symbol not in strategy.open_positions:
                    avg_price = float(pos.get('netavgprc', 0))
                    product_type = pos.get('prd', 'C')
                    position_type = 'long' if netqty > 0 else 'short'
                    _, day_h, day_l, _, _, _ = get_live_price(api, symbol)
                    day_h = day_h if day_h != 0 else avg_price * 1.01
                    day_l = day_l if day_l != 0 else avg_price * 0.99
                    if position_type == 'long':
                        stop_loss = avg_price * (1 - strategy.stop_loss_pct)
                        target = avg_price * (1 + strategy.target_pct)
                    else:
                        stop_loss = avg_price * (1 + strategy.stop_loss_pct)
                        target = avg_price * (1 - strategy.target_pct)
                    strategy.open_positions[symbol] = {
                        'type': position_type, 'entry_price': avg_price, 'size': abs(netqty),
                        'stop_loss': stop_loss, 'target': target, 'trailing_stop': None,
                        'day_high': day_h, 'day_low': day_l, 'product_type': product_type
                    }
        symbols_to_remove = [sym for sym in strategy.open_positions if sym not in api_open_symbols]
        for sym in symbols_to_remove:
            logging.info(f"Position for {sym} closed/exited via broker. Removing from internal tracking.")
            del strategy.open_positions[sym]
        return True, api_open_symbols
    except Exception as e:
        logging.error(f"Error syncing positions: {e}")
        return False, None

def exit_all_positions(api, strategy, exit_message_placeholder):
    if not strategy.open_positions:
        exit_message_placeholder.info("No open positions to exit.")
        return
    symbols_to_exit_now = list(strategy.open_positions.keys())
    for sym in symbols_to_exit_now:
        pos = strategy.open_positions[sym]
        ltp, _, _, _, _, quote_data = get_live_price(api, sym)
        if ltp is None or ltp == 0:
            exit_message_placeholder.warning(f"Could not get live price for {sym}. Skipping manual exit.")
            continue
        trantype = 'S' if pos['type'] == 'long' else 'B'
        product_type_to_use = pos['product_type']
        exit_limit_price = get_mid_price(quote_data)
        if exit_limit_price == 0.0:
             exit_message_placeholder.error(f"❌ Exit Order FAILED for {sym}. Cannot calculate mid-price for Limit Order.")
             continue
        exit_message_placeholder.warning(f"Manual Exit: Placing {product_type_to_use} Limit Order for {sym} @ {exit_limit_price:.2f}...")
        res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)
        if res and res.get('stat') == 'Ok':
            exit_message_placeholder.success(f"✅ Manual Exit Limit Order Placed for {sym}. (Order No: {res.get('norenordno')}). Waiting for broker sync...")
        else:
            exit_message_placeholder.error(f"❌ Manual Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
    time_module.sleep(1)
    check_and_sync_positions(api, strategy)

def create_chart(df, position_data=None):
    if df is None or df.empty: return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price', 'Volume'), row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    if position_data:
        if position_data.get('entry_price'): fig.add_hline(y=position_data['entry_price'], line_dash="dash", line_color="blue", annotation_text="Entry", row=1, col=1)
        if position_data.get('stop_loss'): fig.add_hline(y=position_data['stop_loss'], line_dash="dash", line_color="red", annotation_text="Stop Loss", row=1, col=1)
        if position_data.get('target'): fig.add_hline(y=position_data['target'], line_dash="dash", line_color="green", annotation_text="Target", row=1, col=1)
        if position_data.get('trailing_stop'): fig.add_hline(y=position_data['trailing_stop'], line_dash="dot", line_color="orange", annotation_text="Trailing", row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig


# --- MAIN APP ---

def main():
    st.set_page_config(page_title="Heavy Volume Midpoint Breakout Bot", layout="wide")
    
    try:
        api = init_flattrade_api()
        if 'strategy' not in st.session_state: st.session_state.strategy = FnOBreakoutStrategy()
        strategy = st.session_state.strategy
        
        limits_resp = api.get_limits()
        if limits_resp.get('stat') != 'Ok':
            st.error("API Connection Failed: Could not fetch limits. Please check credentials or network.")
            return
            
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Bot Configuration")
        
        st.subheader("Flexible Buy/Sell Variables (Exit Conditions)")
        # This section corresponds to 'flexible buy sell variables' requested by user
        stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 5.0, 1.0, 0.1, key="sl_pct")
        target_pct = st.slider("Target (%)", 0.1, 10.0, 3.0, 0.1, key="tg_pct")
        trailing_stop_pct = st.slider("Trailing Stop (%)", 0.5, 3.0, 1.0, 0.1, key="ts_pct")
        
        strategy.stop_loss_pct = stop_loss_pct / 100
        strategy.target_pct = target_pct / 100
        strategy.trailing_stop_pct = trailing_stop_pct / 100
        
        st.subheader("Volume Screening Parameters")
        
        volume_multiplier = st.slider("Volume Multiplier (X times 3-day Avg)", 2.0, 10.0, 3.0, 0.5, key="vol_mult_slider", help="Current cumulative volume must be this many times the 3-day average volume for the same time.")
        
        scan_limit = st.number_input("Max Stocks to Track (5-20)", min_value=5, max_value=20, value=10, step=1, key="scan_limit")
        
        max_price = st.number_input("Stock Price Limit (₹)", min_value=10.0, max_value=10000.0, value=500.0, step=10.0, key="max_prc")
        min_trade_value = st.number_input("Min Daily Trade Value (₹)", min_value=0, value=5000000, step=1000000, key="min_trade_value", help="Minimum total value of shares traded today (Volume * LTP).")
        
        st.subheader("Smart Bias Filters (Entry Conditions)")
        
        enable_smart_bias_filter = st.checkbox("Enable Complex Candle Bias Filter", True, key="complex_bias_toggle", help="Applies volume/range checks on the final entry candle (Rule 2).")
        # Defaulting to 5 Crore (50,000,000) for the single candle trade value filter
        min_candle_trade_value = st.number_input("Min Single Candle Trade Value (₹)", min_value=1000000, value=50000000, step=1000000, key="min_candle_tv", help="Minimum trade value (Price * Volume) for the latest completed 1-minute candle to qualify for entry.")
        
        st.subheader("Trade Settings")
        
        st.radio("New Entry Product Priority", ['MIS (Intraday) -> CNC (Delivery)'], index=0, disabled=True, help="Bot will attempt MIS first, then CNC if MIS fails.")
            
        max_positions = st.number_input("Max Open Positions Limit", min_value=1, value=5, step=1, key="max_pos_limit")

        st.subheader("Position Sizing")
        sizing_mode = st.radio("Mode", ["Fixed Quantity", "Fixed Amount"], key="size_mode")
        if sizing_mode == "Fixed Quantity":
            quantity = st.number_input("Qty", 1, value=1, key="qty_input")
            trade_amount = None
        else:
            trade_amount = st.number_input("Amount (₹)", 1000, value=10000, key="amt_input")
            quantity = None
            
        st.divider()
        
        # Original buttons (kept)
        auto_scan = st.toggle("🤖 Auto Scan (on load)", False, key="auto_scan_toggle", help="Automatically scans and generates the watchlist.")
        auto_trading = st.toggle("Enable Auto Trading (Multi-Stock)", False, key="auto_trade_toggle")
        
        close_time_str = st.selectbox("Auto-Close Time", ["15:15:00"], index=0, key="close_time_select")
        current_time = datetime.now().time()
        close_time = datetime.strptime(close_time_str, "%H:%M:%S").time()
        is_close_time = current_time >= close_time
        
        st.caption(f"Current Time: {current_time.strftime('%H:%M:%S')}")
        
        if st.button("🔴 Exit All Positions Now", disabled=not strategy.open_positions, type="secondary"):
            st.session_state.manual_exit_all = True

        st.divider()
        auto_refresh = st.toggle("🔄 Auto Refresh (5s)", value=False, key="auto_refresh_toggle")
        
        if not auto_scan:
            # Kept original button for manual scan (updated label)
            if st.button("🔍 Scan Heavy Volume Stocks", type="primary", key="scan_button"):
                st.session_state.run_scanning = True
    
    # --- Top Dashboard (Funds) ---
    st.title("🧪 Heavy Volume Midpoint Breakout Bot")
    
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
    strategy.update_position_parameters()
    
    # --- Manual Exit All Handler ---
    exit_placeholder = st.empty()
    if st.session_state.get('manual_exit_all', False):
        with exit_placeholder:
            exit_all_positions(api, strategy, st.status("Exiting all positions...", expanded=True))
        st.session_state.manual_exit_all = False
        st.rerun() 

    # --- Scanning Logic ---
    should_run_scan = st.session_state.get('run_scanning', False) or (auto_scan and not strategy.screened_stocks)

    if should_run_scan:
        with st.spinner(f"Scanning Heavy Volume Stocks (Multiplier: {volume_multiplier}x, Max: {scan_limit})..."):
            heavy_results = get_heavy_volume_stocks(api, volume_multiplier, int(scan_limit), max_price, min_trade_value)
            
            strategy.screened_stocks = [x['Symbol'] for x in heavy_results]
            strategy.stock_bias = {x['Symbol']: x['Bias'] for x in heavy_results}
            st.session_state.heavy_results = {x['Symbol']: x for x in heavy_results}
            st.session_state.run_scanning = False 
            st.success(f"Screened {len(strategy.screened_stocks)} stocks meeting volume criteria.")

    # --- Tabs for Main Interface ---
    tab_monitor, tab_positions, tab_orders, tab_trades = st.tabs([
        "📈 Strategy Monitor", "📍 Open Positions & Exit", "📋 Order Book", "📒 Trade Book"
    ])
    
    # TAB 1: Strategy Monitor
    with tab_monitor:
        
        # --- CENTRALIZED AUTO-TRADING ENTRY CHECK (Updated to include smart bias filter) ---
        entry_placeholder = st.empty()
        
        if auto_trading and strategy.screened_stocks and not is_close_time:
            st.subheader("Auto-Trade Monitor")
            entry_placeholder.info(f"Monitoring {len(strategy.screened_stocks)} stocks for entry signal. Open positions: {len(strategy.open_positions)}.")
            
            if len(strategy.open_positions) >= max_positions:
                entry_placeholder.warning(f"Position limit ({max_positions}) reached. Skipping new entries.")
            else:
                for symbol in strategy.screened_stocks:
                    if symbol in strategy.open_positions: continue
                    if symbol in strategy.rejected_symbols: continue
                        
                    ltp, day_h, day_l, open_price, _, quote_data = get_live_price(api, symbol)
                    
                    if ltp is None or ltp == 0 or open_price == 0 or day_h == 0 or day_l == 0: continue

                    signal = strategy.check_breakout_entry(None, ltp, open_price, day_h, day_l, symbol)
                    
                    if signal:
                        # --- NEW: COMPLEX BIAS FILTER CHECK ---
                        if enable_smart_bias_filter:
                            if not check_smart_bias_filter(api, symbol, min_candle_trade_value):
                                logging.info(f"Entry signal for {symbol} REJECTED by Complex Candle Bias Filter.")
                                entry_placeholder.info(f"Signal for **{symbol}** rejected: Candle volume/range criteria not met.")
                                continue # Skip trade
                        
                        # --- Proceed with trade if basic and complex checks pass ---
                        limit_price = get_mid_price(quote_data)
                        if limit_price == 0.0: continue
                            
                        qty = int(trade_amount / limit_price) if sizing_mode == "Fixed Amount" and trade_amount else quantity
                        if not quantity and not trade_amount or qty is None or qty <= 0: continue

                        trantype = 'B' if signal == 'BUY' else 'S'
                        
                        entry_placeholder.success(f"⚡ {signal} Signal FOUND on **{symbol}**! Placing **MIS** Limit Order @ {limit_price:.2f}...")
                        res_mis = execute_trade(api, trantype, qty, symbol, product_type='M', limit_price=limit_price)
                        
                        if res_mis and res_mis.get('stat') == 'Ok':
                            strategy.enter_position(signal, limit_price, day_h, day_l, qty, symbol, product_type='M')
                            entry_placeholder.success(f"✅ MIS Limit Order Placed (Order No: {res_mis.get('norenordno')}) for **{symbol}**.")
                            continue

                        error_mis = res_mis.get('emsg', 'Unknown Error') if res_mis else 'No API Response'
                        logging.warning(f"MIS Order Failed for {symbol}. Error: {error_mis}. Trying CNC...")
                        entry_placeholder.warning(f"MIS failed for **{symbol}**. Error: {error_mis}. Retrying with **CNC**...")
                        
                        res_cnc = execute_trade(api, trantype, qty, symbol, product_type='C', limit_price=limit_price)
                        
                        if res_cnc and res_cnc.get('stat') == 'Ok':
                            strategy.enter_position(signal, limit_price, day_h, day_l, qty, symbol, product_type='C')
                            entry_placeholder.success(f"✅ CNC Limit Order Placed (Order No: {res_cnc.get('norenordno')}) for **{symbol}**.")
                            continue
                        
                        error_cnc = res_cnc.get('emsg', 'Unknown Error') if res_cnc else 'No API Response'
                        strategy.rejected_symbols.add(symbol)
                        entry_placeholder.error(f"❌ Order Failed for **{symbol}**. MIS Error: {error_mis}. CNC Error: {error_cnc}. Added to rejection list for today.")


                entry_placeholder.info(f"Monitoring complete. No new entry signals found. Open positions: {len(strategy.open_positions)}.")

        col_scan, col_live = st.columns([1, 2])
        
        # Watchlist and Charting logic
        with col_scan:
            st.subheader("Heavy Volume Watchlist")
            if strategy.screened_stocks:
                w_data = []
                heavy_map = st.session_state.get('heavy_results', {})
                for sym in strategy.screened_stocks:
                    m = heavy_map.get(sym)
                    is_open = sym in strategy.open_positions
                    is_rejected = sym in strategy.rejected_symbols
                    
                    if m:
                        status_icon = "⬆️" if m['Bias'] == 'BUY' else "⬇️"
                        status_text = f"{status_icon} {m['Bias']}"
                        
                        if is_open: status_text = "✅ OPEN"
                        if is_rejected: status_text = "⛔ REJECTED"
                        
                        w_data.append({
                            "Symbol": sym,
                            "Status": status_text,
                            "Bias": m['Bias'],
                            "Vol Ratio": f"{m['VolumeRatio']:.2f}x",
                            "Current Vol": f"{m['CurrentVolume']:,}",
                            "Avg Vol (3d)": f"{m['AvgVolume']:,}",
                            "LTP": f"{m['LTP']:.2f}",
                            "Change (%)": f"{m['Change']:.2f}%"
                        })
                    else:
                        ltp, _, _, _, _, _ = get_live_price(api, sym)
                        w_data.append({"Symbol": sym, "Status": "N/A", "Bias": strategy.stock_bias.get(sym, 'N/A'), "Vol Ratio": "N/A", "Current Vol": "N/A", "Avg Vol (3d)": "N/A", "LTP": f"{ltp:.2f}" if ltp else "N/A", "Change (%)": "N/A"})

                st.dataframe(pd.DataFrame(w_data), hide_index=True, use_container_width=True)
            else:
                st.info("Run scan to populate the watchlist based on 3-day heavy volume criteria.")
                
        with col_live:
            st.subheader("Live Chart")
            all_tradable_symbols = list(set(strategy.screened_stocks) | set(strategy.open_positions.keys()))
            if all_tradable_symbols:
                sorted_symbols = sorted(strategy.open_positions.keys()) + sorted([s for s in all_tradable_symbols if s not in strategy.open_positions])
                sel_stock = st.selectbox("Select Stock to Chart", sorted_symbols, key="chart_stock_select") 
            else:
                sel_stock = None
                
            if sel_stock:
                df = get_historical_data_for_volume(api, sel_stock, days_back=5) # Reusing get_historical_data_for_volume for charting
                position_data = strategy.open_positions.get(sel_stock)
                
                if df is not None and len(df) > 0:
                    fig = create_chart(df.tail(100), position_data)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Historical data is unavailable for charting {sel_stock}.")
    
    # TAB 2: Positions & Auto-Exit
    with tab_positions:
        st.subheader("📍 Open Positions Manager (Bot Tracked)")
        
        open_positions = strategy.open_positions
        pos_data = []
        
        exit_message_placeholder = st.empty()
        
        if open_positions:
            symbols_to_exit = []

            for sym, pos in open_positions.items():
                ltp, day_h, day_l, _, _, quote_data = get_live_price(api, sym)
                
                if ltp is None or ltp == 0:
                    ltp = pos['entry_price']
                    day_h = pos.get('day_high', ltp * 1.01)
                    day_l = pos.get('day_low', ltp * 0.99)
                
                should_exit, reason = strategy.check_exit_conditions(sym, ltp, day_h, day_l)
                
                if not should_exit and is_close_time and pos['product_type'] == 'M':
                    should_exit = True
                    reason = "Market Close"
                
                netqty = pos['size'] if pos['type'] == 'long' else -pos['size']
                avg_prc = pos['entry_price']
                
                pnl = (ltp - avg_prc) * netqty if pos['type'] == 'long' else (avg_prc - ltp) * abs(netqty)
                
                exit_button_key = f"exit_button_{sym}"
                
                pos_data.append({
                    "Symbol": sym,
                    "Side": pos['type'].upper(),
                    "Qty": netqty,
                    "Avg": f"{avg_prc:.2f}",
                    "LTP": f"{ltp:.2f}",
                    "Target": f"{pos['target']:.2f}",
                    "SL": f"{pos['stop_loss']:.2f}",
                    "Trailing": f"{pos['trailing_stop']:.2f}" if pos['trailing_stop'] else 'N/A',
                    "Product": pos['product_type'],
                    "P&L": f"{pnl:,.2f}",
                    "Exit Hit?": f"⚠️ {reason.upper()}" if should_exit else "No",
                    "Action": st.button("Exit", key=exit_button_key)
                })
                
                # --- AUTO EXIT EXECUTION ---
                if auto_trading and should_exit:
                    symbols_to_exit.append(sym)
                    exit_message_placeholder.warning(f"⚡ Auto-Exiting {sym} due to: {reason.upper()}... Placing Limit Order.")
                    
                    trantype = 'S' if pos['type'] == 'long' else 'B'
                    product_type_to_use = pos['product_type']
                    
                    exit_limit_price = get_mid_price(quote_data)
                    if exit_limit_price == 0.0:
                         exit_message_placeholder.error(f"❌ Exit Order FAILED for {sym}. Cannot calculate mid-price for Limit Order. Try placing manually.")
                         continue
                    
                    res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)
                    
                    if res and res.get('stat') == 'Ok':
                        exit_message_placeholder.success(f"✅ Exit Limit Order Placed for {sym}. (Order No: {res.get('norenordno')}, Product: {product_type_to_use})")
                    else:
                        exit_message_placeholder.error(f"❌ Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
                        
            # --- MANUAL PER-POSITION EXIT EXECUTION ---
            manual_exits = [item for item in pos_data if item['Action']]
            if manual_exits:
                for item in manual_exits:
                    sym = item['Symbol']
                    pos = open_positions[sym]
                    
                    ltp, _, _, _, _, quote_data = get_live_price(api, sym)
                    
                    trantype = 'S' if pos['type'] == 'long' else 'B'
                    product_type_to_use = pos['product_type']
                    
                    exit_limit_price = get_mid_price(quote_data)
                    if exit_limit_price == 0.0:
                         exit_message_placeholder.error(f"❌ Manual Exit Order FAILED for {sym}. Cannot calculate mid-price for Limit Order.")
                         continue
                         
                    exit_message_placeholder.warning(f"Manual Exit: Placing {product_type_to_use} Limit Order for {sym} @ {exit_limit_price:.2f}...")

                    res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)

                    if res and res.get('stat') == 'Ok':
                        exit_message_placeholder.success(f"✅ Manual Exit Limit Order Placed for {sym}. (Order No: {res.get('norenordno')})")
                    else:
                        exit_message_placeholder.error(f"❌ Manual Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
                
                time_module.sleep(1)
                st.rerun()

            df_display = pd.DataFrame(pos_data).drop(columns=['Action'])
            st.dataframe(df_display, use_container_width=True)
            
        else:
            st.info("No open positions being tracked by the bot.")

        st.divider()
        st.subheader("🏦 Broker Account Positions (Open & Settled Today)")
        
        all_positions = api.get_positions()
        
        if all_positions and not (isinstance(all_positions, dict) and all_positions.get('stat') != 'Ok'):
            pos_df = pd.DataFrame(all_positions)
            
            def determine_status(row):
                netqty = int(row.get('netqty', 0))
                if netqty != 0:
                    return 'OPEN'
                elif int(row.get('sellqty', 0)) > 0 and int(row.get('buyqty', 0)) > 0:
                    return 'CLOSED (Settled)'
                else:
                    return 'CLOSED'
                    
            pos_df['Status'] = pos_df.apply(determine_status, axis=1)
            
            cols = ['tsym', 'netqty', 'buyqty', 'sellqty', 'netavgprc', 'ltp', 'rpnl', 'm2m_pnl', 'prd', 'Status']
            show_cols = [c for c in cols if c in pos_df.columns]

            st.dataframe(pos_df[show_cols], use_container_width=True)
            st.caption("Note: **netqty** = Qty remaining open. **rpnl** = Realized P&L. **m2m_pnl** = Unrealized P&L. **prd** is the Product Type.")
        else:
            st.info("Failed to fetch all broker positions or no activity today.")
        
        if st.button("Force Refresh Positions"):
            time_module.sleep(1)
            st.rerun() 
            

    # TAB 3: Order Book
    with tab_orders:
        st.subheader("📋 Today's Orders")
        orders = api.get_order_book()
        if orders and not (isinstance(orders, dict) and orders.get('stat') != 'Ok'):
            ord_df = pd.DataFrame(orders)
            cols = ['norenordno', 'tsym', 'trantype', 'qty', 'prc', 'status', 'rejreason', 'norentm', 'prd']
            show_cols = [c for c in cols if c in ord_df.columns]
            st.dataframe(ord_df[show_cols], use_container_width=True)
            st.caption("Product Type (`prd`): **C** = CNC/Delivery, **M** = MIS/Intraday.")
        else:
            st.info("No orders placed today or API error fetching orders.")

    # TAB 4: Trade Book
    with tab_trades:
        st.subheader("📒 Executed Trades")
        trades = api.get_trade_book()
        if trades and not (isinstance(trades, dict) and trades.get('stat') != 'Ok'):
            trd_df = pd.DataFrame(trades)
            cols = ['norentm', 'tsym', 'trantype', 'qty', 'flprc', 'exchordid']
            show_cols = [c for c in cols if c in trd_df.columns]
            st.dataframe(trd_df[show_cols], use_container_width=True)
        else:
            st.info("No trades executed today or API error fetching trades.")

    # --- AUTO REFRESH LOGIC ---
    if auto_refresh:
        time_module.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
