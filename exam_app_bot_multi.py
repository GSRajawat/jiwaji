
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta, timezone
import time as time_module
from supabase import create_client, Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# --- Helper function for explicit IST time ---
# IST is UTC+5:30.
def get_current_ist_time():
    """Returns the current explicit IST datetime object with +05:30 offset."""
    utc_time = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    # The resulting object is offset-aware (+05:30)
    return utc_time + ist_offset

# Add the parent directory to the path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
# Setting logging level to INFO for production use, but DEBUG for development/troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = "2811ab88839428bf06f939ffdca6046a02055346d367cd7fc614c0e5503e0a46"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@3"

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXH62eRvvouh_OiMjwqGU"

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
            # NOTE: rejected_symbols are reset daily, so they persist across script reruns on the same day.
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
            if pos['trailing_stop'] is None or new_trailing > pos['trailing_stop']: pos['trailing_stop'] = new_trailing
        elif pos['type'] == 'short':
            new_trailing = current_day_low * (1 + self.trailing_stop_pct)
            if pos['trailing_stop'] is None or new_trailing < pos['trailing_stop']: pos['trailing_stop'] = new_trailing

    def check_exit_conditions(self, symbol, current_price, current_day_high, current_day_low):
        """Check exit conditions (Target, SL, Trailing) for a specific symbol"""
        if symbol not in self.open_positions: return False, None
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
            'type': position_type, 'entry_price': entry_price, 'size': quantity,
            'stop_loss': stop_loss, 'target': target, 'trailing_stop': None,
            'day_high': day_high, 'day_low': day_low, 'product_type': product_type
        }

# --- Helper Functions ---

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
                # Return the full list for in-app filtering
                return equity_df[symbol_col].dropna().unique().tolist()
        
        return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'BAJFINANCE', 'ADANIPORTS', 'KOTAKBANK', 'MARUTI']
    except Exception as e:
        logging.error(f"Critical error during stock list generation: {e}")
        return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'BAJFINANCE', 'ADANIPORTS', 'KOTAKBANK', 'MARUTI']

def get_token_from_isin(api, symbol, exchange="NSE"):
    try:
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result and len(result['values']) > 0:
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ" or item.get('symname') == symbol: return item.get('token')
            return result['values'][0].get('token')
        return None
    except:
        return None

def get_historical_data_for_volume(api, symbol, days_back=5, interval='1'):
    try:
        token = get_token_from_isin(api, symbol)
        if not token: return None
        
        # --- IST TIME CORRECTION ---
        end_date = get_current_ist_time()
        start_date = end_date - timedelta(days=days_back) 
        # API expects naive datetime strings in DD-MM-YYYY HH:MM:SS format
        start_time_str = start_date.strftime("%d-%m-%Y") + " 09:00:00"
        end_time_str = end_date.strftime("%d-%m-%Y %H:%M:%S")
        # --- END IST TIME CORRECTION ---
        
        hist_data = api.get_time_price_series(exchange='NSE', token=token, starttime=start_time_str, endtime=end_time_str, interval=interval)
        if not hist_data or (isinstance(hist_data, dict) and hist_data.get('stat') != 'Ok'): return None
        data_list = []
        for item in hist_data:
            data_list.append({
                'Date': pd.to_datetime(item['time'], format='%d-%m-%Y %H:%M:%S'),
                'Volume': int(item.get('v', 0)), 'Open': float(item['into']), 'Close': float(item['intc']),
                'High': float(item['inth']), 'Low': float(item['intl']),
            })
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        return df.sort_index()
    except Exception as e:
        return None

def check_smart_bias_filter(api, symbol, min_candle_trade_value):
    try:
        hist_df = get_historical_data_for_volume(api, symbol, days_back=1, interval='1') 
        if hist_df is None or hist_df.empty: return False
        if len(hist_df) < 10: return False 
        current_candle = hist_df.iloc[-1]
        current_vol = current_candle['Volume']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        current_range = current_high - current_low
        current_mid_price = (current_high + current_low) / 2
        current_trade_value = current_vol * current_mid_price
        last_10_candles_vol = hist_df.iloc[-11:-1]['Volume']
        avg_vol_10 = last_10_candles_vol.mean()
        last_2_range_candles = hist_df.iloc[-3:-1] 
        avg_range_2 = (last_2_range_candles['High'] - last_2_range_candles['Low']).mean()
        if avg_vol_10 <= 0 or avg_range_2 <= 0: return False
        if current_trade_value < min_candle_trade_value: return False
        volume_condition = current_vol > (3 * avg_vol_10)
        range_condition = current_range > (2 * avg_range_2)
        return volume_condition or range_condition
    except Exception as e:
        return False

def apply_tick_size(price, reference_price):
    """
    Applies updated NSE equity tick size rules for limit prices:
    - Below ₹250: tick size 0.01
    - ₹251 to ₹1,000: tick size 0.05
    - ₹1,001 to ₹5,000: tick size 0.10
    - ₹5,001 to ₹10,000: tick size 0.50
    - ₹10,001 to ₹20,000: tick size 1.00
    - Above ₹20,001: tick size 5.00
    """
    if reference_price < 250:
        tick_size = 0.01
    elif 251 <= reference_price <= 1000:
        tick_size = 0.05
    elif 1001 <= reference_price <= 5000:
        tick_size = 0.10
    elif 5001 <= reference_price <= 10000:
        tick_size = 0.50
    elif 10001 <= reference_price <= 20000:
        tick_size = 1.00
    elif reference_price > 20001:
        tick_size = 5.00
    else:
        # For prices exactly 250 or between 250 and 251, safest fallback
        tick_size = 0.05

    return round(round(price / tick_size) * tick_size, 2)

def get_bid_price(quote_data, ltp):
    """Get the best bid price (for SELL orders)"""
    try:
        bp1 = float(quote_data.get('bp1', 0.0))
        if bp1 > 0:
            return apply_tick_size(bp1, ltp)
        return apply_tick_size(ltp, ltp)  # Fallback to LTP
    except Exception as e:
        logging.error(f"Error getting bid price: {e}")
        return apply_tick_size(ltp, ltp)

def get_ask_price(quote_data, ltp):
    """Get the best ask/offer price (for BUY orders)"""
    try:
        sp1 = float(quote_data.get('sp1', 0.0))
        if sp1 > 0:
            return apply_tick_size(sp1, ltp)
        return apply_tick_size(ltp, ltp)  # Fallback to LTP
    except Exception as e:
        logging.error(f"Error getting ask price: {e}")
        return apply_tick_size(ltp, ltp)

def has_pending_order(api, symbol):
    """Check if there's already a pending/open order for the symbol"""
    try:
        orders = api.get_order_book()
        if orders and not (isinstance(orders, dict) and orders.get('stat') != 'Ok'):
            for order in orders:
                order_symbol = order.get('tsym', '').replace('-EQ', '')
                order_status = order.get('status', '').upper()
                # Check for pending/open orders (not rejected, cancelled, or complete)
                if order_symbol == symbol and order_status in ['PENDING', 'OPEN', 'TRIGGER_PENDING']:
                    logging.info(f"Pending order found for {symbol}. Status: {order_status}")
                    return True
        return False
    except Exception as e:
        logging.error(f"Error checking pending orders for {symbol}: {e}")
        return False  # On error, allow order placement

def get_live_price(api, symbol, exchange="NSE"):
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
    try:
        if not isinstance(quantity, int) or quantity <= 0:
            return {'stat': 'Not_Ok', 'emsg': f'Invalid quantity: {quantity}'}
        
        ltp, _, _, _, _, _ = get_live_price(api, symbol)
        
        if ltp is not None and ltp != 0.0:
             final_limit_price = apply_tick_size(limit_price, ltp)
        else:
             final_limit_price = round(limit_price, 2) 

        formatted_price = f"{final_limit_price:.2f}"
        
        logging.info(f"Placing order for {symbol}. Type: {buy_or_sell}, Qty: {quantity}, Prd: {product_type}, Price: {formatted_price}")
        
        # Determine the correct product type string for the API
        # 'I' for Intraday (MIS), 'C' for Delivery (CNC)
        # Note: Some API wrappers might use 'M' for MIS, but standard Noren uses 'I' and 'C'.
        api_product_type = 'I' if product_type == 'M' else 'C'
        
        return api.place_order(buy_or_sell=buy_or_sell, product_type=api_product_type, exchange='NSE',
                             tradingsymbol=f"{symbol}-EQ", quantity=quantity, discloseqty=0,
                             price_type='LMT', price=formatted_price, retention='DAY')
    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")
        return {'stat': 'Not_Ok', 'emsg': f"API Exception: {e}"}


# --- VOLUME SCREENING FUNCTION (3-Day Average) ---
def get_heavy_volume_stocks(api, volume_multiplier, max_stocks, max_price, min_trade_value):
    # --- IST TIME CORRECTION ---
    now_ist = get_current_ist_time()
    current_date = now_ist.date()
    last_minute = now_ist.replace(second=0, microsecond=0)
    if now_ist.second < 30 and now_ist.minute != 17 and now_ist.hour >= 9:
         last_minute -= timedelta(minutes=1)
    time_filter = last_minute.time()
    # --- END IST TIME CORRECTION ---

    stock_list = get_fno_stocks_list()
    results = []
    
    for symbol in stock_list: 
        if len(results) >= max_stocks: 
            break
            
        try:
            # 1. Get Live Data 
            ltp, _, _, open_p, current_volume, quote_data = get_live_price(api, symbol)

            if ltp is None or ltp == 0 or open_p == 0: continue
            
            # **STRICT PRICE FILTER**: Reject stocks priced above max_price
            if ltp > max_price: continue
            trade_value = current_volume * ltp
            if trade_value < min_trade_value: continue
            
            # 2. Get Historical Volume Data 
            hist_df = get_historical_data_for_volume(api, symbol, days_back=5) 
            if hist_df is None or hist_df.empty: continue
            
            # 3. Calculate 3-Day Average Volume 
            past_days_df = hist_df[hist_df.index.date != current_date]
            historical_dates = past_days_df.index.normalize().unique()[-3:] 
            
            if len(historical_dates) < 3: continue 

            daily_volumes = []
            for date in historical_dates:
                day_data = past_days_df.loc[past_days_df.index.date == date]
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
                    # Normalize product type: API returns 'I' for MIS, 'C' for CNC
                    broker_prd = pos.get('prd', 'C')
                    product_type = 'M' if broker_prd in ('I', 'M') else 'C'
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
        
        # **CHECK FOR PENDING ORDERS**
        if has_pending_order(api, sym):
            exit_message_placeholder.warning(f"â³ Pending order already exists for **{sym}**. Skipping exit to avoid duplicate.")
            continue
        
        trantype = 'S' if pos['type'] == 'long' else 'B'
        product_type_to_use = pos['product_type']
        
        # Use bid for SELL, ask for BUY
        if trantype == 'S':
            exit_limit_price = get_bid_price(quote_data, ltp)
            price_type = "Bid"
        else:
            exit_limit_price = get_ask_price(quote_data, ltp)
            price_type = "Ask"
        
        if exit_limit_price == 0.0:
             exit_message_placeholder.error(f"âŒ Exit Order FAILED for {sym}. Cannot calculate {price_type} price for Limit Order.")
             continue
        exit_message_placeholder.warning(f"Manual Exit: Placing {product_type_to_use} Limit Order for {sym} @ {exit_limit_price:.2f} ({price_type}, Tick Adjusted)...")
        
        res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)
        
        if res and res.get('stat') == 'Ok':
            order_no = res.get('norenordno')
            if order_no and 'order_conditions' in st.session_state:
                st.session_state.order_conditions[order_no] = "Manual Exit"
            
            exit_message_placeholder.success(f"âœ… Manual Exit Limit Order Placed for {sym}. (Order No: {res.get('norenordno')}). Waiting 5s for broker sync...")
            time_module.sleep(5)  # **5 SECOND DELAY**
        else:
            exit_message_placeholder.error(f"âŒ Manual Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
    
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

def get_total_positions_count(api):
    """Get count of total positions (open + closed/settled) in broker account today"""
    try:
        all_positions = api.get_positions()
        
        if all_positions and not (isinstance(all_positions, dict) and all_positions.get('stat') != 'Ok'):
            # Count all positions that have any activity today
            total_count = 0
            for pos in all_positions:
                # Count if there's any buy or sell activity
                buyqty = int(pos.get('buyqty', 0))
                sellqty = int(pos.get('sellqty', 0))
                
                if buyqty > 0 or sellqty > 0:
                    total_count += 1
            
            return total_count
        return 0
    except Exception as e:
        logging.error(f"Error getting total positions count: {e}")
        return 0

# --- MAIN APP ---

def main():
    st.set_page_config(page_title="Heavy Volume Breakout Bot (Day High/Low Only)", layout="wide")
    
    try:
        api = init_flattrade_api()
        if 'strategy' not in st.session_state: st.session_state.strategy = FnOBreakoutStrategy()
        if 'order_conditions' not in st.session_state: st.session_state.order_conditions = {}
        
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
        stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 5.0, 1.0, 0.1, key="sl_pct")
        target_pct = st.slider("Target (%)", 0.1, 10.0, 3.0, 0.1, key="tg_pct")
        trailing_stop_pct = st.slider("Trailing Stop (%)", 0.5, 3.0, 1.0, 0.1, key="ts_pct")
        
        strategy.stop_loss_pct = stop_loss_pct / 100
        strategy.target_pct = target_pct / 100
        strategy.trailing_stop_pct = trailing_stop_pct / 100
        
        st.subheader("Volume Screening Parameters")
        
        enable_daily_volume_screen = st.checkbox("✅ Enable Daily Heavy Volume Screen", value=True, key="enable_daily_vol_screen", help="If unchecked, the bot monitors F&O stocks but ignores the 3-day average cumulative volume filter.")

        if enable_daily_volume_screen:
            volume_multiplier = st.slider("Volume Multiplier (X times 3-day Avg)", 2.0, 10.0, 3.0, 0.5, key="vol_mult_slider", help="Current cumulative volume must be this many times the 3-day average volume for the same time.")
        else:
             volume_multiplier = 3.0 # Default value, ignored if screen is off
        
        scan_limit = st.number_input("Max Stocks to Track (5-20)", min_value=5, max_value=200, value=20, step=1, key="scan_limit")
        
        max_price = st.number_input("Stock Price Limit (₹)", min_value=10.0, max_value=20000.0, value=1000.0, step=100.0, key="max_prc")
        
        min_trade_value = st.number_input("Min Daily Trade Value (₹)", min_value=0, value=5000000, step=1000000, key="min_trade_value", help="Minimum total value of shares traded today (Volume * LTP).")
        
        
        st.subheader("Entry Filters")
        
        enable_day_high_low_entry = st.checkbox("⭐ Enable Day High/Low Breakout Entry (ONLY Entry)", True, key="day_high_low_entry_toggle", help="Highest priority entry. Places order if LTP = Day High (Long) or LTP = Day Low (Short). Midpoint logic is removed.")
        
        # NEW: Entry Time Window Filter
        enable_entry_time_window = st.checkbox("⏰ Enable Entry Time Window", True, key="entry_time_window_toggle", help="Only allow new entries between specified times")
        
        if enable_entry_time_window:
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                entry_start_time = st.time_input("Start Time", value=time(9, 17, 0), key="entry_start_time", help="Entries allowed after this time")
            with col_time2:
                entry_end_time = st.time_input("End Time", value=time(13, 0, 0), key="entry_end_time", help="Entries allowed before this time")
        else:
            entry_start_time = time(9, 17, 0)  # Default values
            entry_end_time = time(13, 0, 0)
        
        # NEW: Max Total Positions (Open + Closed) Limit
        enable_total_positions_limit = st.checkbox("📊 Enable Max Total Positions Limit (Broker Account)", False, key="enable_total_pos_limit_toggle", help="Limits total positions (open + closed today) in broker account")
        
        if enable_total_positions_limit:
            max_total_positions = st.number_input("Max Total Positions (Open + Closed Today)", min_value=1, max_value=100, value=20, step=1, key="max_total_pos_input", help="Total number of positions (open + settled) allowed in broker account today")
        else:
            max_total_positions = 999  # No limit
        
        st.subheader("Trade Settings")
        
        st.radio("New Entry Product Priority", ['MIS (Intraday) -> CNC (Delivery)'], index=0, disabled=True, help="Bot will attempt MIS first, then CNC if MIS fails.")
            
        max_positions = st.number_input("Max Open Positions Limit", min_value=1, value=10, step=1, key="max_pos_limit")

        st.subheader("Position Sizing")
        sizing_mode = st.radio("Mode", ["Fixed Quantity", "Fixed Amount"], key="size_mode")
        if sizing_mode == "Fixed Quantity":
            quantity = st.number_input("Qty", 1, value=1, key="qty_input")
            trade_amount = None
        else:
            trade_amount = st.number_input("Amount (₹)", 1000, value=1000, key="amt_input")
            quantity = None
            
        st.divider()
        
        auto_scan = st.toggle("🤖 Auto Scan (on load)", False, key="auto_scan_toggle", help="Automatically scans and generates the watchlist.")
        auto_trading = st.toggle("Enable Auto Trading (Multi-Stock)", False, key="auto_trade_toggle")
        
        close_time_str = st.selectbox("Auto-Close Time", ["15:05:00"], index=0, key="close_time_select")
        
        current_time_ist_dt = get_current_ist_time()
        current_time_ist_time_only = current_time_ist_dt.time()
        close_time = datetime.strptime(close_time_str, "%H:%M:%S").time()
        is_close_time = current_time_ist_time_only >= close_time
        
        ist_display_str = current_time_ist_dt.strftime('%H:%M:%S') + ' (+05:30 IST)'
        
        st.caption(f"Current Time: **{ist_display_str}**")

        if st.button("🔴 Exit All Positions Now", disabled=not strategy.open_positions, type="secondary"):
            st.session_state.manual_exit_all = True

        st.divider()
        
        auto_refresh = st.toggle("🔄 Auto Refresh (5s)", value=False, key="auto_refresh_toggle")
        
        if not auto_scan:
            if st.button("🔍 Scan Watchlist", type="primary", key="scan_button"):
                st.session_state.run_scanning = True
        
        # --- DANGER ZONE: Separated exit button ---
        st.divider()
        st.markdown("### ⚠️ DANGER ZONE")
        st.caption("⚠️ **Warning:** This will exit ALL open positions immediately!")
        
        # Add extra spacing
        st.write("")
        
        # Confirmation toggle before showing button
        show_exit_button = st.checkbox("Enable Exit All Button", value=False, key="enable_exit_all_checkbox", help="Check this box to enable the Exit All button")
        
        if show_exit_button:
            st.write("")  # Extra spacing
            if st.button("🔴 EXIT ALL POSITIONS NOW", disabled=not strategy.open_positions, type="secondary", key="exit_all_btn", use_container_width=True):
                st.session_state.manual_exit_all = True
        else:
            st.info("☑️ Check the box above to enable the Exit All button")
    
    st.title("🧪 Heavy Volume Breakout Bot (Day High/Low Only)")
    
    limits_resp = api.get_limits()
    cash_avail = float(limits_resp.get('cash', 0))
    margin_used = float(limits_resp.get('marginused', 0))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Available Cash", f"₹{cash_avail:,.2f}")
    c2.metric("🔒 Margin Used", f"₹{margin_used:,.2f}")
    c3.metric("📊 Auto-Trading", "Active" if auto_trading else "Inactive", delta_color="normal")
    
    # NEW: Show Total Positions Count
    if enable_total_positions_limit:
        total_pos_count = get_total_positions_count(api)
        c4.metric("📊 Total Positions Today", f"{total_pos_count}/{max_total_positions}", 
                  delta=f"{max_total_positions - total_pos_count} remaining", 
                  delta_color="normal")
    else:
        c4.metric("⏰ Entry Window", 
                  "Active" if enable_entry_time_window and entry_start_time <= current_time_ist_time_only <= entry_end_time else "All Day",
                  delta_color="normal")
    
    st.divider()

    current_date = get_current_ist_time().date()
    strategy.reset_daily_data(current_date)
    
    check_and_sync_positions(api, strategy)
    strategy.update_position_parameters()
    
    exit_placeholder = st.empty()
    if st.session_state.get('manual_exit_all', False):
        with exit_placeholder:
            exit_all_positions(api, strategy, st.status("Exiting all positions...", expanded=True))
        st.session_state.manual_exit_all = False
        st.rerun() 

    should_run_scan = st.session_state.get('run_scanning', False) or (auto_scan and not strategy.screened_stocks)

    if should_run_scan:
        if enable_daily_volume_screen:
            with st.spinner(f"Scanning Heavy Volume Stocks (Multiplier: {volume_multiplier}x, Max: {scan_limit})..."):
                heavy_results = get_heavy_volume_stocks(api, volume_multiplier, int(scan_limit), max_price, min_trade_value)
                
                strategy.screened_stocks = [x['Symbol'] for x in heavy_results]
                strategy.stock_bias = {x['Symbol']: x['Bias'] for x in heavy_results}
                st.session_state.heavy_results = {x['Symbol']: x for x in heavy_results}
                st.session_state.run_scanning = False 
                st.success(f"Screened {len(strategy.screened_stocks)} stocks meeting **3-day heavy volume** criteria.")
        else:
            with st.spinner(f"Scanning F&O Stocks (Max: {scan_limit}, Max Price: ₹{max_price})..."):
                
                full_fno_list = get_fno_stocks_list()
                filtered_symbols = []
                temp_results = {}
                
                for symbol in full_fno_list:
                    if len(filtered_symbols) >= scan_limit:
                        break
                        
                    ltp, _, _, open_p, current_volume, _ = get_live_price(api, symbol)

                    if ltp is None or ltp == 0 or open_p == 0: continue
                    
                    if ltp > max_price: continue
                    trade_value = current_volume * ltp
                    if trade_value < min_trade_value: continue
                    
                    filtered_symbols.append(symbol)
                    
                    chg = ((ltp - open_p) / open_p) * 100
                    bias = 'BUY' if ltp >= open_p else 'SELL'

                    temp_results[symbol] = {
                        'Symbol': symbol, 'LTP': ltp, 'Change': chg, 'Bias': bias,
                        'CurrentVolume': int(current_volume), 'AvgVolume': 0, 'VolumeRatio': 0.0,
                        'TradeValue': trade_value
                    }

                strategy.screened_stocks = filtered_symbols
                strategy.stock_bias = {s: temp_results[s]['Bias'] for s in filtered_symbols}
                st.session_state.heavy_results = temp_results
                st.session_state.run_scanning = False 
                st.success(f"**Volume screening disabled.** Monitoring {len(strategy.screened_stocks)} stocks filtered by price/value criteria.")

    tab_monitor, tab_positions, tab_orders, tab_trades = st.tabs([
        "📈 Strategy Monitor", "📍 Open Positions & Exit", "📋 Order Book", "📒 Trade Book"
    ])
    
    with tab_monitor:
        entry_placeholder = st.empty()
        
        if auto_trading and strategy.screened_stocks and not is_close_time:
            st.subheader("Auto-Trade Monitor")
            
            # **NEW: Check Entry Time Window**
            if enable_entry_time_window:
                if not (entry_start_time <= current_time_ist_time_only <= entry_end_time):
                    entry_placeholder.info(f"⏰ Outside entry time window ({entry_start_time.strftime('%H:%M:%S')} - {entry_end_time.strftime('%H:%M:%S')}). Current time: {ist_display_str}. Monitoring paused.")
                    # Don't return, just skip the entry loop but continue monitoring
                    st.subheader("Watchlist (Time Filter Active)")
                else:
                    entry_placeholder.info(f"✅ Within entry time window. Monitoring {len(strategy.screened_stocks)} stocks for entry signal. Open positions: {len(strategy.open_positions)}.")
            else:
                entry_placeholder.info(f"Monitoring {len(strategy.screened_stocks)} stocks for entry signal. Open positions: {len(strategy.open_positions)}.")
            
            # **NEW: Check Total Positions Limit in Broker Account**
            if enable_total_positions_limit:
                current_total_positions = get_total_positions_count(api)
                
                if current_total_positions >= max_total_positions:
                    entry_placeholder.warning(f"🚫 Total positions limit reached! Broker Account: {current_total_positions}/{max_total_positions} positions (open + closed today). No new entries allowed.")
                    # Skip entry scanning
                    st.caption(f"📊 Total Broker Positions Today: **{current_total_positions}** (Limit: {max_total_positions})")
                else:
                    st.caption(f"📊 Total Broker Positions Today: **{current_total_positions}** / {max_total_positions}")
            
            # Only proceed with entry scanning if all conditions are met
            can_enter_new_positions = True
            
            if enable_entry_time_window and not (entry_start_time <= current_time_ist_time_only <= entry_end_time):
                can_enter_new_positions = False
            
            if enable_total_positions_limit and get_total_positions_count(api) >= max_total_positions:
                can_enter_new_positions = False
            
            if can_enter_new_positions:
                for symbol in strategy.screened_stocks:
                    if symbol in strategy.open_positions: 
                        logging.debug(f"Skipping {symbol}: Position already open.")
                        continue
                    if symbol in strategy.rejected_symbols: 
                        logging.debug(f"Skipping {symbol}: Previously rejected today.")
                        continue
                    
                    if len(strategy.open_positions) >= max_positions:
                        entry_placeholder.warning(f"Position limit ({max_positions}) reached mid-scan. Stopping new entries.")
                        break 
                    
                    # **DOUBLE CHECK: Total positions limit before each order**
                    if enable_total_positions_limit:
                        if get_total_positions_count(api) >= max_total_positions:
                            entry_placeholder.warning(f"🚫 Total positions limit reached during scan! No more entries.")
                            break
                        
                    ltp, day_h, day_l, open_price, _, quote_data = get_live_price(api, symbol)
                    
                    if ltp is None or ltp == 0 or open_price == 0 or day_h == 0 or day_l == 0: continue

                    signal = None
                    signal_source = None
                    
                    if enable_day_high_low_entry:
                        if ltp >= day_h and ltp > open_price: 
                             signal = 'BUY'
                             signal_source = 'Day High Break'
                        elif ltp <= day_l and ltp < open_price: 
                             signal = 'SELL'
                             signal_source = 'Day Low Break'

                    if signal:
                        # ... [rest of the existing signal processing code remains the same] ...
                    # **CHECK FOR PENDING ORDERS FIRST**
                        if has_pending_order(api, symbol):
                            entry_placeholder.warning(f"⏳ Pending order already exists for **{symbol}**. Skipping to avoid duplicate orders.")
                            time_module.sleep(0.5)  # Small delay before next iteration
                            continue
                        
                        # Determine correct limit price based on signal direction
                        if signal == 'BUY':
                            limit_price = get_ask_price(quote_data, ltp)
                            price_type = "Ask/Offer"
                        else:  # SELL
                            limit_price = get_bid_price(quote_data, ltp)
                            price_type = "Bid"
                        
                        if limit_price == 0.0: 
                            entry_placeholder.warning(f"Cannot calculate {price_type} price for {symbol}. Skipping.")
                            continue
                            
                        qty = int(trade_amount / limit_price) if sizing_mode == "Fixed Amount" and trade_amount else quantity
                        if not quantity and not trade_amount or qty is None or qty <= 0: continue

                        trantype = 'B' if signal == 'BUY' else 'S'
                        
                        # 1. Attempt MIS (Intraday)
                        entry_placeholder.success(f"⚡ {signal_source} Signal FOUND on **{symbol}**! Placing **MIS** Limit Order @ {limit_price:.2f} ({price_type}, Tick Adjusted)...")
                        res_mis = execute_trade(api, trantype, qty, symbol, product_type='M', limit_price=limit_price)
                        
                        if res_mis and res_mis.get('stat') == 'Ok':
                            order_no = res_mis.get('norenordno')
                            if order_no and 'order_conditions' in st.session_state:
                                st.session_state.order_conditions[order_no] = signal_source 
                            
                            entry_placeholder.success(f"✅ MIS Order Placed (Order No: {order_no}) for **{symbol}**. Waiting 5s to verify execution...")
                            time_module.sleep(5)  # **5 SECOND DELAY**
                            
                            # VERIFY ORDER WAS FILLED before adding to positions
                            order_status = api.get_order_book()
                            is_filled = False
                            is_rejected = False
                            
                            if order_status and not (isinstance(order_status, dict) and order_status.get('stat') != 'Ok'):
                                for order in order_status:
                                    if order.get('norenordno') == order_no:
                                        status = order.get('status', '').upper()
                                        if status == 'COMPLETE':
                                            is_filled = True
                                            fill_price = float(order.get('avgprc', limit_price))
                                            # NOW add to positions with actual fill price
                                            strategy.enter_position(signal, fill_price, day_h, day_l, qty, symbol, product_type='M')
                                            entry_placeholder.success(f"✅ MIS Order FILLED for **{symbol}** @ {fill_price:.2f}. Position added.")
                                            time_module.sleep(1)
                                            break
                                        elif status == 'REJECTED':
                                            is_rejected = True
                                            error_mis = order.get('rejreason', 'Order Rejected')
                                            entry_placeholder.warning(f"⚠️ MIS Order REJECTED for **{symbol}**. Reason: {error_mis}")
                                            break
                            
                            if is_filled:
                                continue  # Move to next symbol - SUCCESS!
                            
                            # If order was rejected or not filled, get error message
                            if is_rejected:
                                # Get the actual rejection reason
                                for order in order_status:
                                    if order.get('norenordno') == order_no:
                                        error_mis = order.get('rejreason', 'Order Rejected')
                                        break
                            else:
                                error_mis = "Order not filled within 5 seconds"
                        else:
                            error_mis = res_mis.get('emsg', 'No API Response') if res_mis else 'No API Response'
                            is_rejected = True  # API level rejection
                        
                        # --- CNC FALLBACK LOGIC ---
                        # Only attempt CNC if:
                        # 1. MIS order failed/rejected
                        # 2. Signal is BUY (CNC doesn't support short selling)
                        
                        if trantype == 'S':
                            # SELL orders cannot use CNC fallback
                            strategy.rejected_symbols.add(symbol)
                            st.session_state.strategy = strategy 
                            entry_placeholder.error(f"❌ MIS Sell Order Failed for **{symbol}**. Error: {error_mis}. CNC Fallback SKIPPED (CNC doesn't support short selling). Symbol rejected for today.")
                            time_module.sleep(1)
                            st.rerun()
                        
                        # --- If we reach here, it's a BUY order that failed MIS ---
                        logging.warning(f"MIS Order Failed/Not Filled for {symbol}. Error: {error_mis}. Attempting CNC fallback...")
                        entry_placeholder.warning(f"⚠️ MIS failed for **{symbol}**. Error: {error_mis}. Checking CNC Fallback in 2s...")
                        
                        # --- Explicit 2-second delay ---
                        time_module.sleep(2)
                        
                        # 2. Attempt CNC (Delivery) - Only for BUY orders
                        entry_placeholder.info(f"💼 Attempting CNC (Delivery) Buy for **{symbol}** @ {limit_price:.2f} ({price_type})...")
                        res_cnc = execute_trade(api, trantype, qty, symbol, product_type='C', limit_price=limit_price)
                        
                        if res_cnc and res_cnc.get('stat') == 'Ok':
                            order_no = res_cnc.get('norenordno')
                            if order_no and 'order_conditions' in st.session_state:
                                st.session_state.order_conditions[order_no] = signal_source 
                            
                            entry_placeholder.success(f"✅ CNC Order Placed (Order No: {order_no}) for **{symbol}**. Waiting 5s to verify execution...")
                            time_module.sleep(5)  # **5 SECOND DELAY**
                            
                            # VERIFY ORDER WAS FILLED before adding to positions
                            order_status = api.get_order_book()
                            is_filled = False
                            
                            if order_status and not (isinstance(order_status, dict) and order_status.get('stat') != 'Ok'):
                                for order in order_status:
                                    if order.get('norenordno') == order_no:
                                        status = order.get('status', '').upper()
                                        if status == 'COMPLETE':
                                            is_filled = True
                                            fill_price = float(order.get('avgprc', limit_price))
                                            # NOW add to positions with actual fill price
                                            strategy.enter_position(signal, fill_price, day_h, day_l, qty, symbol, product_type='C')
                                            entry_placeholder.success(f"✅ CNC Order FILLED for **{symbol}** @ {fill_price:.2f}. Position added.")
                                            time_module.sleep(1)
                                            break
                                        elif status == 'REJECTED':
                                            error_cnc = order.get('rejreason', 'Order Rejected')
                                            entry_placeholder.error(f"❌ CNC Order REJECTED for **{symbol}**. Reason: {error_cnc}")
                                            break
                            
                            if is_filled:
                                continue  # Move to next symbol - SUCCESS!
                            
                            error_cnc = "Order not filled within 5 seconds" if not is_filled else res_cnc.get('emsg', 'Unknown Error')
                        else:
                            error_cnc = res_cnc.get('emsg', 'Unknown Error') if res_cnc else 'No API Response'
                        
                        # 3. Both MIS and CNC failed
                        strategy.rejected_symbols.add(symbol)
                        st.session_state.strategy = strategy 
                        
                        entry_placeholder.error(f"❌ All Order Attempts Failed for **{symbol}**. MIS Error: {error_mis}. CNC Error: {error_cnc}. Added to rejection list for today (will not re-order).")
                        time_module.sleep(1)
                        st.rerun() 

                entry_placeholder.info(f"Monitoring complete. No new entry signals found. Open positions: {len(strategy.open_positions)}.")

            col_scan, col_live = st.columns([1, 2])
            
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
                                "Vol Ratio": f"{m['VolumeRatio']:.2f}x" if enable_daily_volume_screen else "N/A (Screen Off)",
                                "Current Vol": f"{m['CurrentVolume']:,}",
                                "Avg Vol (3d)": f"{m['AvgVolume']:,}" if enable_daily_volume_screen else "N/A",
                                "LTP": f"{m['LTP']:.2f}",
                                "Change (%)": f"{m['Change']:.2f}%"
                            })
                        else:
                            ltp, _, _, _, _, _ = get_live_price(api, sym)
                            status_text = "✅ OPEN" if is_open else "⛔ REJECTED" if is_rejected else "Monitoring"
                            w_data.append({"Symbol": sym, "Status": status_text, "Bias": strategy.stock_bias.get(sym, 'N/A'), "Vol Ratio": "N/A (Screen Off)", "Current Vol": "N/A", "Avg Vol (3d)": "N/A", "LTP": f"{ltp:.2f}" if ltp else "N/A", "Change (%)": "N/A"})

                    st.dataframe(pd.DataFrame(w_data), hide_index=True, use_container_width=True)
                else:
                    st.info("Run scan to populate the watchlist based on criteria.")
                    
            with col_live:
                st.subheader("Live Chart")
                all_tradable_symbols = list(set(strategy.screened_stocks) | set(strategy.open_positions.keys()))
                if all_tradable_symbols:
                    sorted_symbols = sorted(strategy.open_positions.keys()) + sorted([s for s in all_tradable_symbols if s not in strategy.open_positions])
                    sel_stock = st.selectbox("Select Stock to Chart", sorted_symbols, key="chart_stock_select") 
                else:
                    sel_stock = None
                    
                if sel_stock:
                    df = get_historical_data_for_volume(api, sel_stock, days_back=5) 
                    position_data = strategy.open_positions.get(sel_stock)
                    
                    if df is not None and len(df) > 0:
                        fig = create_chart(df.tail(100), position_data)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Historical data is unavailable for charting {sel_stock}.")
        
        with tab_positions:
            st.subheader("📍 Open Positions Manager (Bot Tracked)")
            
            open_positions = strategy.open_positions
            pos_data = []
            total_pnl = 0.0 
            
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
                    total_pnl += pnl 
                    
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
                        "P&L_Float": pnl, 
                        "Exit Hit?": f"⚠️ {reason.upper()}" if should_exit else "No",
                        "Action": st.button("Exit", key=exit_button_key)
                    })
                    
                    if auto_trading and should_exit:
                        # **CHECK FOR PENDING ORDERS**
                        if has_pending_order(api, sym):
                            exit_message_placeholder.warning(f"â³ Pending order already exists for **{sym}**. Skipping auto-exit.")
                            continue
                        
                        symbols_to_exit.append(sym)
                        exit_message_placeholder.warning(f"âš¡ Auto-Exiting {sym} due to: {reason.upper()}... Placing Limit Order.")
                        
                        trantype = 'S' if pos['type'] == 'long' else 'B'
                        product_type_to_use = pos['product_type']
                        
                        # Use bid for SELL, ask for BUY
                        if trantype == 'S':
                            exit_limit_price = get_bid_price(quote_data, ltp)
                            price_type = "Bid"
                        else:
                            exit_limit_price = get_ask_price(quote_data, ltp)
                            price_type = "Ask"
                        
                        if exit_limit_price == 0.0:
                            exit_message_placeholder.error(f"âŒ Exit Order FAILED for {sym}. Cannot calculate {price_type} price for Limit Order. Try placing manually.")
                            continue
                        
                        res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)
                        
                        if res and res.get('stat') == 'Ok':
                            order_no = res.get('norenordno')
                            if order_no and 'order_conditions' in st.session_state:
                                st.session_state.order_conditions[order_no] = reason 
                            exit_message_placeholder.success(f"âœ… Exit Limit Order Placed for {sym}. (Order No: {order_no}, Product: {product_type_to_use}, Price: {exit_limit_price:.2f} {price_type})")
                            time_module.sleep(5)  # **5 SECOND DELAY**
                        else:
                            exit_message_placeholder.error(f"âŒ Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
                            
                manual_exits = [item for item in pos_data if item['Action']]
                if manual_exits:
                    for item in manual_exits:
                        sym = item['Symbol']
                        pos = open_positions[sym]
                        
                        ltp, _, _, _, _, quote_data = get_live_price(api, sym)
                        
                        trantype = 'S' if pos['type'] == 'long' else 'B'
                        product_type_to_use = pos['product_type']
                        
                        exit_limit_price = get_mid_price(quote_data, ltp) 
                        
                        if exit_limit_price == 0.0:
                            exit_message_placeholder.error(f"❌ Manual Exit Order FAILED for {sym}. Cannot calculate mid-price for Limit Order.")
                            continue
                            
                        exit_message_placeholder.warning(f"Manual Exit: Placing {product_type_to_use} Limit Order for {sym} @ {exit_limit_price:.2f} (Tick Adjusted)...")

                        res = execute_trade(api, trantype, pos['size'], sym, product_type=product_type_to_use, limit_price=exit_limit_price)

                        if res and res.get('stat') == 'Ok':
                            order_no = res.get('norenordno')
                            if order_no and 'order_conditions' in st.session_state:
                                st.session_state.order_conditions[order_no] = "Manual Exit"
                            exit_message_placeholder.success(f"✅ Manual Exit Limit Order Placed for {sym}. (Order No: {order_no})")
                        else:
                            exit_message_placeholder.error(f"❌ Manual Exit Order FAILED for {sym}. Error: {res.get('emsg', 'Unknown Error')}")
                    
                    time_module.sleep(1)
                    st.rerun()

                pos_data_display = []
                for item in pos_data:
                    item['P&L'] = f"{item.pop('P&L_Float'):,.2f}" 
                    pos_data_display.append({k: item[k] for k in item if k != 'P&L_Float' and k != 'Action'})
                    
                total_row = {
                    "Symbol": "**TOTAL**",
                    "Side": "",
                    "Qty": "",
                    "Avg": "",
                    "LTP": "",
                    "Target": "",
                    "SL": "",
                    "Trailing": "",
                    "Product": "",
                    "P&L": f"**{total_pnl:,.2f}**",
                    "Exit Hit?": "",
                }
                pos_data_display.append(total_row)
                
                df_display = pd.DataFrame(pos_data_display)
                
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
                

        with tab_orders:
            st.subheader("📋 Today's Orders")
            orders = api.get_order_book()
            
            if orders and not (isinstance(orders, dict) and orders.get('stat') != 'Ok'):
                ord_df = pd.DataFrame(orders)
                
                ord_df['Condition Met'] = ord_df['norenordno'].astype(str).map(st.session_state.get('order_conditions', {}))
                ord_df['Condition Met'] = ord_df['Condition Met'].fillna('N/A (Sync or Rejected)')
                
                cols = ['norenordno', 'tsym', 'trantype', 'qty', 'prc', 'status', 'Condition Met', 'rejreason', 'norentm', 'prd']
                show_cols = [c for c in cols if c in ord_df.columns]
                st.dataframe(ord_df[show_cols], use_container_width=True)
                st.caption("Product Type (`prd`): **C** = CNC/Delivery, **M** = MIS/Intraday. **Condition Met** tracks the strategy signal (Day High Break, SL Hit, Manual Exit, etc).")
            else:
                st.info("No orders placed today or API error fetching orders.")

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

        if auto_refresh:
            time_module.sleep(5)
            st.rerun()

if __name__ == "__main__":
    main()
