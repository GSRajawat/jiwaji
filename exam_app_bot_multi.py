import streamlit as st
import os
import sys
import logging
import datetime
import time
import pandas as pd
import numpy as np

# Add the parent directory to the sys.path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "fcdc3a9d93f15000ea83b027eb2bc5ed9e2249c40ed346fc3e2f98c1492dbf14")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 50  # Enough candles for SDVWAP calculation
MARKET_START_TIME = datetime.time(9, 15)  # Market starts at 9:15 AM

# Initialize session state variables
if 'predetermined_capital' not in st.session_state:
    st.session_state.predetermined_capital = 100000
if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {}
if 'daily_profit_exits' not in st.session_state:
    st.session_state.daily_profit_exits = {}  # Track profit exits per stock per day
if 'trading_start_time' not in st.session_state:
    st.session_state.trading_start_time = datetime.time(9, 30)  # Updated to match bt2 logic
if 'trading_end_time' not in st.session_state:
    st.session_state.trading_end_time = datetime.time(15, 20)  # Updated to match bt2 logic
if 'profit_target_pct' not in st.session_state:
    st.session_state.profit_target_pct = 5.0  # 5% profit target from bt2

@st.cache_resource
def get_api_instance(user_id, user_session):
    """Initializes and caches the NorenApiPy instance."""
    api = NorenApiPy()
    logging.info("Attempting to set API session...")
    try:
        ret = api.set_session(userid=user_id, password='', usertoken=user_session)
        if ret is True or (isinstance(ret, dict) and ret.get('stat') == 'Ok'):
            logging.info(f"API session set successfully for user: {user_id}")
            st.success(f"API session connected for user: {user_id}")
            return api
        else:
            error_msg = ret.get('emsg', 'Unknown error') if isinstance(ret, dict) else str(ret)
            st.error(f"Failed to set API session: {error_msg}. Please check your credentials.")
            return None
    except Exception as e:
        st.error(f"An exception occurred during API session setup: {e}")
        return None

api = get_api_instance(USER_ID, USER_SESSION)
if api is None:
    st.stop()

@st.cache_data
def load_symbols_from_csv(file_path="NSE_Equity.csv"):
    """Loads stock symbols and tokens from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        if all(col in df.columns for col in ['Exchange', 'Token', 'Tradingsymbol', 'Instrument']):
            equity_symbols = df[df['Instrument'] == 'EQ'][['Exchange', 'Token', 'Tradingsymbol']].copy()
            
            symbols_list = []
            for index, row in equity_symbols.iterrows():
                symbols_list.append({
                    'exchange': row['Exchange'],
                    'token': str(row['Token']),
                    'tsym': row['Tradingsymbol']
                })
            st.success(f"Loaded {len(symbols_list)} equity symbols from {file_path}.")
            return symbols_list
        else:
            st.error(f"CSV file '{file_path}' must contain required columns.")
            return []
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
        return []
    except Exception as e:
        st.error(f"Error loading symbols from CSV: {e}")
        return []

def place_market_order(api, symbol, exchange, action, quantity, product_type='MIS', remarks='SDVWAP_Strategy'):
    """Place a market order"""
    try:
        st.info(f"Placing {action} order for {symbol}: Qty={quantity}")
        
        response = api.place_order(
            buy_or_sell=action,  # 'B' for Buy, 'S' for Sell
            product_type=product_type,
            exchange=exchange,
            tradingsymbol=symbol,
            quantity=int(quantity),
            discloseqty=0,
            price_type='MKT',  # Market order
            price=0,
            trigger_price=None,
            retention='DAY',
            remarks=remarks
        )
        
        if response and response.get('stat') == 'Ok':
            order_no = response.get('norenordno')
            st.success(f"✅ Order placed successfully - Order No: {order_no}")
            logging.info(f"Order placed: {symbol} {action} {quantity} - Order No: {order_no}")
            return order_no, True
        else:
            error_msg = response.get('emsg', 'Unknown error') if isinstance(response, dict) else str(response)
            st.error(f"❌ Order failed for {symbol}: {error_msg}")
            logging.error(f"Order failed: {symbol} - {error_msg}")
            return None, False
            
    except Exception as e:
        st.error(f"Exception placing order for {symbol}: {e}")
        logging.error(f"Exception placing order: {symbol} - {e}")
        return None, False

def get_current_position(symbol, api):
    """Get current position for a symbol"""
    try:
        positions = api.get_positions()
        if isinstance(positions, list):
            for pos in positions:
                if pos.get('tsym') == symbol:
                    net_qty = int(pos.get('netqty', 0))
                    buy_avg = float(pos.get('daybuyavgprc', 0))
                    sell_avg = float(pos.get('daysellavgprc', 0))
                    ltp = float(pos.get('lp', 0))
                    pnl = float(pos.get('rpnl', 0)) + float(pos.get('urmtom', 0))
                    
                    return {
                        'net_qty': net_qty,
                        'buy_avg': buy_avg,
                        'sell_avg': sell_avg,
                        'ltp': ltp,
                        'pnl': pnl,
                        'raw': pos
                    }
        return None
    except Exception as e:
        logging.error(f"Error getting position for {symbol}: {e}")
        return None

def check_profit_target(symbol, position_data):
    """Check if position has reached profit target (5%)"""
    if not position_data or position_data['net_qty'] == 0:
        return False, 0
        
    net_qty = position_data['net_qty']
    current_price = position_data['ltp']
    
    if net_qty > 0:  # Long position
        entry_price = position_data['buy_avg']
        if entry_price > 0:
            profit_pct = (current_price - entry_price) / entry_price * 100
            return profit_pct >= st.session_state.profit_target_pct, profit_pct
    elif net_qty < 0:  # Short position
        entry_price = position_data['sell_avg']
        if entry_price > 0:
            profit_pct = (entry_price - current_price) / entry_price * 100
            return profit_pct >= st.session_state.profit_target_pct, profit_pct
    
    return False, 0

def execute_trade_logic(symbol, signal, current_price, api, position_data=None):
    """Execute trade logic based on bt2 strategy"""
    try:
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Initialize tracking for symbol if not exists
        if symbol not in st.session_state.open_tracked_trades:
            st.session_state.open_tracked_trades[symbol] = {
                'status': 'NONE',
                'entry_price': 0,
                'entry_time': None,
                'quantity': 0,
                'direction': None,
                'target_hit_today': False,
                'last_exit_direction': None,
                'last_exit_date': None
            }
        
        trade_data = st.session_state.open_tracked_trades[symbol]
        
        # Reset daily flags if new day
        if trade_data.get('last_exit_date') != today_str:
            trade_data['target_hit_today'] = False
            trade_data['last_exit_direction'] = None
            trade_data['last_exit_date'] = None
        
        # Get current position from broker
        current_position = get_current_position(symbol, api)
        current_qty = current_position['net_qty'] if current_position else 0
        
        # Check profit target for existing positions
        if current_position and current_qty != 0:
            target_reached, profit_pct = check_profit_target(symbol, current_position)
            if target_reached:
                # Close position due to profit target
                action = 'S' if current_qty > 0 else 'B'
                order_no, success = place_market_order(api, symbol, EXCHANGE, action, abs(current_qty), 
                                                     remarks=f'PROFIT_TARGET_{profit_pct:.2f}%')
                if success:
                    trade_data['target_hit_today'] = True
                    trade_data['last_exit_direction'] = 'long' if current_qty > 0 else 'short'
                    trade_data['last_exit_date'] = today_str
                    st.success(f"🎯 Profit target reached for {symbol}: {profit_pct:.2f}%")
                    return True
        
        # If target was hit today, no new trades allowed (from bt2 logic)
        if trade_data.get('target_hit_today'):
            return False
        
        # Calculate base quantity
        base_quantity = calculate_initial_quantity(st.session_state.predetermined_capital, current_price)
        
        # Execute trade logic based on current position and signal
        if current_qty == 0:  # No position
            # Enter fresh position
            if signal == 'BUY':
                # Check if we can enter long (not same direction as last exit)
                if (trade_data.get('last_exit_direction') != 'long' or 
                    trade_data.get('last_exit_date') != today_str):
                    order_no, success = place_market_order(api, symbol, EXCHANGE, 'B', base_quantity)
                    if success:
                        trade_data.update({
                            'status': 'OPEN',
                            'entry_price': current_price,
                            'entry_time': datetime.datetime.now(),
                            'quantity': base_quantity,
                            'direction': 'LONG',
                            'order_no': order_no
                        })
                        return True
                        
            elif signal == 'SELL':
                # Check if we can enter short (not same direction as last exit)
                if (trade_data.get('last_exit_direction') != 'short' or 
                    trade_data.get('last_exit_date') != today_str):
                    order_no, success = place_market_order(api, symbol, EXCHANGE, 'S', base_quantity)
                    if success:
                        trade_data.update({
                            'status': 'OPEN',
                            'entry_price': current_price,
                            'entry_time': datetime.datetime.now(),
                            'quantity': base_quantity,
                            'direction': 'SHORT',
                            'order_no': order_no
                        })
                        return True
        
        else:  # Have position - check for opposite signal (triple logic from bt2)
            if current_qty > 0 and signal == 'SELL':
                # Long position + short signal = triple quantity
                triple_qty = current_qty * 3
                # Sell total quantity: closes current long + creates short position
                order_no, success = place_market_order(api, symbol, EXCHANGE, 'S', triple_qty, 
                                                     remarks='TRIPLE_REVERSAL_LONG_TO_SHORT')
                if success:
                    trade_data.update({
                        'entry_price': current_price,
                        'entry_time': datetime.datetime.now(),
                        'quantity': triple_qty - current_qty,  # Net short quantity
                        'direction': 'SHORT',
                        'order_no': order_no
                    })
                    st.info(f"🔄 Triple reversal: Long to Short for {symbol}")
                    return True
                    
            elif current_qty < 0 and signal == 'BUY':
                # Short position + long signal = triple quantity
                current_short_qty = abs(current_qty)
                triple_qty = current_short_qty * 3
                # Buy total quantity: closes current short + creates long position
                order_no, success = place_market_order(api, symbol, EXCHANGE, 'B', triple_qty,
                                                     remarks='TRIPLE_REVERSAL_SHORT_TO_LONG')
                if success:
                    trade_data.update({
                        'entry_price': current_price,
                        'entry_time': datetime.datetime.now(),
                        'quantity': triple_qty - current_short_qty,  # Net long quantity
                        'direction': 'LONG',
                        'order_no': order_no
                    })
                    st.info(f"🔄 Triple reversal: Short to Long for {symbol}")
                    return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error executing trade logic for {symbol}: {e}")
        return False

def get_market_start_timestamp():
    """Get timestamp for today's market start (9:15 AM)"""
    today = datetime.datetime.now().date()
    market_start = datetime.datetime.combine(today, MARKET_START_TIME)
    return int(market_start.timestamp())

def calculate_sdvwap_from_market_start(candle_data):
    """Calculate SDVWAP1+ and SDVWAP1- from candle data starting from 9:15 AM."""
    try:
        if len(candle_data) < 3:
            return None, None, None
        
        # Sort candles by timestamp to ensure proper order
        sorted_candles = sorted(candle_data, key=lambda x: int(x.get('ssboe', 0)))
        
        # Filter candles to start from market opening (9:15 AM)
        market_start_ts = get_market_start_timestamp()
        filtered_candles = []
        
        for candle in sorted_candles:
            candle_time = int(candle.get('ssboe', 0))
            if candle_time >= market_start_ts:
                filtered_candles.append(candle)
        
        if len(filtered_candles) < 3:
            return None, None, None
        
        prices = []
        volumes = []
        candle_details = []
        
        for candle in filtered_candles:
            high = float(candle.get('inth', 0))
            low = float(candle.get('intl', 0))
            close = float(candle.get('intc', 0))
            open_price = float(candle.get('into', 0))
            volume = float(candle.get('intv', 0))
            timestamp = int(candle.get('ssboe', 0))
            
            if high > 0 and low > 0 and close > 0 and volume > 0:
                typical_price = (high + low + close) / 3
                prices.append(typical_price)
                volumes.append(volume)
                candle_details.append({
                    'timestamp': timestamp,
                    'time': datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M'),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'typical_price': typical_price
                })
        
        if len(prices) < 3:
            return None, None, None
        
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        # Calculate VWAP
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # Calculate weighted standard deviation
        weighted_mean = vwap
        weighted_variance = np.sum(volumes * (prices - weighted_mean) ** 2) / np.sum(volumes)
        weighted_std = np.sqrt(weighted_variance)
        
        # SDVWAP bands
        sdvwap_plus = vwap + weighted_std
        sdvwap_minus = vwap - weighted_std
        
        calculation_details = {
            'vwap': vwap,
            'std_dev': weighted_std,
            'sdvwap_plus': sdvwap_plus,
            'sdvwap_minus': sdvwap_minus,
            'total_candles': len(filtered_candles),
            'candle_details': candle_details
        }
        
        return sdvwap_plus, sdvwap_minus, calculation_details
        
    except Exception as e:
        logging.error(f"Error calculating SDVWAP: {e}")
        return None, None, None

def calculate_initial_quantity(predetermined_capital, stock_price):
    """Calculate initial quantity as predetermined_capital/stock_price"""
    if stock_price <= 0:
        return 0
    return max(1, int(predetermined_capital / stock_price))

def is_trading_time():
    """Check if current time is within trading hours (9:30 AM to 3:20 PM)"""
    current_time = datetime.datetime.now().time()
    return st.session_state.trading_start_time <= current_time <= st.session_state.trading_end_time

def screen_and_trade_stock_sdvwap(stock_info, api, cash_margin, auto_trade=False):
    """Screen a stock and optionally execute trades based on SDVWAP strategy."""
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']

    if not is_trading_time():
        return tradingsymbol, 'NEUTRAL', 'Outside trading hours', None, None, None, False

    # Get candles from market start (9:15 AM) to current time
    market_start_ts = get_market_start_timestamp()
    current_ts = int(datetime.datetime.now().timestamp())
    
    try:
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=market_start_ts,
            endtime=current_ts,
            interval=CANDLE_INTERVAL
        )

        if not candle_data or len(candle_data) < 3:
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None, None, False

        sdvwap_plus, sdvwap_minus, calculation_details = calculate_sdvwap_from_market_start(candle_data)
        
        if sdvwap_plus is None or sdvwap_minus is None:
            return tradingsymbol, 'NEUTRAL', 'Could not calculate SDVWAP', None, None, None, False

        # Get the last 3 candles for signal detection
        sorted_candles = sorted(candle_data, key=lambda x: int(x.get('ssboe', 0)))
        
        if len(sorted_candles) < 3:
            return tradingsymbol, 'NEUTRAL', 'Need at least 3 candles', None, None, None, False

        # Current candle (most recent)
        current_candle = sorted_candles[-1]
        # Previous candle (C-1)
        candle_minus_1 = sorted_candles[-2]
        # Candle before that (C-2)  
        candle_minus_2 = sorted_candles[-3]

        # Extract OHLC values
        c2_open = float(candle_minus_2.get('into', 0))
        c2_close = float(candle_minus_2.get('intc', 0))
        c1_open = float(candle_minus_1.get('into', 0))
        c1_close = float(candle_minus_1.get('intc', 0))
        
        current_price = float(current_candle.get('intc', 0))

        # SDVWAP Buy Condition: C-2 and C-1 both green candles above SDVWAP+
        c2_is_green = c2_close > c2_open
        c2_above_sdvwap_plus = c2_close > sdvwap_plus
        c1_is_green = c1_close > c1_open  
        c1_above_sdvwap_plus = c1_close > sdvwap_plus
        
        buy_condition = c2_is_green and c2_above_sdvwap_plus and c1_is_green and c1_above_sdvwap_plus
        
        # SDVWAP Sell Condition: C-2 and C-1 both red candles below SDVWAP-
        c2_is_red = c2_close < c2_open
        c2_below_sdvwap_minus = c2_close < sdvwap_minus
        c1_is_red = c1_close < c1_open
        c1_below_sdvwap_minus = c1_close < sdvwap_minus
        
        sell_condition = c2_is_red and c2_below_sdvwap_minus and c1_is_red and c1_below_sdvwap_minus

        # Create detailed condition parameters
        condition_params = {
            'sdvwap_plus': sdvwap_plus,
            'sdvwap_minus': sdvwap_minus,
            'vwap': calculation_details['vwap'],
            'std_dev': calculation_details['std_dev'],
            'c2_open': c2_open,
            'c2_close': c2_close,
            'c1_open': c1_open,
            'c1_close': c1_close,
            'current_price': current_price,
            'buy_condition_met': buy_condition,
            'sell_condition_met': sell_condition,
            'candle_times': {
                'c2_time': datetime.datetime.fromtimestamp(int(candle_minus_2.get('ssboe', 0))).strftime('%H:%M'),
                'c1_time': datetime.datetime.fromtimestamp(int(candle_minus_1.get('ssboe', 0))).strftime('%H:%M'),
                'current_time': datetime.datetime.fromtimestamp(int(current_candle.get('ssboe', 0))).strftime('%H:%M')
            }
        }

        # Execute trade logic if conditions are met
        trade_executed = False
        if buy_condition and auto_trade:
            trade_executed = execute_trade_logic(tradingsymbol, 'BUY', current_price, api)
        elif sell_condition and auto_trade:
            trade_executed = execute_trade_logic(tradingsymbol, 'SELL', current_price, api)

        # Determine signal and reason
        if buy_condition:
            reason = f'BUY Signal: C-2({condition_params["candle_times"]["c2_time"]})={c2_close:.2f}>{sdvwap_plus:.2f}, C-1({condition_params["candle_times"]["c1_time"]})={c1_close:.2f}>{sdvwap_plus:.2f}'
            return tradingsymbol, 'BUY', reason, current_price, calculate_initial_quantity(st.session_state.predetermined_capital, current_price), condition_params, trade_executed
        elif sell_condition:
            reason = f'SELL Signal: C-2({condition_params["candle_times"]["c2_time"]})={c2_close:.2f}<{sdvwap_minus:.2f}, C-1({condition_params["candle_times"]["c1_time"]})={c1_close:.2f}<{sdvwap_minus:.2f}'
            return tradingsymbol, 'SELL', reason, current_price, calculate_initial_quantity(st.session_state.predetermined_capital, current_price), condition_params, trade_executed
        else:
            return tradingsymbol, 'NEUTRAL', 'No SDVWAP signal', None, None, condition_params, False

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}")
        return tradingsymbol, 'NEUTRAL', f'Error: {e}', None, None, None, False

def get_account_info(api):
    """Get account balance and margin information"""
    try:
        limits = api.get_limits()
        if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
            cash_margin = None
            
            # Try to get cash from top level
            if 'cash' in limits and limits['cash'] is not None:
                try:
                    cash_margin = float(limits['cash'])
                except ValueError:
                    pass
            
            # Try to get from prange if not found at top level
            if cash_margin is None and 'prange' in limits and isinstance(limits['prange'], list):
                for item in limits['prange']:
                    if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                        try:
                            cash_margin = float(item['cash'])
                            break
                        except ValueError:
                            continue
            
            return {
                'cash_margin': cash_margin,
                'status': 'success',
                'raw_data': limits
            }
        else:
            return {
                'cash_margin': None,
                'status': 'failed',
                'error': limits.get('emsg', 'Unknown error') if isinstance(limits, dict) else str(limits)
            }
    except Exception as e:
        return {
            'cash_margin': None,
            'status': 'error',
            'error': str(e)
        }

def get_positions(api):
    """Get positions from broker"""
    try:
        positions = api.get_positions()
        if isinstance(positions, list):
            position_data = []
            for pos in positions:
                net_qty = int(pos.get('netqty', 0))
                if net_qty != 0:  # Only show non-zero positions
                    position_data.append({
                        'Symbol': pos.get('tsym', 'N/A'),
                        'Exchange': pos.get('exch', 'N/A'),
                        'Product': pos.get('prd', 'N/A'),
                        'Net Qty': net_qty,
                        'Buy Avg': float(pos.get('daybuyavgprc', 0)),
                        'Sell Avg': float(pos.get('daysellavgprc', 0)),
                        'LTP': float(pos.get('lp', 0)),
                        'PNL': float(pos.get('rpnl', 0)) + float(pos.get('urmtom', 0)),
                        'Raw': pos  # Keep raw data for exit operations
                    })
            return position_data
        return []
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return []

def exit_all_positions_at_time(api):
    """Exit all positions at 3:20 PM"""
    current_time = datetime.datetime.now().time()
    if current_time >= st.session_state.trading_end_time:
        positions = get_positions(api)
        for pos in positions:
            symbol = pos['Symbol']
            exchange = pos['Exchange']
            net_qty = pos['Net Qty']
            
            if net_qty != 0:
                action = 'S' if net_qty > 0 else 'B'
                order_no, success = place_market_order(api, symbol, exchange, action, abs(net_qty), 
                                                     remarks='TIME_EXIT_3_20_PM')
                if success:
                    st.info(f"🕐 Time exit: Closed {symbol} position")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Enhanced SDVWAP Strategy with BT2 Logic")
st.title("Enhanced SDVWAP Trading Strategy with BT2 Logic")
st.markdown("SDVWAP-based trading with triple reversal logic and 5% profit targets")

# Sidebar
st.sidebar.header("Strategy Settings")
auto_trade_enabled = st.sidebar.checkbox("Enable Auto Trading", value=False)
screen_interval = st.sidebar.slider("Screening Interval (seconds)", min_value=5, max_value=60, value=30)
run_strategy = st.sidebar.checkbox("Run Strategy Continuously", value=False)
show_detailed_params = st.sidebar.checkbox("Show Detailed Parameters", value=True)

st.sidebar.subheader("Trading Parameters")
st.session_state.predetermined_capital = st.sidebar.number_input(
    "Capital per Trade (INR)",
    min_value=100,
    max_value=10000000,
    value=int(st.session_state.predetermined_capital),
    step=1000
)

st.session_state.profit_target_pct = st.sidebar.number_input(
    "Profit Target (%)",
    min_value=1.0,
    max_value=20.0,
    value=st.session_state.profit_target_pct,
    step=0.5
)

# Display current time and trading status
current_time = datetime.datetime.now().time()
st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**Trading Hours:** {st.session_state.trading_start_time} - {st.session_state.trading_end_time}")
st.sidebar.markdown(f"**Profit Target:** {st.session_state.profit_target_pct}%")

if is_trading_time():
    st.sidebar.success("Within Trading Hours")
else:
    st.sidebar.warning("Outside Trading Hours")

# Account Information Section
st.header("Account Information")
account_info = get_account_info(api)
if account_info['status'] == 'success' and account_info['cash_margin'] is not None:
    st.success(f"**Available Cash Margin:** ₹{account_info['cash_margin']:,.2f}")
    cash_margin = account_info['cash_margin']
else:
    st.error(f"Could not fetch account info: {account_info.get('error', 'Unknown error')}")
    cash_margin = 0

st.markdown("---")

# Load symbols
nifty500_symbols = load_symbols_from_csv()
if not nifty500_symbols:
    st.stop()

# Main Strategy Section
st.header("SDVWAP Signals with Auto Trading")

if auto_trade_enabled:
    st.warning("🚨 AUTO TRADING IS ENABLED - Orders will be placed automatically!")

signals_placeholder = st.empty()

# Manual screening and trading
col1, col2 = st.columns(2)
with col1:
    if st.button("Run Manual SDVWAP Screening", key="manual_sdvwap_screen"):
        buy_signals = []
        sell_signals = []
        trade_executions = []
        
        progress_bar = st.progress(0, text="Running SDVWAP screening from 9:15 AM...")
        screening_symbols = nifty500_symbols[:20]  # Test with first 20 symbols
        
        for i, stock in enumerate(screening_symbols):
            symbol, signal, reason, price, quantity, params, trade_executed = screen_and_trade_stock_sdvwap(
                stock, api, cash_margin, auto_trade=auto_trade_enabled)
            
            if signal == 'BUY' and price and quantity and params:
                signal_data = {
                    'Symbol': symbol,
                    'Signal': 'BUY',
                    'Price': price,
                    'Quantity': quantity,
                    'Reason': reason,
                    'Parameters': params,
                    'Trade Executed': trade_executed
                }
                buy_signals.append(signal_data)
                if trade_executed:
                    trade_executions.append(f"BUY {symbol} at ₹{price:.2f}")
                
            elif signal == 'SELL' and price and quantity and params:
                signal_data = {
                    'Symbol': symbol,
                    'Signal': 'SELL', 
                    'Price': price,
                    'Quantity': quantity,
                    'Reason': reason,
                    'Parameters': params,
                    'Trade Executed': trade_executed
                }
                sell_signals.append(signal_data)
                if trade_executed:
                    trade_executions.append(f"SELL {symbol} at ₹{price:.2f}")
            
            progress_bar.progress((i + 1) / len(screening_symbols), text=f"Screening {stock['tsym']}...")
        
        progress_bar.empty()
        
        # Show trade executions summary
        if trade_executions:
            st.success(f"Trades Executed: {', '.join(trade_executions)}")
        
        # Display results with parameter details
        with signals_placeholder.container():
            st.subheader("SDVWAP Screening Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Buy Signals**")
                if buy_signals:
                    for i, signal in enumerate(buy_signals):
                        status_icon = "✅" if signal['Trade Executed'] else "⏸️"
                        with st.expander(f"{status_icon} {signal['Symbol']} - BUY at ₹{signal['Price']:.2f}"):
                            st.write(f"**Quantity:** {signal['Quantity']}")
                            st.write(f"**Order Value:** ₹{signal['Price'] * signal['Quantity']:,.2f}")
                            st.write(f"**Trade Status:** {'Executed' if signal['Trade Executed'] else 'Signal Only'}")
                            st.write(f"**Reason:** {signal['Reason']}")
                            
                            if show_detailed_params and signal['Parameters']:
                                params = signal['Parameters']
                                st.write("**Detailed Parameters:**")
                                st.write(f"• VWAP: {params['vwap']:.2f}")
                                st.write(f"• Std Dev: {params['std_dev']:.2f}")
                                st.write(f"• SDVWAP+: {params['sdvwap_plus']:.2f}")
                                st.write(f"• SDVWAP-: {params['sdvwap_minus']:.2f}")
                                
                                st.write("**Candle Analysis:**")
                                st.write(f"• C-2 ({params['candle_times']['c2_time']}): Open={params['c2_open']:.2f}, Close={params['c2_close']:.2f}")
                                st.write(f"• C-1 ({params['candle_times']['c1_time']}): Open={params['c1_open']:.2f}, Close={params['c1_close']:.2f}")
                else:
                    st.info("No buy signals found")
            
            with col2:
                st.write("**Sell Signals**")
                if sell_signals:
                    for i, signal in enumerate(sell_signals):
                        status_icon = "✅" if signal['Trade Executed'] else "⏸️"
                        with st.expander(f"{status_icon} {signal['Symbol']} - SELL at ₹{signal['Price']:.2f}"):
                            st.write(f"**Quantity:** {signal['Quantity']}")
                            st.write(f"**Order Value:** ₹{signal['Price'] * signal['Quantity']:,.2f}")
                            st.write(f"**Trade Status:** {'Executed' if signal['Trade Executed'] else 'Signal Only'}")
                            st.write(f"**Reason:** {signal['Reason']}")
                            
                            if show_detailed_params and signal['Parameters']:
                                params = signal['Parameters']
                                st.write("**Detailed Parameters:**")
                                st.write(f"• VWAP: {params['vwap']:.2f}")
                                st.write(f"• Std Dev: {params['std_dev']:.2f}")
                                st.write(f"• SDVWAP+: {params['sdvwap_plus']:.2f}")
                                st.write(f"• SDVWAP-: {params['sdvwap_minus']:.2f}")
                                
                                st.write("**Candle Analysis:**")
                                st.write(f"• C-2 ({params['candle_times']['c2_time']}): Open={params['c2_open']:.2f}, Close={params['c2_close']:.2f}")
                                st.write(f"• C-1 ({params['candle_times']['c1_time']}): Open={params['c1_open']:.2f}, Close={params['c1_close']:.2f}")
                else:
                    st.info("No sell signals found")

with col2:
    if st.button("Exit All Positions (Emergency)", key="emergency_exit"):
        positions = get_positions(api)
        exit_count = 0
        for pos in positions:
            symbol = pos['Symbol']
            exchange = pos['Exchange']
            net_qty = pos['Net Qty']
            
            if net_qty != 0:
                action = 'S' if net_qty > 0 else 'B'
                order_no, success = place_market_order(api, symbol, exchange, action, abs(net_qty), 
                                                     remarks='EMERGENCY_EXIT')
                if success:
                    exit_count += 1
        
        if exit_count > 0:
            st.success(f"Emergency exit orders placed for {exit_count} positions")
        else:
            st.info("No positions to exit")

st.markdown("---")

# Continuous Strategy Runner
if run_strategy:
    st.header("Continuous Strategy Runner")
    
    strategy_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if 'strategy_running' not in st.session_state:
        st.session_state.strategy_running = True
    
    # Auto exit at 3:20 PM
    exit_all_positions_at_time(api)
    
    while st.session_state.strategy_running and run_strategy:
        with status_placeholder.container():
            current_time = datetime.datetime.now()
            st.write(f"**Strategy Running - {current_time.strftime('%H:%M:%S')}**")
            
            if is_trading_time():
                st.success("Within trading hours - Scanning for signals...")
                
                # Run screening on a subset of symbols
                screening_symbols = nifty500_symbols[:10]  # Monitor top 10 symbols
                signals_found = []
                
                for stock in screening_symbols:
                    symbol, signal, reason, price, quantity, params, trade_executed = screen_and_trade_stock_sdvwap(
                        stock, api, cash_margin, auto_trade=auto_trade_enabled)
                    
                    if signal in ['BUY', 'SELL']:
                        signals_found.append({
                            'symbol': symbol,
                            'signal': signal,
                            'price': price,
                            'executed': trade_executed,
                            'time': current_time.strftime('%H:%M:%S')
                        })
                
                if signals_found:
                    st.write("**Recent Signals:**")
                    for sig in signals_found[-5:]:  # Show last 5 signals
                        status = "✅ Executed" if sig['executed'] else "⏸️ Signal Only"
                        st.write(f"• {sig['time']} - {sig['signal']} {sig['symbol']} at ₹{sig['price']:.2f} {status}")
                else:
                    st.write("No signals detected in current scan")
                    
            else:
                st.warning("Outside trading hours")
                
        time.sleep(screen_interval)

st.markdown("---")

# Trading Books Section
st.header("Trading Books & Positions")

# Create tabs for different books
tab1, tab2, tab3 = st.tabs(["Open Positions", "Strategy Trades", "Profit Monitoring"])

with tab1:
    st.subheader("Broker Open Positions")
    positions_data = get_positions(api)
    
    if positions_data:
        for idx, pos in enumerate(positions_data):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                pnl_color = "🟢" if pos['PNL'] >= 0 else "🔴"
                st.write(f"{pnl_color} **{pos['Symbol']}** ({pos['Exchange']})")
                st.write(f"Qty: {pos['Net Qty']}, LTP: ₹{pos['LTP']:.2f}")
            
            with col2:
                st.write(f"P&L: ₹{pos['PNL']:.2f}")
                
                # Check profit target status
                position_data = {
                    'net_qty': pos['Net Qty'],
                    'buy_avg': pos['Buy Avg'],
                    'sell_avg': pos['Sell Avg'],
                    'ltp': pos['LTP']
                }
                target_reached, profit_pct = check_profit_target(pos['Symbol'], position_data)
                
                if target_reached:
                    st.success(f"🎯 Target Reached: {profit_pct:.1f}%")
                else:
                    st.write(f"Progress: {profit_pct:.1f}% / {st.session_state.profit_target_pct}%")
            
            with col3:
                if st.button(f"Close", key=f"close_pos_{idx}_{pos['Symbol']}"):
                    action = 'S' if pos['Net Qty'] > 0 else 'B'
                    order_no, success = place_market_order(api, pos['Symbol'], pos['Exchange'], 
                                                         action, abs(pos['Net Qty']), remarks='MANUAL_CLOSE')
                    if success:
                        st.success("Close order placed")
                        time.sleep(1)
                        st.rerun()
        
        # Summary
        total_pnl = sum(pos['PNL'] for pos in positions_data)
        pnl_color = "🟢" if total_pnl >= 0 else "🔴"
        st.markdown(f"**Total P&L: {pnl_color} ₹{total_pnl:,.2f}**")
    else:
        st.info("No open positions found")

with tab2:
    st.subheader("Strategy Active Trades")
    active_trades = {k: v for k, v in st.session_state.open_tracked_trades.items() if v['status'] == 'OPEN'}
    
    if active_trades:
        trades_data = []
        for symbol, trade in active_trades.items():
            trades_data.append({
                'Symbol': symbol,
                'Direction': 'BUY' if trade['direction'] == 'LONG' else 'SELL',
                'Quantity': trade['quantity'],
                'Entry Price': f"₹{trade['entry_price']:.2f}",
                'Entry Time': trade['entry_time'].strftime('%H:%M:%S'),
                'Target Hit Today': trade.get('target_hit_today', False),
                'Order No': trade.get('order_no', 'N/A')
            })
        df_strategy_trades = pd.DataFrame(trades_data)
        st.dataframe(df_strategy_trades, use_container_width=True)
    else:
        st.info("No active strategy trades")

with tab3:
    st.subheader("Profit Target Monitoring")
    
    # Monitor all positions for profit targets
    positions_data = get_positions(api)
    if positions_data:
        for pos in positions_data:
            position_data = {
                'net_qty': pos['Net Qty'],
                'buy_avg': pos['Buy Avg'],
                'sell_avg': pos['Sell Avg'],
                'ltp': pos['LTP']
            }
            target_reached, profit_pct = check_profit_target(pos['Symbol'], position_data)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                direction = "LONG" if pos['Net Qty'] > 0 else "SHORT"
                st.write(f"**{pos['Symbol']}** ({direction})")
                
            with col2:
                if target_reached:
                    st.success(f"🎯 {profit_pct:.1f}%")
                else:
                    progress = min(profit_pct / st.session_state.profit_target_pct, 1.0)
                    st.progress(progress, text=f"{profit_pct:.1f}%")
                    
            with col3:
                if target_reached and auto_trade_enabled:
                    st.write("Auto-exit enabled")
                elif target_reached:
                    if st.button(f"Exit", key=f"target_exit_{pos['Symbol']}"):
                        action = 'S' if pos['Net Qty'] > 0 else 'B'
                        order_no, success = place_market_order(api, pos['Symbol'], pos['Exchange'], 
                                                             action, abs(pos['Net Qty']), 
                                                             remarks=f'PROFIT_TARGET_{profit_pct:.1f}%')
                        if success:
                            st.success("Profit exit order placed")
                            st.rerun()
    else:
        st.info("No positions to monitor")

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Capital per Trade",
        value=f"₹{st.session_state.predetermined_capital:,}"
    )

with col2:
    st.metric(
        label="Available Margin", 
        value=f"₹{cash_margin:,.0f}" if cash_margin else "N/A"
    )

with col3:
    st.metric(
        label="Profit Target",
        value=f"{st.session_state.profit_target_pct}%"
    )

with col4:
    active_positions = len(get_positions(api))
    st.metric(
        label="Active Positions",
        value=active_positions
    )
