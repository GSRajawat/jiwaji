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
MARKET_START_TIME = datetime.time(3, 45)  # Market starts at 9:15 AM

# Initialize session state variables
if 'predetermined_capital' not in st.session_state:
    st.session_state.predetermined_capital = 100000
if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {}
if 'daily_profit_exits' not in st.session_state:
    st.session_state.daily_profit_exits = {}  # Track profit exits per stock per day
if 'trading_start_time' not in st.session_state:
    st.session_state.trading_start_time = datetime.time(4, 00)
if 'trading_end_time' not in st.session_state:
    st.session_state.trading_end_time = datetime.time(10, 00)
if 'exit_time' not in st.session_state:
    st.session_state.exit_time = datetime.time(9, 45)

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

def get_order_book(api):
    """Get order book from broker"""
    try:
        orders = api.get_order_book()
        if isinstance(orders, list):
            order_data = []
            for order in orders:
                order_data.append({
                    'Order No': order.get('norenordno', 'N/A'),
                    'Symbol': order.get('tsym', 'N/A'),
                    'Exchange': order.get('exch', 'N/A'),
                    'Action': order.get('trantype', 'N/A'),
                    'Product': order.get('prd', 'N/A'),
                    'Quantity': int(order.get('qty', 0)),
                    'Price': float(order.get('prc', 0)),
                    'Status': order.get('status', 'N/A'),
                    'Time': order.get('norentm', 'N/A')
                })
            return order_data
        return []
    except Exception as e:
        st.error(f"Error fetching order book: {e}")
        return []

def get_trade_book(api):
    """Get trade book from broker"""
    try:
        trades = api.get_trade_book()
        if isinstance(trades, list):
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'Order No': trade.get('norenordno', 'N/A'),
                    'Symbol': trade.get('tsym', 'N/A'),
                    'Exchange': trade.get('exch', 'N/A'),
                    'Action': trade.get('trantype', 'N/A'),
                    'Product': trade.get('prd', 'N/A'),
                    'Quantity': int(trade.get('qty', 0)),
                    'Price': float(trade.get('flprc', 0)),
                    'Time': trade.get('fltm', 'N/A')
                })
            return trade_data
        return []
    except Exception as e:
        st.error(f"Error fetching trade book: {e}")
        return []

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
                        'PNL': float(pos.get('rpnl', 0)) + float(pos.get('urmtm', 0)),
                        'Raw': pos  # Keep raw data for exit operations
                    })
            return position_data
        return []
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return []

def exit_broker_position(exchange, tradingsymbol, product_type, netqty, api):
    """Exit a broker position"""
    try:
        st.info(f"Exiting position: {tradingsymbol} (Qty: {netqty})")
        response = api.place_order(
            buy_or_sell='S' if netqty > 0 else 'B',  # Opposite action
            product_type=product_type,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=abs(int(netqty)),
            discloseqty=0,
            price_type='MKT',  # Market order for quick exit
            price=0,
            trigger_price=None,
            retention='DAY',
            remarks='Manual_Position_Exit'
        )
        
        if response and response.get('stat') == 'Ok':
            st.success(f"Exit order placed for {tradingsymbol}: {response.get('norenordno')}")
            return True
        else:
            st.error(f"Failed to exit {tradingsymbol}: {response.get('emsg', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Error exiting position {tradingsymbol}: {e}")
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

def is_trading_time():
    """Check if current time is within trading hours"""
    current_time = datetime.datetime.now().time()
    return st.session_state.trading_start_time <= current_time <= st.session_state.trading_end_time

def is_exit_time():
    """Check if it's time to exit all positions"""
    current_time = datetime.datetime.now().time()
    return current_time >= st.session_state.exit_time

def calculate_initial_quantity(predetermined_capital, stock_price):
    """Calculate initial quantity as predetermined_capital/stock_price"""
    if stock_price <= 0:
        return 0
    return max(1, int(predetermined_capital / stock_price))

def check_margin_requirement(order_value, cash_margin):
    """Check if order value/4 is less than available cash margin"""
    if cash_margin is None or cash_margin <= 0:
        return False, f"No cash margin available"
    
    margin_required = order_value / 4
    if margin_required <= cash_margin:
        return True, f"Margin OK: Required ₹{margin_required:,.2f}, Available ₹{cash_margin:,.2f}"
    else:
        return False, f"Insufficient margin: Required ₹{margin_required:,.2f}, Available ₹{cash_margin:,.2f}"

def screen_stock_sdvwap(stock_info, api, cash_margin):
    """Screen a stock based on SDVWAP strategy with detailed condition parameters."""
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']

    if not is_trading_time():
        return tradingsymbol, 'NEUTRAL', 'Outside trading hours', None, None, None

    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    profit_exit_restriction = None
    if tradingsymbol in st.session_state.daily_profit_exits:
        if today_str in st.session_state.daily_profit_exits[tradingsymbol]:
            profit_exit_restriction = st.session_state.daily_profit_exits[tradingsymbol][today_str]
    
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
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None, None

        sdvwap_plus, sdvwap_minus, calculation_details = calculate_sdvwap_from_market_start(candle_data)
        
        if sdvwap_plus is None or sdvwap_minus is None:
            return tradingsymbol, 'NEUTRAL', 'Could not calculate SDVWAP', None, None, None

        # Get the last 3 candles for signal detection
        sorted_candles = sorted(candle_data, key=lambda x: int(x.get('ssboe', 0)))
        
        if len(sorted_candles) < 3:
            return tradingsymbol, 'NEUTRAL', 'Need at least 3 candles', None, None, None

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
        
        # Calculate order value and check margin
        initial_quantity = calculate_initial_quantity(st.session_state.predetermined_capital, current_price)
        order_value = initial_quantity * current_price
        margin_ok, margin_msg = check_margin_requirement(order_value, cash_margin)

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
            'c2_is_green': c2_is_green,
            'c2_above_sdvwap_plus': c2_above_sdvwap_plus,
            'c1_is_green': c1_is_green,
            'c1_above_sdvwap_plus': c1_above_sdvwap_plus,
            'c2_is_red': c2_is_red,
            'c2_below_sdvwap_minus': c2_below_sdvwap_minus,
            'c1_is_red': c1_is_red,
            'c1_below_sdvwap_minus': c1_below_sdvwap_minus,
            'buy_condition_met': buy_condition,
            'sell_condition_met': sell_condition,
            'candle_times': {
                'c2_time': datetime.datetime.fromtimestamp(int(candle_minus_2.get('ssboe', 0))).strftime('%H:%M'),
                'c1_time': datetime.datetime.fromtimestamp(int(candle_minus_1.get('ssboe', 0))).strftime('%H:%M'),
                'current_time': datetime.datetime.fromtimestamp(int(current_candle.get('ssboe', 0))).strftime('%H:%M')
            }
        }

        # Apply profit exit restrictions
        if profit_exit_restriction == 'BUY':
            buy_condition = False
        elif profit_exit_restriction == 'SELL':
            sell_condition = False

        if buy_condition and margin_ok:
            reason = f'BUY Signal: C-2({condition_params["candle_times"]["c2_time"]})={c2_close:.2f}>{sdvwap_plus:.2f}, C-1({condition_params["candle_times"]["c1_time"]})={c1_close:.2f}>{sdvwap_plus:.2f} | VWAP={calculation_details["vwap"]:.2f}, StdDev={calculation_details["std_dev"]:.2f}'
            return tradingsymbol, 'BUY', reason, current_price, initial_quantity, condition_params
        elif sell_condition and margin_ok:
            reason = f'SELL Signal: C-2({condition_params["candle_times"]["c2_time"]})={c2_close:.2f}<{sdvwap_minus:.2f}, C-1({condition_params["candle_times"]["c1_time"]})={c1_close:.2f}<{sdvwap_minus:.2f} | VWAP={calculation_details["vwap"]:.2f}, StdDev={calculation_details["std_dev"]:.2f}'
            return tradingsymbol, 'SELL', reason, current_price, initial_quantity, condition_params
        elif (buy_condition or sell_condition) and not margin_ok:
            return tradingsymbol, 'NEUTRAL', f'Signal detected but {margin_msg}', current_price, initial_quantity, condition_params
        else:
            return tradingsymbol, 'NEUTRAL', 'No SDVWAP signal', None, None, condition_params

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}")
        return tradingsymbol, 'NEUTRAL', f'Error: {e}', None, None, None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Enhanced SDVWAP Strategy")
st.title("📈 Enhanced SDVWAP Trading Strategy")
st.markdown("SDVWAP-based trading starting from 9:15 AM with detailed signal parameters")

# Sidebar
st.sidebar.header("Strategy Settings")
screen_interval = st.sidebar.slider("Screening Interval (seconds)", min_value=5, max_value=60, value=30)
run_strategy = st.sidebar.checkbox("Run Strategy Continuously", value=False)
show_detailed_params = st.sidebar.checkbox("Show Detailed Parameters", value=True)

st.sidebar.subheader("Trading Parameters")
st.session_state.predetermined_capital = st.sidebar.number_input(
    "Predetermined Capital per Trade (INR)",
    min_value=100,
    max_value=10000000,
    value=int(st.session_state.predetermined_capital),
    step=1000
)

# Display current time and trading status
current_time = datetime.datetime.now().time()
st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**Market Start:** {MARKET_START_TIME}")
st.sidebar.markdown(f"**Trading Hours:** {st.session_state.trading_start_time} - {st.session_state.trading_end_time}")

if is_trading_time():
    st.sidebar.success("✅ Within Trading Hours")
elif is_exit_time():
    st.sidebar.error("🔴 Exit Time")
else:
    st.sidebar.warning("⏸️ Outside Trading Hours")

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
st.header("SDVWAP Signals with Parameter Details")
signals_placeholder = st.empty()

# Manual screening for demonstration
if st.button("🔍 Run Manual SDVWAP Screening", key="manual_sdvwap_screen"):
    buy_signals = []
    sell_signals = []
    
    progress_bar = st.progress(0, text="Running SDVWAP screening from 9:15 AM...")
    screening_symbols = nifty500_symbols[:20]  # Test with first 20 symbols
    
    for i, stock in enumerate(screening_symbols):
        symbol, signal, reason, price, quantity, params = screen_stock_sdvwap(stock, api, cash_margin)
        
        if signal == 'BUY' and price and quantity and params:
            signal_data = {
                'Symbol': symbol,
                'Signal': 'BUY',
                'Price': price,
                'Quantity': quantity,
                'Reason': reason,
                'Parameters': params
            }
            buy_signals.append(signal_data)
            
        elif signal == 'SELL' and price and quantity and params:
            signal_data = {
                'Symbol': symbol,
                'Signal': 'SELL', 
                'Price': price,
                'Quantity': quantity,
                'Reason': reason,
                'Parameters': params
            }
            sell_signals.append(signal_data)
        
        progress_bar.progress((i + 1) / len(screening_symbols), text=f"Screening {stock['tsym']}...")
    
    progress_bar.empty()
    
    # Display results with parameter details
    with signals_placeholder.container():
        st.subheader("🔍 SDVWAP Screening Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🟢 **Buy Signals**")
            if buy_signals:
                for i, signal in enumerate(buy_signals):
                    with st.expander(f"📈 {signal['Symbol']} - BUY at ₹{signal['Price']:.2f}"):
                        st.write(f"**Quantity:** {signal['Quantity']}")
                        st.write(f"**Order Value:** ₹{signal['Price'] * signal['Quantity']:,.2f}")
                        st.write(f"**Reason:** {signal['Reason']}")
                        
                        if show_detailed_params and signal['Parameters']:
                            params = signal['Parameters']
                            st.write("**📊 Detailed Parameters:**")
                            
                            # SDVWAP Values
                            st.write(f"• VWAP: {params['vwap']:.2f}")
                            st.write(f"• Std Dev: {params['std_dev']:.2f}")
                            st.write(f"• SDVWAP+: {params['sdvwap_plus']:.2f}")
                            st.write(f"• SDVWAP-: {params['sdvwap_minus']:.2f}")
                            
                            # Candle Analysis
                            st.write("**🕯️ Candle Analysis:**")
                            st.write(f"• C-2 ({params['candle_times']['c2_time']}): Open={params['c2_open']:.2f}, Close={params['c2_close']:.2f} {'✅Green' if params['c2_is_green'] else '❌Red'}")
                            st.write(f"• C-1 ({params['candle_times']['c1_time']}): Open={params['c1_open']:.2f}, Close={params['c1_close']:.2f} {'✅Green' if params['c1_is_green'] else '❌Red'}")
                            
                            # Condition Checks
                            st.write("**✅ Buy Conditions:**")
                            st.write(f"• C-2 Green & > SDVWAP+: {params['c2_is_green']} & {params['c2_above_sdvwap_plus']}")
                            st.write(f"• C-1 Green & > SDVWAP+: {params['c1_is_green']} & {params['c1_above_sdvwap_plus']}")
            else:
                st.info("No buy signals found")
        
        with col2:
            st.write("🔴 **Sell Signals**")
            if sell_signals:
                for i, signal in enumerate(sell_signals):
                    with st.expander(f"📉 {signal['Symbol']} - SELL at ₹{signal['Price']:.2f}"):
                        st.write(f"**Quantity:** {signal['Quantity']}")
                        st.write(f"**Order Value:** ₹{signal['Price'] * signal['Quantity']:,.2f}")
                        st.write(f"**Reason:** {signal['Reason']}")
                        
                        if show_detailed_params and signal['Parameters']:
                            params = signal['Parameters']
                            st.write("**📊 Detailed Parameters:**")
                            
                            # SDVWAP Values
                            st.write(f"• VWAP: {params['vwap']:.2f}")
                            st.write(f"• Std Dev: {params['std_dev']:.2f}")
                            st.write(f"• SDVWAP+: {params['sdvwap_plus']:.2f}")
                            st.write(f"• SDVWAP-: {params['sdvwap_minus']:.2f}")
                            
                            # Candle Analysis  
                            st.write("**🕯️ Candle Analysis:**")
                            st.write(f"• C-2 ({params['candle_times']['c2_time']}): Open={params['c2_open']:.2f}, Close={params['c2_close']:.2f} {'❌Red' if params['c2_is_red'] else '✅Green'}")
                            st.write(f"• C-1 ({params['candle_times']['c1_time']}): Open={params['c1_open']:.2f}, Close={params['c1_close']:.2f} {'❌Red' if params['c1_is_red'] else '✅Green'}")
                            
                            # Condition Checks
                            st.write("**❌ Sell Conditions:**")
                            st.write(f"• C-2 Red & < SDVWAP-: {params['c2_is_red']} & {params['c2_below_sdvwap_minus']}")
                            st.write(f"• C-1 Red & < SDVWAP-: {params['c1_is_red']} & {params['c1_below_sdvwap_minus']}")
            else:
                st.info("No sell signals found")

st.markdown("---")

# Trading Books Section
st.header("Trading Books & Positions")

# Create tabs for different books
tab1, tab2, tab3, tab4 = st.tabs(["📊 Open Positions", "📋 Order Book", "💼 Trade Book", "🎯 Strategy Trades"])

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
                st.write(f"Product: {pos['Product']}")
            
            with col3:
                if st.button(f"❌ Close", key=f"close_pos_{idx}_{pos['Symbol']}"):
                    if exit_broker_position(
                        pos['Exchange'],
                        pos['Symbol'],
                        pos['Product'],
                        pos['Net Qty'],
                        api
                    ):
                        time.sleep(1)
                        st.rerun()
        
        # Summary
        total_pnl = sum(pos['PNL'] for pos in positions_data)
        pnl_color = "🟢" if total_pnl >= 0 else "🔴"
        st.markdown(f"**Total P&L: {pnl_color} ₹{total_pnl:,.2f}**")
    else:
        st.info("No open positions found")

with tab2:
    st.subheader("Order Book")
    orders_data = get_order_book(api)
    if orders_data:
        df_orders = pd.DataFrame(orders_data)
        st.dataframe(df_orders, use_container_width=True)
    else:
        st.info("No orders found")

with tab3:
    st.subheader("Trade Book")
    trades_data = get_trade_book(api)
    if trades_data:
        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No trades found")

with tab4:
    st.subheader("Strategy Active Trades")
    active_trades = {k: v for k, v in st.session_state.open_tracked_trades.items() if v['status'] == 'OPEN'}
    
    if active_trades:
        trades_data = []
        for symbol, trade in active_trades.items():
            trades_data.append({
                'Symbol': symbol,
                'Direction': '📈 BUY' if trade['buy_or_sell'] == 'B' else '📉 SELL',
                'Quantity': trade['quantity'],
                'Entry Price': f"₹{trade['entry_price']:.2f}",
                'Entry Time': trade['entry_time'].strftime('%H:%M:%S'),
                'Type': trade.get('order_type', 'NEW'),
                'Order No': trade.get('order_no', 'N/A')
            })
        df_strategy_trades = pd.DataFrame(trades_data)
        st.dataframe(df_strategy_trades, use_container_width=True)
    else:
        st.info("No active strategy trades")

st.markdown("---")

# Main Strategy Section
st.header("Strategy Status & Signals")
status_placeholder = st.empty()
signals_placeholder = st.empty()

# Load symbols
nifty500_symbols = load_symbols_from_csv()
if not nifty500_symbols:
    st.stop()

# Exit all positions at 3:15 PM
def exit_all_positions_at_time(api):
    """Exit all positions at 3:15 PM"""
    for tradingsymbol, trade in st.session_state.open_tracked_trades.items():
        if trade['status'] == 'OPEN':
            st.info(f"Time exit: Closing {tradingsymbol}")
            exit_position_sdvwap(tradingsymbol, api, 'TIME_EXIT')
# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💰 Capital per Trade",
        value=f"₹{st.session_state.predetermined_capital:,}"
    )

with col2:
    st.metric(
        label="💵 Available Margin", 
        value=f"₹{cash_margin:,.0f}" if cash_margin else "N/A"
    )

with col3:
    market_start_str = MARKET_START_TIME.strftime('%H:%M')
    st.metric(
        label="🕘 Market Start",
        value=market_start_str
    )
