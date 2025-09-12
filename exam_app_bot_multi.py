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
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "68a88fc4f2c954042d11f9380a36f948872843444fb63dabf8b8863e2bbcdb29")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 50  # Enough candles for SDVWAP calculation

# Initialize session state variables
if 'predetermined_capital' not in st.session_state:
    st.session_state.predetermined_capital = 100000
if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {}
if 'daily_profit_exits' not in st.session_state:
    st.session_state.daily_profit_exits = {}  # Track profit exits per stock per day
if 'trading_start_time' not in st.session_state:
    st.session_state.trading_start_time = datetime.time(9, 30)
if 'trading_end_time' not in st.session_state:
    st.session_state.trading_end_time = datetime.time(15, 0)
if 'exit_time' not in st.session_state:
    st.session_state.exit_time = datetime.time(15, 15)

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

def calculate_sdvwap(candle_data):
    """Calculate SDVWAP1+ and SDVWAP1- from candle data."""
    try:
        if len(candle_data) < 20:
            return None, None
        
        prices = []
        volumes = []
        
        for candle in candle_data:
            high = float(candle.get('inth', 0))
            low = float(candle.get('intl', 0))
            close = float(candle.get('intc', 0))
            volume = float(candle.get('intv', 0))
            
            if high > 0 and low > 0 and close > 0 and volume > 0:
                typical_price = (high + low + close) / 3
                prices.append(typical_price)
                volumes.append(volume)
        
        if len(prices) < 20:
            return None, None
        
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        weighted_mean = vwap
        weighted_variance = np.sum(volumes * (prices - weighted_mean) ** 2) / np.sum(volumes)
        weighted_std = np.sqrt(weighted_variance)
        
        sdvwap_plus = vwap + weighted_std
        sdvwap_minus = vwap - weighted_std
        
        return sdvwap_plus, sdvwap_minus
        
    except Exception as e:
        logging.error(f"Error calculating SDVWAP: {e}")
        return None, None

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
    """Screen a stock based on SDVWAP strategy."""
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']

    if not is_trading_time():
        return tradingsymbol, 'NEUTRAL', 'Outside trading hours', None, None

    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    profit_exit_restriction = None
    if tradingsymbol in st.session_state.daily_profit_exits:
        if today_str in st.session_state.daily_profit_exits[tradingsymbol]:
            profit_exit_restriction = st.session_state.daily_profit_exits[tradingsymbol][today_str]
    
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=REQUIRED_CANDLES + 10)
    
    try:
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=int(start_time.timestamp()),
            endtime=int(end_time.timestamp()),
            interval=CANDLE_INTERVAL
        )

        if not candle_data or len(candle_data) < 3:
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None

        sdvwap_plus, sdvwap_minus = calculate_sdvwap(candle_data)
        
        if sdvwap_plus is None or sdvwap_minus is None:
            return tradingsymbol, 'NEUTRAL', 'Could not calculate SDVWAP', None, None

        current_candle = candle_data[0]
        candle_minus_1 = candle_data[1] if len(candle_data) > 1 else None
        candle_minus_2 = candle_data[2] if len(candle_data) > 2 else None

        if not candle_minus_1 or not candle_minus_2:
            return tradingsymbol, 'NEUTRAL', 'Need at least 3 candles', None, None

        c1_open = float(candle_minus_1.get('into', 0))
        c1_close = float(candle_minus_1.get('intc', 0))
        c2_open = float(candle_minus_2.get('into', 0))
        c2_close = float(candle_minus_2.get('intc', 0))
        
        current_price = float(current_candle.get('intc', 0))
        
        # Calculate order value and check margin
        initial_quantity = calculate_initial_quantity(st.session_state.predetermined_capital, current_price)
        order_value = initial_quantity * current_price
        margin_ok, margin_msg = check_margin_requirement(order_value, cash_margin)

        buy_condition = (
            c2_close > c2_open and c2_close > sdvwap_plus and
            c1_close > c1_open and c1_close > sdvwap_plus
        )
        
        sell_condition = (
            c2_close < c2_open and c2_close < sdvwap_minus and
            c1_close < c1_open and c1_close < sdvwap_minus
        )

        # Apply profit exit restrictions
        if profit_exit_restriction == 'BUY':
            buy_condition = False
        elif profit_exit_restriction == 'SELL':
            sell_condition = False

        if buy_condition and margin_ok:
            return tradingsymbol, 'BUY', f'SDVWAP Buy Signal: SDVWAP+={sdvwap_plus:.2f}, {margin_msg}', current_price, initial_quantity
        elif sell_condition and margin_ok:
            return tradingsymbol, 'SELL', f'SDVWAP Sell Signal: SDVWAP-={sdvwap_minus:.2f}, {margin_msg}', current_price, initial_quantity
        elif (buy_condition or sell_condition) and not margin_ok:
            return tradingsymbol, 'NEUTRAL', f'Signal detected but {margin_msg}', current_price, initial_quantity
        else:
            return tradingsymbol, 'NEUTRAL', 'No SDVWAP signal', None, None

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}")
        return tradingsymbol, 'NEUTRAL', f'Error: {e}', None, None

def place_order_sdvwap(buy_or_sell, tradingsymbol, quantity, current_price, api, order_type='NEW'):
    """Place order with SDVWAP strategy logic."""
    if quantity == 0:
        st.warning(f"Cannot place order for {tradingsymbol}: Quantity is zero.")
        return {'stat': 'Not_Ok', 'emsg': 'Quantity is zero'}

    st.info(f"Placing {buy_or_sell} order for {tradingsymbol}: Qty={quantity}, Price={current_price}, Type={order_type}")

    try:
        order_response = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type='I',  # Intraday
            exchange=EXCHANGE,
            tradingsymbol=tradingsymbol,
            quantity=int(quantity),
            discloseqty=0,
            price_type='LMT',
            price=current_price,
            trigger_price=None,
            retention='DAY',
            remarks=f'SDVWAP_Strategy_{order_type}'
        )
        
        if order_response and order_response.get('stat') == 'Ok':
            st.success(f"Order placed successfully for {tradingsymbol}: {order_response.get('norenordno')}")
            
            # Update tracking
            st.session_state.open_tracked_trades[tradingsymbol] = {
                'order_no': order_response.get('norenordno'),
                'entry_price': current_price,
                'quantity': quantity,
                'buy_or_sell': buy_or_sell,
                'status': 'OPEN',
                'entry_time': datetime.datetime.now(),
                'order_type': order_type
            }
            
        else:
            st.error(f"Failed to place order for {tradingsymbol}: {order_response.get('emsg', 'Unknown error')}")
        return order_response
    except Exception as e:
        st.error(f"Error placing order for {tradingsymbol}: {e}")
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def exit_position_sdvwap(tradingsymbol, api, reason='Manual'):
    """Exit existing position"""
    if tradingsymbol not in st.session_state.open_tracked_trades:
        return {'stat': 'Not_Ok', 'emsg': 'No tracked trade found'}
    
    trade = st.session_state.open_tracked_trades[tradingsymbol]
    if trade['status'] != 'OPEN':
        return {'stat': 'Not_Ok', 'emsg': 'Trade not open'}
    
    exit_action = 'S' if trade['buy_or_sell'] == 'B' else 'B'
    
    try:
        exit_response = api.place_order(
            buy_or_sell=exit_action,
            product_type='I',
            exchange=EXCHANGE,
            tradingsymbol=tradingsymbol,
            quantity=trade['quantity'],
            discloseqty=0,
            price_type='MKT',
            price=0,
            trigger_price=None,
            retention='DAY',
            remarks=f'SDVWAP_EXIT_{reason}'
        )
        
        if exit_response and exit_response.get('stat') == 'Ok':
            st.success(f"Exit order placed for {tradingsymbol}: {exit_response.get('norenordno')}")
            trade['status'] = 'CLOSED'
            trade['exit_time'] = datetime.datetime.now()
            trade['exit_reason'] = reason
        else:
            st.error(f"Failed to exit {tradingsymbol}: {exit_response.get('emsg', 'Unknown error')}")
        
        return exit_response
    except Exception as e:
        st.error(f"Error exiting {tradingsymbol}: {e}")
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def monitor_positions_sdvwap(api):
    """Monitor positions for profit exits"""
    for tradingsymbol, trade in st.session_state.open_tracked_trades.items():
        if trade['status'] != 'OPEN':
            continue
        
        try:
            token = None
            for symbol_data in nifty500_symbols:
                if symbol_data['tsym'] == tradingsymbol:
                    token = symbol_data['token']
                    break
            
            if not token:
                continue
            
            quotes = api.get_quotes([f"{EXCHANGE}|{token}"])
            if not quotes or quotes.get('stat') != 'Ok':
                continue
            
            current_price = float(quotes['values'][0]['lp'])
            entry_price = trade['entry_price']
            
            if trade['buy_or_sell'] == 'B':
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            st.markdown(f"**{tradingsymbol}**: Entry={entry_price:.2f}, Current={current_price:.2f}, P&L={profit_pct:.2f}%")
            
            if profit_pct >= 2.0:
                st.success(f"2% profit reached for {tradingsymbol}! Exiting position.")
                exit_response = exit_position_sdvwap(tradingsymbol, api, 'PROFIT_2PCT')
                
                if exit_response.get('stat') == 'Ok':
                    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
                    if tradingsymbol not in st.session_state.daily_profit_exits:
                        st.session_state.daily_profit_exits[tradingsymbol] = {}
                    st.session_state.daily_profit_exits[tradingsymbol][today_str] = trade['buy_or_sell']
        
        except Exception as e:
            logging.error(f"Error monitoring {tradingsymbol}: {e}")

def process_reverse_signals(signals_data, api, cash_margin):
    """Process signals that might trigger position reversals"""
    for signal_data in signals_data:
        tradingsymbol = signal_data['Symbol']
        signal = signal_data['Signal']
        current_price = signal_data['Price']
        quantity = signal_data['Quantity']
        
        if current_price <= 0 or quantity <= 0:
            continue
        
        # Check if there's an existing position
        if tradingsymbol in st.session_state.open_tracked_trades:
            existing_trade = st.session_state.open_tracked_trades[tradingsymbol]
            if existing_trade['status'] == 'OPEN':
                existing_direction = existing_trade['buy_or_sell']
                
                # Check for reverse condition
                if ((existing_direction == 'B' and signal == 'SELL') or 
                    (existing_direction == 'S' and signal == 'BUY')):
                    
                    st.warning(f"Reverse signal detected for {tradingsymbol}!")
                    
                    # Calculate 3x quantity and check margin
                    reverse_quantity = existing_trade['quantity'] * 3
                    reverse_order_value = reverse_quantity * current_price
                    margin_ok, margin_msg = check_margin_requirement(reverse_order_value, cash_margin)
                    
                    if not margin_ok:
                        st.error(f"Cannot reverse {tradingsymbol}: {margin_msg}")
                        continue
                    
                    # Exit existing position
                    exit_response = exit_position_sdvwap(tradingsymbol, api, 'REVERSE')
                    
                    if exit_response.get('stat') == 'Ok':
                        time.sleep(0.5)
                        new_direction = 'B' if signal == 'BUY' else 'S'
                        place_order_sdvwap(
                            buy_or_sell=new_direction,
                            tradingsymbol=tradingsymbol,
                            quantity=reverse_quantity,
                            current_price=current_price,
                            api=api,
                            order_type='REVERSE'
                        )
                    continue
        
        # No existing position, place new order
        direction = 'B' if signal == 'BUY' else 'S'
        place_order_sdvwap(
            buy_or_sell=direction,
            tradingsymbol=tradingsymbol,
            quantity=quantity,
            current_price=current_price,
            api=api,
            order_type='NEW'
        )

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="SDVWAP Strategy")
st.title("📈 SDVWAP Trading Strategy")
st.markdown("Automated SDVWAP-based trading with margin management")

# Sidebar
st.sidebar.header("Strategy Settings")
screen_interval = st.sidebar.slider("Screening Interval (seconds)", min_value=5, max_value=60, value=30)
run_strategy = st.sidebar.checkbox("Run Strategy Continuously", value=False)

st.sidebar.subheader("Trading Parameters")
st.session_state.predetermined_capital = st.sidebar.number_input(
    "Predetermined Capital per Trade (INR)",
    min_value=1000,
    max_value=10000000,
    value=int(st.session_state.predetermined_capital),
    step=1000
)

# Display current time and trading status
current_time = datetime.datetime.now().time()
st.sidebar.markdown(f"**Current Time:** {current_time.strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**Trading Hours:** {st.session_state.trading_start_time} - {st.session_state.trading_end_time}")

if is_trading_time():
    st.sidebar.success("✅ Within Trading Hours")
elif is_exit_time():
    st.sidebar.error("🔴 Exit Time")
else:
    st.sidebar.warning("⏸️ Outside Trading Hours")

# Account Information Section
st.header("Account Information")
col1, col2 = st.columns([3, 1])

with col1:
    account_info = get_account_info(api)
    if account_info['status'] == 'success' and account_info['cash_margin'] is not None:
        st.success(f"**Available Cash Margin:** ₹{account_info['cash_margin']:,.2f}")
        cash_margin = account_info['cash_margin']
    else:
        st.error(f"Could not fetch account info: {account_info.get('error', 'Unknown error')}")
        cash_margin = 0

with col2:
    if st.button("🔄 Refresh Account", key="refresh_account"):
        st.rerun()

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

# Main strategy loop
if run_strategy:
    while run_strategy:
        with status_placeholder.container():
            current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
            st.info(f"🔄 Running SDVWAP strategy at {current_time_str}")
        
        # Check for exit time
        if is_exit_time():
            exit_all_positions_at_time(api)
            st.warning("⏰ Exit time reached. All positions closed.")
            break
        
        # Skip screening outside trading hours
        if not is_trading_time():
            with status_placeholder.container():
                st.warning("⏸️ Outside trading hours. Waiting...")
            time.sleep(screen_interval)
            continue
        
        # Get fresh account info for margin checks
        account_info = get_account_info(api)
        current_cash_margin = account_info.get('cash_margin', 0)
        
        buy_signals = []
        sell_signals = []
        
        # Screen stocks for signals
        progress_bar = st.progress(0, text="🔍 Screening stocks...")
        
        # Limit screening to manageable number
        screening_symbols = nifty500_symbols[:100]  # First 100 symbols
        
        for i, stock in enumerate(screening_symbols):
            symbol, signal, reason, price, quantity = screen_stock_sdvwap(stock, api, current_cash_margin)
            
            if signal == 'BUY' and price and quantity:
                buy_signals.append({
                    'Symbol': symbol,
                    'Signal': 'BUY',
                    'Reason': reason,
                    'Price': price,
                    'Quantity': quantity,
                    'Order Value': f"₹{price * quantity:,.2f}"
                })
            elif signal == 'SELL' and price and quantity:
                sell_signals.append({
                    'Symbol': symbol,
                    'Signal': 'SELL',
                    'Reason': reason,
                    'Price': price,
                    'Quantity': quantity,
                    'Order Value': f"₹{price * quantity:,.2f}"
                })
            
            progress_bar.progress((i + 1) / len(screening_symbols), text=f"Screening {stock['tsym']}...")
        
        progress_bar.empty()
        
        # Display signals
        with signals_placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Buy Signals")
                if buy_signals:
                    # Convert for display
                    buy_display = []
                    for signal in buy_signals:
                        buy_display.append({
                            'Symbol': signal['Symbol'],
                            'Price': f"₹{signal['Price']:.2f}",
                            'Qty': signal['Quantity'],
                            'Order Value': signal['Order Value'],
                            'Reason': signal['Reason'][:50] + '...' if len(signal['Reason']) > 50 else signal['Reason']
                        })
                    st.dataframe(pd.DataFrame(buy_display), use_container_width=True)
                else:
                    st.info("No buy signals")
            
            with col2:
                st.subheader("🔴 Sell Signals")
                if sell_signals:
                    # Convert for display
                    sell_display = []
                    for signal in sell_signals:
                        sell_display.append({
                            'Symbol': signal['Symbol'],
                            'Price': f"₹{signal['Price']:.2f}",
                            'Qty': signal['Quantity'],
                            'Order Value': signal['Order Value'],
                            'Reason': signal['Reason'][:50] + '...' if len(signal['Reason']) > 50 else signal['Reason']
                        })
                    st.dataframe(pd.DataFrame(sell_display), use_container_width=True)
                else:
                    st.info("No sell signals")
        
        # Process signals (including reversals)
        all_signals = buy_signals + sell_signals
        if all_signals:
            st.info(f"📊 Processing {len(all_signals)} signals...")
            process_reverse_signals(all_signals, api, current_cash_margin)
        
        # Monitor existing positions for profit exits
        if st.session_state.open_tracked_trades:
            st.subheader("📈 Position Monitoring")
            monitor_positions_sdvwap(api)
        
        # Wait for next cycle
        time.sleep(screen_interval)

else:
    with status_placeholder.container():
        st.info("⏸️ Strategy is paused. Enable 'Run Strategy Continuously' in the sidebar to start.")
        
        # Show manual screening option when paused
        if st.button("🔍 Run Single Screen", key="manual_screen"):
            account_info = get_account_info(api)
            current_cash_margin = account_info.get('cash_margin', 0)
            
            buy_signals = []
            sell_signals = []
            
            progress_bar = st.progress(0, text="Running manual screening...")
            screening_symbols = nifty500_symbols[:50]  # Smaller set for manual screening
            
            for i, stock in enumerate(screening_symbols):
                symbol, signal, reason, price, quantity = screen_stock_sdvwap(stock, api, current_cash_margin)
                
                if signal == 'BUY' and price and quantity:
                    buy_signals.append({
                        'Symbol': symbol,
                        'Signal': 'BUY',
                        'Reason': reason,
                        'Price': price,
                        'Quantity': quantity,
                        'Order Value': f"₹{price * quantity:,.2f}"
                    })
                elif signal == 'SELL' and price and quantity:
                    sell_signals.append({
                        'Symbol': symbol,
                        'Signal': 'SELL',
                        'Reason': reason,
                        'Price': price,
                        'Quantity': quantity,
                        'Order Value': f"₹{price * quantity:,.2f}"
                    })
                
                progress_bar.progress((i + 1) / len(screening_symbols), text=f"Screening {stock['tsym']}...")
            
            progress_bar.empty()
            
            # Display manual screening results
            with signals_placeholder.container():
                st.subheader("📊 Manual Screening Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🟢 **Buy Signals**")
                    if buy_signals:
                        for signal in buy_signals:
                            st.write(f"• {signal['Symbol']}: ₹{signal['Price']:.2f} (Qty: {signal['Quantity']})")
                    else:
                        st.info("No buy signals found")
                
                with col2:
                    st.write("🔴 **Sell Signals**")
                    if sell_signals:
                        for signal in sell_signals:
                            st.write(f"• {signal['Symbol']}: ₹{signal['Price']:.2f} (Qty: {signal['Quantity']})")
                    else:
                        st.info("No sell signals found")

# Footer with key metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

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
    active_count = len([t for t in st.session_state.open_tracked_trades.values() if t['status'] == 'OPEN'])
    st.metric(
        label="📊 Active Trades",
        value=str(active_count)
    )

with col4:
    profit_exits_today = 0
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    for stock_exits in st.session_state.daily_profit_exits.values():
        if today_str in stock_exits:
            profit_exits_today += 1
    
    st.metric(
        label="✅ Profit Exits Today",
        value=str(profit_exits_today)
    )
