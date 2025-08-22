import streamlit as st
import os
import sys
import logging
import datetime
import time
import pandas as pd

# Add the parent directory to the sys.path to import api_helper
# This assumes api_helper.py is in the parent directory of this script.
# Make sure your api_helper.py is correctly placed relative to this Streamlit app file.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
# Set up logging for detailed output. For Streamlit, using st.write or st.info/warning
# is often better than traditional logging for user visibility in the app itself,
# but logging.DEBUG can be useful for console debugging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
# IMPORTANT: Replace with your actual Flattrade user ID and session token.
# For production Streamlit apps, consider using Streamlit Secrets for credentials.
# Learn more: https://docs.streamlit.io/library/advanced-features/secrets
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "6ef8514a342258801329936e109c54720ca84a08596c9adda1d635f1195d4708")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 21 # Latest candle + previous 20 for calculations

# Initialize session state variables for modifiable parameters
if 'volume_multiplier' not in st.session_state:
    st.session_state.volume_multiplier = 6
if 'traded_value_threshold' not in st.session_state:
    st.session_state.traded_value_threshold = 60000000
if 'high_low_diff_multiplier' not in st.session_state:
    st.session_state.high_low_diff_multiplier = 3
if 'capital' not in st.session_state:
    st.session_state.capital = 10000
if 'quantity_factor' not in st.session_state:
    st.session_state.quantity_factor = 10
if 'stoploss_factor' not in st.session_state:
    st.session_state.stoploss_factor = 1000
if 'target_multiplier' not in st.session_state:
    st.session_state.target_multiplier = 4
if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {} # To track trades placed by the app

# --- Initialize Flattrade API (cached for Streamlit efficiency) ---
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
            logging.critical(f"Failed to set API session: {error_msg}")
            return None
    except Exception as e:
        st.error(f"An exception occurred during API session setup: {e}")
        logging.critical(f"An exception occurred during API session setup: {e}")
        return None

# Get API instance
api = get_api_instance(USER_ID, USER_SESSION)

# Ensure API is initialized before proceeding
if api is None:
    st.stop() # Stop the app if API connection fails

@st.cache_data
def load_symbols_from_csv(file_path="NSE_Equity.csv"):
    """
    Loads stock symbols and tokens from the provided CSV file.
    Filters for 'EQ' (Equity) instruments only.
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure column names are correct and case-sensitive as in the CSV header
        if all(col in df.columns for col in ['Exchange', 'Token', 'Tradingsymbol', 'Instrument']):
            # Filter for equity instruments
            equity_symbols = df[df['Instrument'] == 'EQ'][['Exchange', 'Token', 'Tradingsymbol']].copy()
            
            symbols_list = []
            for index, row in equity_symbols.iterrows():
                symbols_list.append({
                    'exchange': row['Exchange'],
                    'token': str(row['Token']), # Ensure token is string
                    'tsym': row['Tradingsymbol']
                })
            st.success(f"Loaded {len(symbols_list)} equity symbols from {file_path}.")
            return symbols_list
        else:
            st.error(f"CSV file '{file_path}' must contain 'Exchange', 'Token', 'Tradingsymbol', and 'Instrument' columns.")
            return []
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory as the Streamlit app.")
        return []
    except Exception as e:
        st.error(f"Error loading symbols from CSV: {e}")
        return []

def get_nifty500_symbols():
    """
    Uses the load_symbols_from_csv function to get the actual symbols.
    """
    return load_symbols_from_csv()


def screen_stock(stock_info, api):
    """
    Screens a single stock based on the defined criteria using 1-minute candle data.

    Args:
        stock_info (dict): A dictionary containing 'exchange', 'token', 'tsym'.
        api (NorenApiPy): The initialized Flattrade API object.

    Returns:
        tuple: (stock_symbol, 'BUY'/'SELL'/'NEUTRAL', reason, current_price)
               current_price is returned only if a signal is generated, else None.
    """
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']

    # st.info(f"Screening {tradingsymbol}...") # Too verbose for Streamlit display

    # Calculate start time for fetching 21 candles (20 for average + 1 current)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=REQUIRED_CANDLES + 5) 
    
    try:
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=int(start_time.timestamp()),
            endtime=int(end_time.timestamp()),
            interval=CANDLE_INTERVAL
        )

        if not candle_data or len(candle_data) < REQUIRED_CANDLES:
            logging.warning(f"Not enough 1-min candle data for {tradingsymbol}. Needed: {REQUIRED_CANDLES}, Got: {len(candle_data) if candle_data else 0}")
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None

        current_candle = candle_data[0]
        previous_20_candles = candle_data[1:REQUIRED_CANDLES] 

        # --- 1. Volume Check ---
        current_volume = float(current_candle.get('intv', 0))
        if current_volume == 0:
            return tradingsymbol, 'NEUTRAL', 'Current candle volume is zero', None
            
        previous_volumes = [float(c.get('intv', 0)) for c in previous_20_candles]
        non_zero_previous_volumes = [v for v in previous_volumes if v > 0]

        if not non_zero_previous_volumes:
            return tradingsymbol, 'NEUTRAL', 'No valid volume data in previous 20 candles', None

        average_volume_last_20 = sum(non_zero_previous_volumes) / len(non_zero_previous_volumes)

        if not (current_volume > st.session_state.volume_multiplier * average_volume_last_20):
            return tradingsymbol, 'NEUTRAL', 'Volume condition not met', None

        # --- 2. Traded Value Check ---
        current_close_price = float(current_candle.get('intc', 0))
        if current_close_price == 0:
            return tradingsymbol, 'NEUTRAL', 'Current candle close price is zero', None
            
        current_traded_value = current_volume * current_close_price
        
        if not (current_traded_value > st.session_state.traded_value_threshold):
            return tradingsymbol, 'NEUTRAL', 'Traded value condition not met', None

        # --- 3. High-Low Difference Check ---
        current_high = float(current_candle.get('inth', 0))
        current_low = float(current_candle.get('intl', 0))
        current_high_low_diff = current_high - current_low

        if current_high_low_diff <= 0: 
            return tradingsymbol, 'NEUTRAL', 'Current high-low difference invalid', None

        previous_high_low_diffs = []
        for c in previous_20_candles:
            high = float(c.get('inth', 0))
            low = float(c.get('intl', 0))
            if high > 0 and low > 0: 
                diff = high - low
                if diff > 0: 
                    previous_high_low_diffs.append(diff)

        if not previous_high_low_diffs:
            return tradingsymbol, 'NEUTRAL', 'No valid high-low diff data in previous 20 candles', None

        average_high_low_diff_last_20 = sum(previous_high_low_diffs) / len(previous_high_low_diffs)

        if not (current_high_low_diff > st.session_state.high_low_diff_multiplier * average_high_low_diff_last_20):
            return tradingsymbol, 'NEUTRAL', 'High-low diff condition not met', None

        # --- 4. Candle Color Check ---
        current_open_price = float(current_candle.get('into', 0))
        
        if current_close_price > current_open_price:
            return tradingsymbol, 'BUY', 'All conditions met: Green candle', current_close_price
        elif current_close_price < current_open_price:
            return tradingsymbol, 'SELL', 'All conditions met: Red candle', current_close_price
        else:
            return tradingsymbol, 'NEUTRAL', 'Current candle is Doji (Open == Close)', None

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}", exc_info=True)
        return tradingsymbol, 'NEUTRAL', f'Error during screening: {e}', None

def place_intraday_order(
    buy_or_sell, tradingsymbol, quantity, current_price, api
):
    """
    Places a regular Intraday (MIS) order. Stoploss and target will be managed by the app.
    """
    if quantity == 0:
        st.warning(f"Cannot place order for {tradingsymbol}: Quantity is zero.")
        return {'stat': 'Not_Ok', 'emsg': 'Quantity is zero'}

    st.info(f"Placing {buy_or_sell} Intraday order for {tradingsymbol}: Qty={quantity}, Price={current_price}")

    try:
        order_response = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type='I',  # 'I' for Intraday (MIS)
            exchange=EXCHANGE,
            tradingsymbol=tradingsymbol,
            quantity=int(quantity),
            discloseqty=0,
            price_type='LMT',  # Entry order as Limit order
            price=current_price,
            trigger_price=None,
            retention='DAY',
            remarks='Automated_Screener_Trade_Manual_SL_TP'
        )
        if order_response and order_response.get('stat') == 'Ok':
            st.success(f"Order placed successfully for {tradingsymbol}: {order_response}")
            logging.info(f"Order placed successfully for {tradingsymbol}: {order_response}")

            # Calculate SL/TP prices based on current price and factors
            stoploss_amount_total = st.session_state.capital / st.session_state.stoploss_factor
            stoploss_points_per_share = round(stoploss_amount_total / quantity, 2)
            target_points_per_share = round(st.session_state.target_multiplier * stoploss_points_per_share, 2)

            if buy_or_sell == 'B':
                sl_price = round(current_price - stoploss_points_per_share, 2)
                tp_price = round(current_price + target_points_per_share, 2)
            else: # SELL
                sl_price = round(current_price + stoploss_points_per_share, 2)
                tp_price = round(current_price - target_points_per_share, 2)

            # Store the trade for monitoring
            st.session_state.open_tracked_trades[tradingsymbol] = {
                'order_no': order_response.get('norenordno'),
                'entry_price': current_price,
                'quantity': quantity, # Use the actual quantity used for order
                'sl_price': sl_price,
                'target_price': tp_price,
                'buy_or_sell': buy_or_sell,
                'status': 'OPEN'
            }
            st.info(f"Tracking trade for {tradingsymbol}: Entry={current_price}, SL={sl_price}, Target={tp_price}")

        else:
            st.error(f"Failed to place order for {tradingsymbol}: {order_response.get('emsg', 'Unknown error')}")
            logging.error(f"Failed to place order for {tradingsymbol}: {order_response.get('emsg', 'Unknown error')}")
        return order_response
    except Exception as e:
        st.error(f"An error occurred while placing order for {tradingsymbol}: {e}")
        logging.error(f"An error occurred while placing order for {tradingsymbol}: {e}", exc_info=True)
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def exit_position(exchange, tradingsymbol, product_type, netqty, api):
    """
    Exits an existing position.
    """
    st.info(f"Attempting to exit position for {tradingsymbol} (Qty: {netqty}, Product: {product_type})...")
    try:
        response = api.exit_order(
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            product_type=product_type,
            quantity=abs(int(netqty))
        )
        if response and response.get('stat') == 'Ok':
            st.success(f"Position for {tradingsymbol} exited successfully: {response}")
            logging.info(f"Position for {tradingsymbol} exited successfully: {response}")
            # Mark the tracked trade as closed
            if tradingsymbol in st.session_state.open_tracked_trades:
                st.session_state.open_tracked_trades[tradingsymbol]['status'] = 'CLOSED'
        else:
            st.error(f"Failed to exit position for {tradingsymbol}: {response.get('emsg', 'Unknown error')}")
            logging.error(f"Failed to exit position for {tradingsymbol}: {response.get('emsg', 'Unknown error')}")
        return response
    except Exception as e:
        st.error(f"An error occurred while exiting position for {tradingsymbol}: {e}")
        logging.error(f"An error occurred while exiting position for {tradingsymbol}: {e}", exc_info=True)
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def monitor_open_trades(api):
    """
    Monitors actively tracked trades for stoploss/target conditions.
    Fetches live quotes and triggers closing orders if conditions are met.
    """
    trades_to_close = []
    
    # Get current quotes for all open tracked trades
    tokens_to_fetch = []
    tsym_to_token_map = {}
    for tsym, trade_info in st.session_state.open_tracked_trades.items():
        if trade_info['status'] == 'OPEN':
            # Find the token from nifty500_symbols based on tsym
            for symbol_data in nifty500_symbols:
                if symbol_data['tsym'] == tsym:
                    tokens_to_fetch.append(f"{EXCHANGE}|{symbol_data['token']}")
                    tsym_to_token_map[f"{EXCHANGE}|{symbol_data['token']}"] = tsym
                    break
    
    if not tokens_to_fetch:
        return # No open trades to monitor

    try:
        quotes = api.get_quotes(tokens_to_fetch)
        if quotes and quotes.get('stat') == 'Ok' and quotes.get('values'):
            live_quotes = {item['tsym']: float(item['lp']) for item in quotes['values']}
            
            for tsym, trade_info in st.session_state.open_tracked_trades.items():
                if trade_info['status'] == 'OPEN' and tsym in live_quotes:
                    current_ltp = live_quotes[tsym]
                    sl_price = trade_info['sl_price']
                    target_price = trade_info['target_price']
                    buy_or_sell = trade_info['buy_or_sell']
                    quantity = trade_info['quantity']

                    st.markdown(f"**Monitoring {tsym}:** LTP={current_ltp:.2f}, SL={sl_price:.2f}, Target={target_price:.2f}")

                    if buy_or_sell == 'B': # Long position
                        if current_ltp <= sl_price:
                            st.warning(f"Stoploss HIT for BUY {tsym}! LTP {current_ltp} <= SL {sl_price}")
                            trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'SELL'})
                            trade_info['status'] = 'CLOSING_SL' # Mark for closing
                        elif current_ltp >= target_price:
                            st.success(f"Target HIT for BUY {tsym}! LTP {current_ltp} >= Target {target_price}")
                            trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'SELL'})
                            trade_info['status'] = 'CLOSING_TP' # Mark for closing
                    elif buy_or_sell == 'S': # Short position
                        if current_ltp >= sl_price:
                            st.warning(f"Stoploss HIT for SELL {tsym}! LTP {current_ltp} >= SL {sl_price}")
                            trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'BUY'})
                            trade_info['status'] = 'CLOSING_SL' # Mark for closing
                        elif current_ltp <= target_price:
                            st.success(f"Target HIT for SELL {tsym}! LTP {current_ltp} <= Target {target_price}")
                            trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'BUY'})
                            trade_info['status'] = 'CLOSING_TP' # Mark for closing
        else:
            logging.warning(f"Failed to get live quotes for monitoring: {quotes}")

    except Exception as e:
        logging.error(f"Error monitoring trades: {e}", exc_info=True)

    # Execute closing orders
    for trade in trades_to_close:
        tsym = trade['tsym']
        action = trade['action']
        quantity = trade['quantity']

        st.info(f"Placing closing {action} order for {tsym} (Qty: {quantity})...")
        # For closing, we generally place a market order for quick execution
        close_response = api.place_order(
            buy_or_sell=action,
            product_type='I', # Assuming it's an Intraday position ('I')
            exchange=EXCHANGE,
            tradingsymbol=tsym,
            quantity=int(quantity),
            discloseqty=0,
            price_type='MKT', # Market order to ensure quick exit
            price=0, # Not relevant for MKT order
            trigger_price=None,
            retention='DAY',
            remarks=f'Automated_Exit_{st.session_state.open_tracked_trades[tsym]["status"]}'
        )
        if close_response and close_response.get('stat') == 'Ok':
            st.success(f"Closing order for {tsym} successful: {close_response}")
            st.session_state.open_tracked_trades[tsym]['status'] = 'CLOSED' # Officially mark as closed
        else:
            st.error(f"Failed to place closing order for {tsym}: {close_response.get('emsg', 'Unknown error')}")
            # If failed, keep status as CLOSING_SL/TP to retry in next cycle or manually handle
            logging.error(f"Failed to place closing order for {tsym}: {close_response}")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Flattrade Algo Screener")

st.title("📈 Flattrade Algo Screener & Trader")
st.markdown("Automated stock screening and position management for Nifty500.")

# Sidebar for controls
st.sidebar.header("Settings")
screen_interval = st.sidebar.slider("Screening Interval (seconds)", min_value=5, max_value=60, value=10)
run_screener = st.sidebar.checkbox("Run Screener Continuously", value=False)

st.sidebar.subheader("Screening Conditions")
st.session_state.volume_multiplier = st.sidebar.number_input(
    "Volume Multiplier (Current vs Avg 20)",
    min_value=1, value=int(st.session_state.volume_multiplier), step=1
)
st.session_state.traded_value_threshold = st.sidebar.number_input(
    "Min Traded Value (INR)",
    min_value=100000, value=int(st.session_state.traded_value_threshold), step=100000, format="%d"
)
st.session_state.high_low_diff_multiplier = st.sidebar.number_input(
    "High-Low Difference Multiplier (Current vs Avg 20)",
    min_value=1, value=int(st.session_state.high_low_diff_multiplier), step=1
)

st.sidebar.subheader("Trading Parameters")
st.session_state.capital = st.sidebar.number_input(
    "Available Capital (INR)",
    min_value=1000,
    max_value=10000000,
    value=int(st.session_state.capital),
    step=1000
)
st.session_state.quantity_factor = st.sidebar.number_input(
    "Quantity Factor (Capital / Factor = Investment per trade)",
    min_value=1, value=int(st.session_state.quantity_factor), step=1
)
st.session_state.stoploss_factor = st.sidebar.number_input(
    "Stoploss Factor (Capital / Factor = SL amount per trade)",
    min_value=100, value=int(st.session_state.stoploss_factor), step=100
)
st.session_state.target_multiplier = st.sidebar.number_input(
    "Target Multiplier (SL amount * Multiplier = Target amount)",
    min_value=1, value=int(st.session_state.target_multiplier), step=1
)

st.sidebar.markdown(f"**Calculated Order Value (per trade):** ₹{st.session_state.capital / st.session_state.quantity_factor:,.2f}")
st.sidebar.markdown(f"**Risk per trade (approx.):** ₹{st.session_state.capital / st.session_state.stoploss_factor:,.2f}")
st.sidebar.markdown(f"**Reward per trade (approx.):** ₹{st.session_state.capital / st.session_state.stoploss_factor * st.session_state.target_multiplier:,.2f}")


# Main content area setup with columns for fixed and dynamic sections

st.header("Account Information")
account_info_col, refresh_btn_col = st.columns([0.7, 0.3])

# Account Information section (now fixed, updates on full rerun)
with account_info_col:
    account_info_placeholder = st.empty()
    with account_info_placeholder.container():
        st.info("Fetching account limits...")
        cash_margin = None # Initialize to None
        try:
            limits = api.get_limits()
            if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
                # Prioritize top-level 'cash'
                if 'cash' in limits and limits['cash'] is not None:
                    try:
                        cash_margin = float(limits['cash'])
                    except ValueError:
                        logging.error(f"Could not convert top-level 'cash' to float: {limits['cash']}")

                # If 'cash' not found at top-level or conversion failed, check 'prange'
                if cash_margin is None and 'prange' in limits and isinstance(limits['prange'], list):
                    for item in limits['prange']:
                        if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                            try:
                                cash_margin = float(item['cash'])
                                break # Found cash, exit loop
                            except ValueError:
                                logging.error(f"Could not convert 'prange' item 'cash' to float: {item['cash']}")
            
            if cash_margin is not None:
                st.success(f"**Available Cash Margin:** ₹{cash_margin:,.2f}")
            else:
                st.warning("Could not fetch account limits: 'cash' data not found in expected locations or invalid format.")
                logging.warning(f"Account limits response (final check failed): {limits}") # Log full response if still not found
        except Exception as e:
            st.error(f"Error fetching account limits: {e}")
            logging.error(f"Error fetching account limits: {e}", exc_info=True)

with refresh_btn_col:
    # Button to force a full rerun and refresh the static data
    if st.button("Refresh Data", key="refresh_all_data"):
        st.rerun()

st.markdown("---") # Separator

st.header("Broker Open Positions 💼 (For manual management or verification)")
position_placeholder = st.empty()
# This block will now also run only on a full script rerun, making it "fixed"
with position_placeholder.container():
    st.info("Fetching broker open positions...")
    try:
        positions = api.get_positions()
        if isinstance(positions, list) and len(positions) > 0:
            positions_data = []
            for pos in positions:
                if pos.get('netqty', 0) != 0: 
                    positions_data.append({
                        'Symbol': pos.get('tsym'),
                        'Exchange': pos.get('exch'),
                        'Product Type': pos.get('prd'),
                        'Net Qty': int(pos.get('netqty', 0)),
                        'Buy Avg': float(pos.get('daybuyavgprc', 0)),
                        'Sell Avg': float(pos.get('daysellavgprc', 0)),
                        'LTP': float(pos.get('lp', 0)),
                        'PNL': float(pos.get('rpnl', 0)) + float(pos.get('urmtm', 0))
                    })
            
            df_positions = pd.DataFrame(positions_data)
            if not df_positions.empty:
                st.dataframe(df_positions) # Removed hide_row_index
                st.markdown("---")
                st.markdown("### Exit Broker Positions Manually")
                for _, row in df_positions.iterrows():
                    col1_pos, col2_pos = st.columns([0.7, 0.3])
                    col1_pos.write(f"**{row['Symbol']}** (Qty: {row['Net Qty']}, Product: {row['Product Type']})")
                    if col2_pos.button(f"Exit {row['Symbol']}", key=f"exit_broker_{row['Symbol']}_{row['Product Type']}_{row['Net Qty']}"):
                        exit_response = exit_position(
                            exchange=row['Exchange'],
                            tradingsymbol=row['Symbol'],
                            product_type=row['Product Type'],
                            netqty=row['Net Qty'],
                            api=api
                        )
                        if exit_response and exit_response.get('stat') == 'Ok':
                            st.success(f"Exit order sent for {row['Symbol']}. Refreshing positions...")
                            time.sleep(1)
                            st.rerun() # Trigger a full rerun to update broker positions
                        else:
                            st.error(f"Failed to send exit order for {row['Symbol']}.")
            else:
                st.info("No open positions found in broker account.")
        else:
            st.info("No open positions found in broker account or failed to retrieve positions.")
    except Exception as e:
        st.error(f"Error fetching positions from broker: {e}")
        logging.error(f"Error fetching positions from broker: {e}", exc_info=True)

st.markdown("---") # Separator

st.header("Signals & App-Tracked Trades")

# Create placeholders for dynamic content within the continuous loop
signal_placeholder = st.empty()
tracked_trades_placeholder = st.empty()
status_placeholder = st.empty() # For continuous screening status messages

# Load symbols from CSV file
nifty500_symbols = get_nifty500_symbols()

if not nifty500_symbols:
    st.warning("No equity symbols loaded from CSV. Please ensure 'NSE_Equity.csv' is correct and accessible.")
    st.stop()

# Main continuous screening loop
while run_screener:
    with status_placeholder.container():
        st.info(f"Running screening at {datetime.datetime.now().strftime('%H:%M:%S')}...")
    
    buy_signals_data = []
    sell_signals_data = []

    # --- Run Screening for non-tracked stocks ---
    progress_bar = st.progress(0, text="Screening for new signals...")
    # Create a list of symbols not currently being tracked by the app for new entries
    nontracked_symbols = [
        s for s in nifty500_symbols 
        if s['tsym'] not in st.session_state.open_tracked_trades or 
           st.session_state.open_tracked_trades[s['tsym']]['status'] != 'OPEN'
    ]

    # Handle case where all symbols are being tracked or none loaded
    if not nontracked_symbols:
        status_placeholder.info("All relevant symbols are either being tracked or no new symbols to screen.")
        # If no non-tracked symbols, progress bar should be 100% or empty
        progress_bar.empty() 
    else:
        for i, stock in enumerate(nontracked_symbols):
            symbol_info, signal, reason, current_price = screen_stock(stock, api)
            
            if signal == 'BUY':
                buy_signals_data.append({
                    'Symbol': symbol_info,
                    'Signal': 'BUY',
                    'Reason': reason,
                    'Price': f"{current_price:,.2f}" if current_price else 'N/A'
                })
                if current_price:
                    total_investment_amount = st.session_state.capital / st.session_state.quantity_factor
                    calculated_quantity = int(total_investment_amount / current_price) if current_price > 0 else 0
                    if calculated_quantity > 0:
                        status_placeholder.info(f"Attempting to place BUY order for {symbol_info} with quantity {calculated_quantity}...")
                        place_intraday_order(
                            buy_or_sell='B',
                            tradingsymbol=stock['tsym'],
                            quantity=calculated_quantity,
                            current_price=current_price,
                            api=api
                        )
                    else:
                        status_placeholder.warning(f"Calculated quantity for {symbol_info} is zero. Not placing order. (Capital: {st.session_state.capital}, Price: {current_price}, Investment Amount: {total_investment_amount})")

            elif signal == 'SELL':
                sell_signals_data.append({
                    'Symbol': symbol_info,
                    'Signal': 'SELL',
                    'Reason': reason,
                    'Price': f"{current_price:,.2f}" if current_price else 'N/A'
                })
                if current_price:
                    total_investment_amount = st.session_state.capital / st.session_state.quantity_factor
                    calculated_quantity = int(total_investment_amount / current_price) if current_price > 0 else 0
                    if calculated_quantity > 0:
                        status_placeholder.info(f"Attempting to place SELL order for {symbol_info} with quantity {calculated_quantity}...")
                        place_intraday_order(
                            buy_or_sell='S',
                            tradingsymbol=stock['tsym'],
                            quantity=calculated_quantity,
                            current_price=current_price,
                            api=api
                        )
                    else:
                        status_placeholder.warning(f"Calculated quantity for {symbol_info} is zero. Not placing order. (Capital: {st.session_state.capital}, Price: {current_price}, Investment Amount: {total_investment_amount})")
            
            progress_bar.progress((i + 1) / len(nontracked_symbols), text=f"Screening {stock['tsym']}...")
        progress_bar.empty()


    # Display Signals in their placeholder
    with signal_placeholder.container():
        st.subheader("Current Buy Signals 🟢")
        if buy_signals_data:
            st.dataframe(pd.DataFrame(buy_signals_data))
        else:
            st.info("No BUY signals currently.")

        st.subheader("Current Sell Signals 🔴")
        if sell_signals_data:
            st.dataframe(pd.DataFrame(sell_signals_data))
        else:
            st.info("No SELL signals currently.")

    # --- Monitor and Display Tracked Trades (new section) ---
    with tracked_trades_placeholder.container():
        st.subheader("App-Tracked Trades 🚀")
        # Filter out 'CLOSED' trades for display, but keep them in session_state for history if needed
        active_tracked_trades = {tsym: trade for tsym, trade in st.session_state.open_tracked_trades.items() if trade['status'] != 'CLOSED'}
        
        if active_tracked_trades:
            tracked_data = []
            for tsym, trade in active_tracked_trades.items():
                tracked_data.append({
                    'Symbol': tsym,
                    'Action': trade['buy_or_sell'],
                    'Qty': trade['quantity'],
                    'Entry Price': f"{trade['entry_price']:,.2f}",
                    'SL Price': f"{trade['sl_price']:,.2f}",
                    'Target Price': f"{trade['target_price']:,.2f}",
                    'Status': trade['status']
                })
            st.dataframe(pd.DataFrame(tracked_data))
        else:
            st.info("No active trades being tracked by the app.")

    # --- Monitor Open Positions for SL/TP (triggers closing orders) ---
    monitor_open_trades(api)
    
    # Wait for the specified interval before the next screening cycle
    if run_screener: # Only sleep if continuous mode is still active
        st.info(f"Next screening cycle in {screen_interval} seconds...")
        time.sleep(screen_interval)
        # No st.rerun() here, only updates placeholders

# Message if continuous screener is off
if not run_screener:
    status_placeholder.info("Screener is paused. Check 'Run Screener Continuously' in sidebar to start.")
