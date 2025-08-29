import streamlit as st
import os
import sys
import logging
import datetime
import time
import pandas as pd
from supabase import create_client, Client
import requests # New import for Tradetron API
import json # New import for JSON string formatting
# Removed 'import uuid' as it's no longer needed for key generation in this approach

# Add the parent directory to the sys.path to import api_helper
# This assumes api_helper.py is in the parent directory of this script.
# Make sure your api_helper.py is correctly placed relative to this Streamlit app file.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
# Set logging level to DEBUG to see all detailed messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "11369d3987d181d34ea9ea80e8676b3153e890ca6a7dc21eae8c668465470663")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL","https://zybakxpyibubzjhzdcwl.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 21 # Latest candle + previous 20 for calculations

# --- Entry/Exit Buffers and Risk Parameters (Constants) ---
RISK_PERCENTAGE_OF_CAPITAL = 0.01 # 1% of capital risked per trade


# --- Initialize ALL session state variables first and foremost ---
if 'volume_multiplier' not in st.session_state:
    st.session_state.volume_multiplier = 10
if 'traded_value_divisor' not in st.session_state:
    st.session_state.traded_value_divisor = 100
if 'high_low_diff_multiplier' not in st.session_state:
    st.session_state.high_low_diff_multiplier = 4
if 'capital' not in st.session_state:
    st.session_state.capital = 1000
if 'target_multiplier' not in st.session_state:
    st.session_state.target_multiplier = 4 # Target is X times potential loss
if 'sl_buffer_points' not in st.session_state:
    st.session_state.sl_buffer_points = 0.25 # Points buffer below signal low for initial SL
if 'trailing_step_points' not in st.session_state:
    st.session_state.trailing_step_points = 1.00 # Points for trailing SL adjustment
if 'entry_buffer_percent' not in st.session_state:
    st.session_state.entry_buffer_percent = 0.0005 # 0.05% buffer for crossing high/low for entry

if 'pending_entries' not in st.session_state:
    st.session_state.pending_entries = {} # {tsym: {signal_candle_high, signal_candle_low, buy_or_sell, initial_sl_price, initial_tp_price, calculated_quantity, token}}

if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {} # {tsym: {order_no, entry_price, quantity, sl_price, target_price, buy_or_sell, status, token, highest_price_seen, lowest_price_seen}}

# Initialize session state
if 'manual_overrides' not in st.session_state:
    st.session_state.manual_overrides = {}

if 'market_watch_symbols' not in st.session_state:
    st.session_state.market_watch_symbols = [] # List of tsyms for market watch

if 'supabase_loaded' not in st.session_state:
    st.session_state.supabase_loaded = False
if 'exit_all_triggered_today' not in st.session_state:
    st.session_state.exit_all_triggered_today = False
if 'last_run_date' not in st.session_state:
    st.session_state.last_run_date = datetime.date.today() # Initialize with current date

if 'tradetron_cookie' not in st.session_state:
    st.session_state.tradetron_cookie = ""
if 'tradetron_user_agent' not in st.session_state:
    st.session_state.tradetron_user_agent = ""
if 'market_watch_source' not in st.session_state:
    st.session_state.market_watch_source = "Flattrade (NorenApiPy)"
    
if 'daily_traded_symbols' not in st.session_state:
    st.session_state.daily_traded_symbols = set()  # Track symbols traded today
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.date.today()

# Reset daily traded symbols at start of new day
current_date = datetime.date.today()
if st.session_state.last_reset_date != current_date:
    st.session_state.daily_traded_symbols = set()
    st.session_state.last_reset_date = current_date

# 1. Add global variable to track current balance (add this near the top after other globals)
if 'current_account_balance' not in st.session_state:
    st.session_state.current_account_balance = None

# --- End of session state initialization ---

# Global for Supabase client
supabase: 'Client' = None 

@st.cache_resource
def get_supabase_client(url, key):
    """Initializes and caches the Supabase client."""
    if not url or not key:
        st.error("Supabase URL or Key is not set in Streamlit secrets. Please configure them.")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}. Please check your credentials.")
        return None

# Initialize Supabase client
supabase = get_supabase_client(SUPABASE_URL, SUPABASE_KEY)

if supabase is None:
    st.stop() # Stop the app if Supabase connection fails

# --- Supabase Database Operations ---
def upsert_trade_to_supabase(trade_data):
    """Inserts or updates a trade record in Supabase."""
    tsym = trade_data['tsym']
    
    # Clean up None values for Supabase to avoid type errors with nullable fields
    for key, value in trade_data.items():
        if value is None:
            trade_data[key] = None # Ensure None, not empty string or other default
        elif isinstance(value, float) and (value == float('inf') or value == float('-inf') or pd.isna(value)):
             trade_data[key] = None # Handle infinite or NaN floats
    
    # Supabase uses `upsert` for insert/update based on unique keys
    try:
        # Check if the trade already exists by tsym (unique identifier for our purpose)
        response = supabase.from_('app_tracked_trades').select('id').eq('tsym', tsym).limit(1).execute()
        
        if response.data:
            # Trade exists, update it by ID
            trade_id = response.data[0]['id']
            # Remove 'id' if present, as we update by it, not include it in payload
            data_to_update = {k: v for k, v in trade_data.items() if k not in ['id', 'created_at']}
            updated_response = supabase.from_('app_tracked_trades').update(data_to_update).eq('id', trade_id).execute()
            if updated_response.data:
                logging.info(f"Updated trade {tsym} in Supabase: {updated_response.data}")
                return True
            else:
                logging.error(f"Failed to update trade {tsym} in Supabase: {updated_response.status_code} - {updated_response.get('error', 'No error message')}")
                return False
        else:
            # Trade does not exist, insert new record
            # Remove 'id' if present, as Supabase will generate one
            data_to_insert = {k: v for k, v in trade_data.items() if k != 'id'}
            inserted_response = supabase.from_('app_tracked_trades').insert(data_to_insert).execute()
            if inserted_response.data:
                logging.info(f"Inserted new trade {tsym} into Supabase: {inserted_response.data}")
                return True
            else:
                logging.error(f"Failed to insert trade {tsym} into Supabase: {inserted_response.status_code} - {inserted_response.get('error', 'No error message')}")
                return False
    except Exception as e:
        logging.error(f"Supabase upsert error for {tsym}: {e}", exc_info=True)
        return False

def delete_trade_from_supabase(tsym):
    """Deletes a trade record from Supabase."""
    try:
        response = supabase.from_('app_tracked_trades').delete().eq('tsym', tsym).execute()
        if response.data:
            logging.info(f"Deleted trade {tsym} from Supabase: {response.data}")
            return True
        else:
            logging.error(f"Failed to delete trade {tsym} from Supabase: {response.status_code} - {response.get('error', 'No error message')}")
            return False
    except Exception as e:
        logging.error(f"Supabase delete error for {tsym}: {e}", exc_info=True)
        return False

def load_tracked_trades_from_supabase():
    """Loads today's tracked trades from Supabase into session state."""
    try:
        # Get the start of the current day in ISO format for the filter
        today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Filter the query to only load trades created today
        response = supabase.from_('app_tracked_trades').select('*').gt('created_at', today_start).execute()

        if response.data:
            logging.info(f"Loaded {len(response.data)} trades from Supabase for today.")
            for trade_record in response.data:
                tsym = trade_record['tsym']
                status = trade_record['status']

                # Convert numeric fields back from Decimal/str to float if necessary, handle None
                for key in ['entry_price', 'sl_price', 'target_price', 'highest_price_seen', 
                            'lowest_price_seen', 'signal_candle_high', 'signal_candle_low',
                            'manual_sl_price', 'manual_target_price']:
                    if key in trade_record and trade_record[key] is not None:
                        try:
                            trade_record[key] = float(trade_record[key])
                        except (ValueError, TypeError):
                            trade_record[key] = None # Set to None if conversion fails

                if status == 'PENDING':
                    st.session_state.pending_entries[tsym] = {
                        'buy_or_sell': trade_record['buy_or_sell'],
                        'signal_candle_high': trade_record['signal_candle_high'],
                        'signal_candle_low': trade_record['signal_candle_low'],
                        'calculated_quantity': trade_record['quantity'],
                        'initial_sl_price': trade_record['sl_price'], # Stored as sl_price in DB
                        'initial_tp_price': trade_record['target_price'], # Stored as target_price in DB
                        'status': 'PENDING',
                        'token': trade_record['token'],
                        'current_ltp': None # Initialize for pending entries
                    }
                    if trade_record['manual_sl_price'] is not None or trade_record['manual_target_price'] is not None:
                         st.session_state.manual_overrides.setdefault(tsym, {})['sl_price'] = trade_record['manual_sl_price']
                         st.session_state.manual_overrides.setdefault(tsym, {})['target_price'] = trade_record['manual_target_price']

                elif status == 'OPEN':
                    st.session_state.open_tracked_trades[tsym] = {
                        'order_no': trade_record.get('order_no'), # May not be present for adopted
                        'entry_price': trade_record['entry_price'],
                        'quantity': trade_record['quantity'],
                        'sl_price': trade_record['sl_price'],
                        'target_price': trade_record['target_price'],
                        'buy_or_sell': trade_record['buy_or_sell'],
                        'status': 'OPEN',
                        'token': trade_record['token'],
                        'highest_price_seen': trade_record['highest_price_seen'],
                        'lowest_price_seen': trade_record['lowest_price_seen'],
                        'current_ltp': None # Initialize for open trades
                    }
                    if trade_record['manual_sl_price'] is not None or trade_record['manual_target_price'] is not None:
                         st.session_state.manual_overrides.setdefault(tsym, {})['sl_price'] = trade_record['manual_sl_price']
                         st.session_state.manual_overrides.setdefault(tsym, {})['target_price'] = trade_record['manual_target_price']
            return True
        else:
            logging.info("No trades found in Supabase for today.")
            return False
    except Exception as e:
        logging.error(f"Supabase load error: {e}", exc_info=True)
        return False

# Load trades from Supabase on first run
if not st.session_state.supabase_loaded:
    if load_tracked_trades_from_supabase():
        st.session_state.supabase_loaded = True
    else:
        st.session_state.supabase_loaded = True # Mark as loaded even if no data


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
        logging.critical(f"An exception occurred during API setup: {e}", exc_info=True)
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
    Filters for 'EQ' (Equity) and 'IND' (Index) instruments. Assumes 'Token' and 'Tradingsymbol' are present.
    Returns a dictionary mapping tradingsymbol (tsym) to its token.
    """
    try:
        df = pd.read_csv(file_path)
        if all(col in df.columns for col in ['Exchange', 'Token', 'Tradingsymbol', 'Instrument']):
            equity_symbols = df[(df['Instrument'] == 'EQ') | (df['Instrument'] == 'IND')][['Exchange', 'Token', 'Tradingsymbol']].copy()
            
            # Create a dictionary mapping tsym to token
            symbols_map = {row['Tradingsymbol']: str(row['Token']) for index, row in equity_symbols.iterrows()}
            st.success(f"Loaded {len(symbols_map)} equity and index symbols from {file_path}.")
            return symbols_map
        else:
            st.error(f"CSV file '{file_path}' must contain 'Exchange', 'Token', 'Tradingsymbol', and 'Instrument' columns. Found: {', '.join(df.columns)}")
            logging.error(f"CSV file '{file_path}' missing required columns. Found: {', '.join(df.columns)}")
            return {}
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory as the Streamlit app.")
        logging.error(f"Error: '{file_path}' not found.", exc_info=True)
        return {}
    except Exception as e:
        st.error(f"Error loading symbols from CSV: {e}")
        logging.error(f"Error loading symbols from CSV: {e}", exc_info=True)
        return {}

def calculate_current_exposure(api):
    """
    Calculates the total traded value of current open positions.
    Returns the sum of (quantity * current_price) for all open positions.
    """
    total_exposure = 0.0
    try:
        # Get current positions from broker
        positions = api.get_positions()
        if positions and isinstance(positions, list):
            for position in positions:
                net_qty = int(position.get('netqty', 0))
                if net_qty != 0:  # Only consider non-zero positions
                    # Get current market price
                    token = position.get('token')
                    tsym = position.get('tsym')
                    if token:
                        try:
                            quote_resp = api.get_quotes(exchange=EXCHANGE, token=token)
                            if quote_resp and quote_resp.get('stat') == 'Ok' and quote_resp.get('values'):
                                current_price = float(quote_resp['values'][0]['lp'])
                                position_value = abs(net_qty) * current_price
                                total_exposure += position_value
                                logging.info(f"Position exposure for {tsym}: {abs(net_qty)} x {current_price} = {position_value}")
                        except Exception as e:
                            logging.error(f"Error getting price for position {tsym}: {e}")
                            # If we can't get current price, use average price from position
                            avg_price = float(position.get('avgprc', 0))
                            if avg_price > 0:
                                position_value = abs(net_qty) * avg_price
                                total_exposure += position_value
                                logging.warning(f"Using avg price for {tsym}: {abs(net_qty)} x {avg_price} = {position_value}")
        
        logging.info(f"Total current exposure: {total_exposure}")
        return total_exposure
        
    except Exception as e:
        logging.error(f"Error calculating current exposure: {e}")
        return 0.0

def calculate_new_order_value(quantity, current_ltp):
    """
    Calculates the traded value for the new order being placed.
    """
    return quantity * current_ltp

def check_balance_before_order(api, new_order_quantity, new_order_ltp, available_balance=None):
    """
    Checks if placing a new order would exceed the leverage limit.
    
    Args:
        api: Trading API instance
        new_order_quantity: Quantity of the new order
        new_order_ltp: Current LTP for the new order
        available_balance: Available cash balance (fetched if None)
    
    Returns:
        tuple: (can_place_order: bool, reason: str, current_exposure: float, new_order_value: float, total_exposure: float, max_allowed: float)
    """
    try:
        # Get current balance if not provided
        if available_balance is None:
            limits = api.get_limits()
            if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
                available_balance = None
                
                if 'cash' in limits and limits['cash'] is not None:
                    try:
                        available_balance = float(limits['cash'])
                    except ValueError:
                        logging.error(f"Could not convert top-level 'cash' to float: {limits['cash']}")

                if available_balance is None and 'prange' in limits and isinstance(limits['prange'], list):
                    for item in limits['prange']:
                        if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                            try:
                                available_balance = float(item['cash'])
                                break
                            except ValueError:
                                continue
                
                if available_balance is None:
                    return False, "Could not retrieve account balance", 0, 0, 0, 0
            else:
                return False, "Failed to fetch account limits", 0, 0, 0, 0
        
        # Calculate current exposure
        current_exposure = calculate_current_exposure(api)
        
        # Calculate new order value
        new_order_value = calculate_new_order_value(new_order_quantity, new_order_ltp)
        
        # Calculate total exposure if this order is placed
        total_exposure = current_exposure + new_order_value
        
        # Calculate maximum allowed exposure (balance * 4.5)
        max_allowed_exposure = available_balance * 4.5
        
        # Check if total exposure exceeds limit
        can_place = total_exposure <= max_allowed_exposure
        
        if can_place:
            reason = f"Order allowed: Total exposure {total_exposure:,.2f} <= Max allowed {max_allowed_exposure:,.2f}"
        else:
            reason = f"Order rejected: Total exposure {total_exposure:,.2f} > Max allowed {max_allowed_exposure:,.2f}"
        
        logging.info(f"Balance check - Balance: {available_balance:,.2f}, Current exposure: {current_exposure:,.2f}, New order: {new_order_value:,.2f}, Total: {total_exposure:,.2f}, Max allowed: {max_allowed_exposure:,.2f}, Result: {can_place}")
        
        return can_place, reason, current_exposure, new_order_value, total_exposure, max_allowed_exposure
        
    except Exception as e:
        logging.error(f"Error in balance check: {e}")
        return False, f"Error during balance check: {e}", 0, 0, 0, 0

def should_stop_screening(api, available_balance=None):
    """
    Determines if screening should be stopped due to leverage limits.
    This is a more conservative check - stops screening if we're already near the limit.
    
    Args:
        api: Trading API instance
        available_balance: Available cash balance (fetched if None)
    
    Returns:
        tuple: (should_stop: bool, reason: str)
    """
    try:
        # Get current balance if not provided
        if available_balance is None:
            limits = api.get_limits()
            if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
                available_balance = None
                
                if 'cash' in limits and limits['cash'] is not None:
                    try:
                        available_balance = float(limits['cash'])
                    except ValueError:
                        pass

                if available_balance is None and 'prange' in limits and isinstance(limits['prange'], list):
                    for item in limits['prange']:
                        if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                            try:
                                available_balance = float(item['cash'])
                                break
                            except ValueError:
                                continue
                
                if available_balance is None:
                    return True, "Could not retrieve account balance - stopping screening"
        
        # Calculate current exposure
        current_exposure = calculate_current_exposure(api)
        
        # Calculate maximum allowed exposure
        max_allowed_exposure = available_balance * 4.5
        
        # Stop screening if we're using more than 90% of allowed leverage
        # This leaves room for one more reasonably sized trade
        utilization_threshold = 0.90
        current_utilization = current_exposure / max_allowed_exposure if max_allowed_exposure > 0 else 1.0
        
        should_stop = current_utilization >= utilization_threshold
        
        if should_stop:
            reason = f"Stopping screening: Current utilization {current_utilization:.1%} >= threshold {utilization_threshold:.1%}"
        else:
            reason = f"Continuing screening: Current utilization {current_utilization:.1%} < threshold {utilization_threshold:.1%}"
        
        logging.info(f"Screening check - {reason}")
        return should_stop, reason
        
    except Exception as e:
        logging.error(f"Error in screening stop check: {e}")
        return True, f"Error during screening check: {e}"


def get_nifty500_symbols():
    """
    Uses the load_symbols_from_csv function to get the actual symbols.
    Returns the map of tsym to token.
    """
    return load_symbols_from_csv()

def get_tradetron_ltp(symbol, tradetron_cookie, tradetron_user_agent):
    """
    Fetches the Last Traded Price (LTP) for a given symbol using the Tradetron API.
    This function uses 1-minute candle data and takes the closing price of the latest candle.
    Requires a valid cookie and user-agent for authentication.
    """
    if not tradetron_cookie or not tradetron_user_agent:
        logging.warning(f"Tradetron cookie or user-agent not provided for {symbol}. Cannot fetch LTP.")
        return "Auth Missing"

    # Clean up symbol format - remove -EQ suffix if present
    clean_symbol = symbol.replace('-EQ', '').strip()
    
    current_time = datetime.datetime.now()
    etime_ms = int(current_time.timestamp() * 1000)
    # Fetch data for the last 5 minutes to ensure at least one complete candle is received.
    stime_ms = int((current_time - datetime.timedelta(minutes=5)).timestamp() * 1000)

    url = f"https://tradetron.tech/tv/api/v3?symbol={clean_symbol}&stime={stime_ms}&etime={etime_ms}&candle=1m"
    
    headers = {
        "authority": "tradetron.tech",
        "method": "GET",
        "path": "/tv/api/v3",
        "scheme": "https",
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "referer": "https://tradetron.tech/user/dashboard",
        "sec-ch-ua": '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": tradetron_user_agent,
        "cookie": tradetron_cookie
    }

    logging.debug(f"Fetching Tradetron LTP for {symbol} (cleaned: {clean_symbol}) from: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"Full Tradetron response for {symbol}: {data}")
        
        # Print the full response for debugging
        print(f"DEBUG - Symbol: {symbol}, Clean Symbol: {clean_symbol}")
        print(f"DEBUG - Full API Response: {data}")
        print(f"DEBUG - Response Type: {type(data)}")
        
        # Check for "success": true and if 'Data' key exists and is a non-empty list
        if data and data.get('success') is True:
            print(f"DEBUG - Success is True")
            
            if 'Data' in data:
                print(f"DEBUG - Data key found: {data['Data']}")
                print(f"DEBUG - Data type: {type(data['Data'])}")
                print(f"DEBUG - Data length: {len(data['Data']) if isinstance(data['Data'], list) else 'Not a list'}")
                
                if isinstance(data['Data'], list) and len(data['Data']) > 0:
                    latest_candle = data['Data'][0]
                    print(f"DEBUG - Latest candle: {latest_candle}")
                    print(f"DEBUG - Latest candle type: {type(latest_candle)}")
                    
                    if isinstance(latest_candle, dict) and 'close' in latest_candle:
                        ltp = float(latest_candle['close'])
                        print(f"DEBUG - Successfully extracted LTP: {ltp}")
                        logging.info(f"Successfully fetched Tradetron LTP for {symbol}: {ltp}")
                        return ltp
                    else:
                        print(f"DEBUG - Latest candle format issue or missing close: {latest_candle}")
                        logging.warning(f"Tradetron API latest candle for {symbol} is not in expected format: {latest_candle}")
                        return "Parse Error"
                else:
                    print(f"DEBUG - Data is not a list or is empty")
                    logging.warning(f"Tradetron API Data is not a list or empty for {symbol}: {data['Data']}")
                    return "No Data"
            else:
                print(f"DEBUG - No 'Data' key in response")
                logging.warning(f"Tradetron API response missing 'Data' key for {symbol}: {data}")
                return "No Data"
        else:
            print(f"DEBUG - Success is not True or data is None")
            print(f"DEBUG - Success value: {data.get('success') if data else 'No data'}")
            logging.warning(f"Tradetron API returned success=False or no data for {symbol}: {data}")
            return "No Data"

    except requests.exceptions.HTTPError as e:
        print(f"DEBUG - HTTP Error: {e}")
        logging.error(f"Tradetron HTTP error for {symbol}: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
        if 'response' in locals() and (response.status_code == 401 or response.status_code == 403):
            return "Auth Error (401/403)"
        return "HTTP Error"
    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG - Connection Error: {e}")
        logging.error(f"Tradetron Connection Error for {symbol}: {e}")
        return "Conn Error"
    except requests.exceptions.Timeout as e:
        print(f"DEBUG - Timeout Error: {e}")
        logging.error(f"Tradetron Timeout Error for {symbol}: {e}")
        return "Timeout"
    except requests.exceptions.RequestException as e:
        print(f"DEBUG - Request Error: {e}")
        logging.error(f"Tradetron Request Error for {symbol}: {e}")
        return "Req Error"
    except ValueError as e:
        print(f"DEBUG - Value/JSON Error: {e}")
        logging.error(f"Tradetron data parsing error for {symbol}: {e}")
        return "Parse Error"
    except Exception as e:
        print(f"DEBUG - Unexpected Error: {e}")
        logging.error(f"Unexpected error fetching Tradetron LTP for {symbol}: {e}", exc_info=True)
        return "Unknown Error"

def get_tradetron_ltp_simple(symbol, tradetron_cookie, tradetron_user_agent):
    """ Simplified version for production use after debugging """
    if not tradetron_cookie or not tradetron_user_agent:
        return "Auth Missing"
    
    current_time = datetime.datetime.now()
    # Try different time ranges if first one fails
    time_ranges = [15, 30, 60, 120]  # minutes
    
    for minutes in time_ranges:
        etime_ms = int(current_time.timestamp() * 1000)
        stime_ms = int((current_time - datetime.timedelta(minutes=minutes)).timestamp() * 1000)
        
        url = f"https://tradetron.tech/tv/api/v3?symbol={symbol}&stime={stime_ms}&etime={etime_ms}&candle=1m"
        
        headers = {
            "authority": "tradetron.tech",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "referer": "https://tradetron.tech/user/dashboard",
            "user-agent": tradetron_user_agent,
            "cookie": tradetron_cookie
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if (data and data.get('success') is True and 
                    'Data' in data and 
                    isinstance(data['Data'], list) and 
                    len(data['Data']) > 0 and 
                    isinstance(data['Data'][0], dict) and 
                    'close' in data['Data'][0]):
                    return float(data['Data'][0]['close'])
        except Exception as e:
            logging.error(f"Error fetching Tradetron data for {symbol} with {minutes}min range: {e}")
            continue
            
    # If all time ranges fail
    return "No Data"

def comprehensive_tradetron_debug():
    """
    Comprehensive debugging for Tradetron API issues
    """
    st.write("## 🔍 Comprehensive Tradetron API Debug")
    
    # Step 1: Check credentials
    st.write("### Step 1: Credentials Check")
    cookie = st.session_state.tradetron_cookie
    user_agent = st.session_state.tradetron_user_agent
    
    if not cookie:
        st.error("❌ Cookie is missing")
        return
    else:
        st.success(f"✅ Cookie present (length: {len(cookie)})")
    
    if not user_agent:
        st.error("❌ User-Agent is missing")
        return
    else:
        st.success(f"✅ User-Agent present: {user_agent[:50]}...")
    
    # Step 2: Test basic connectivity
    st.write("### Step 2: Test Basic Connectivity")
    test_url = "https://tradetron.tech/user/dashboard"
    
    try:
        response = requests.get(test_url, headers={"cookie": cookie, "user-agent": user_agent}, timeout=5)
        if response.status_code == 200:
            st.success("✅ Basic connectivity to Tradetron successful")
        else:
            st.error(f"❌ Connectivity issue: Status {response.status_code}")
            st.write("This might indicate your cookie has expired")
    except Exception as e:
        st.error(f"❌ Connectivity error: {e}")
    
    # Step 3: Test API endpoint with different parameters
    st.write("### Step 3: API Endpoint Testing")
    
    symbol = st.selectbox("Select symbol to test:", ["ACC", "RELIANCE", "INFY", "TCS", "HDFCBANK"])
    
    if st.button("Run Comprehensive Test"):
        # Test different time ranges
        current_time = datetime.datetime.now()
        
        time_ranges = [
            ("Last 5 minutes", 5),
            ("Last 15 minutes", 15),
            ("Last 30 minutes", 30),
            ("Last 60 minutes", 60),
            ("Last 2 hours", 120),
        ]
        
        for range_name, minutes in time_ranges:
            st.write(f"**Testing {range_name}:**")
            
            etime_ms = int(current_time.timestamp() * 1000)
            stime_ms = int((current_time - datetime.timedelta(minutes=minutes)).timestamp() * 1000)
            
            url = f"https://tradetron.tech/tv/api/v3?symbol={symbol}&stime={stime_ms}&etime={etime_ms}&candle=1m"
            
            st.write(f"URL: {url}")
            st.write(f"Time range: {datetime.datetime.fromtimestamp(stime_ms/1000)} to {datetime.datetime.fromtimestamp(etime_ms/1000)}")
            
            headers = {
                "authority": "tradetron.tech",
                "method": "GET",
                "path": "/tv/api/v3",
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "referer": "https://tradetron.tech/user/dashboard",
                "sec-ch-ua": '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": user_agent,
                "cookie": cookie
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                st.write(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        st.write(f"Raw Response: {data}")
                        
                        # Detailed analysis
                        if isinstance(data, dict):
                            st.write(f"Response keys: {list(data.keys())}")
                            
                            if 'success' in data:
                                st.write(f"Success: {data['success']}")
                                
                            if 'Data' in data:
                                st.write(f"Data type: {type(data['Data'])}")
                                st.write(f"Data content: {data['Data']}")
                                
                                if isinstance(data['Data'], list):
                                    st.write(f"Data length: {len(data['Data'])}")
                                    if len(data['Data']) > 0:
                                        st.write(f"First item: {data['Data'][0]}")
                                        if isinstance(data['Data'][0], dict) and 'close' in data['Data'][0]:
                                            ltp = data['Data'][0]['close']
                                            st.success(f"🎉 SUCCESS! Found LTP: {ltp}")
                                            return  # Found working configuration
                            
                            if 'error' in data:
                                st.error(f"API Error: {data['error']}")
                                
                        else:
                            st.error(f"Unexpected response format: {type(data)}")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"JSON decode error: {e}")
                        st.write(f"Raw response text: {response.text[:500]}...")
                        
                elif response.status_code == 401:
                    st.error("❌ 401 Unauthorized - Your cookie has expired")
                    st.write("Please get a fresh cookie from...")
            except requests.exceptions.ConnectionError as e:
                st.error(f"❌ Connection error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                
    st.write("---")
    
    # Step 4: Manual URL test
    st.write("### Step 4: Manual URL Test")
    st.write("Copy this URL and test it manually in your browser while logged into Tradetron:")
    current_time = datetime.datetime.now()
    etime_ms = int(current_time.timestamp() * 1000)
    stime_ms = int((current_time - datetime.timedelta(minutes=15)).timestamp() * 1000)
    manual_test_url = f"https://tradetron.tech/tv/api/v3?symbol=ACC&stime={stime_ms}&etime={etime_ms}&candle=1m"
    st.code(manual_test_url)
    st.write("If this URL works in your browser but not in the app, it's a cookie/authentication issue.")

if st.sidebar.button("Debug Tradetron API"):
    comprehensive_tradetron_debug()

if st.button("Test Single Symbol"):
    test_result = get_tradetron_ltp("ACC", st.session_state.tradetron_cookie, st.session_state.tradetron_user_agent)
    st.write(f"Test result: {test_result}")


# Moved function definitions here to ensure they are defined before being called
def fetch_and_update_ltp():
    """
    Fetches LTP from Tradetron API for all pending and open trades and updates the session state.
    This creates a live market watch for the tracked trades using Tradetron API.
    """
    tradetron_cookie = st.session_state.tradetron_cookie
    tradetron_user_agent = st.session_state.tradetron_user_agent
    
    all_tracked_tsyms = list(st.session_state.pending_entries.keys()) + list(st.session_state.open_tracked_trades.keys())
    
    if not tradetron_cookie or not tradetron_user_agent:
        st.warning("Tradetron credentials are not set. Live price updates disabled.")
        return
        
    for tsym in all_tracked_tsyms:
        ltp = get_tradetron_ltp_simple(tsym, tradetron_cookie, tradetron_user_agent)
        
        if isinstance(ltp, float):
            # Update pending entries
            if tsym in st.session_state.pending_entries:
                st.session_state.pending_entries[tsym]['current_ltp'] = ltp
            
            # Update open trades and check for SL/Target hits
            if tsym in st.session_state.open_tracked_trades:
                st.session_state.open_tracked_trades[tsym]['current_ltp'] = ltp
                
                # Update highest/lowest price seen for trailing SL
                trade_info = st.session_state.open_tracked_trades[tsym]
                if trade_info['buy_or_sell'] == 'B': # BUY position
                    if ltp > trade_info['highest_price_seen']:
                        trade_info['highest_price_seen'] = ltp
                else: # SELL position
                    if ltp < trade_info['lowest_price_seen']:
                        trade_info['lowest_price_seen'] = ltp


def calculate_quantity_and_sl_new(entry_price, sl_price, capital):
    """
    Calculate quantity using the new formula: Capital * 0.01 / SL_Points
    Args:
        entry_price (float): Entry price of the trade
        sl_price (float): Stop loss price
        capital (float): Available capital
    Returns:
        tuple: (calculated_quantity, sl_points, potential_loss)
    """
    sl_points = abs(entry_price - sl_price)
    if sl_points == 0:
        return 0, 0, 0
    
    # NEW FORMULA: Quantity = Capital * 0.01 / SL_Points
    calculated_quantity = int((capital * 0.01) / sl_points)
    potential_loss = sl_points * calculated_quantity
    
    return calculated_quantity, sl_points, potential_loss


def calculate_quantity_sl_tp(signal_candle_high, signal_candle_low, signal_type, capital):
    """
    Calculates the quantity, SL, and TP based on signal candle high/low,
    risk percentage of capital, and target multiplier.
    """
    if signal_type == 'BUY':
        # Expected entry price is the high plus a small buffer
        expected_entry_price = round(signal_candle_high * (1 + st.session_state.entry_buffer_percent), 2)
        # SL at signal candle low minus buffer
        sl_price = round(signal_candle_low - st.session_state.sl_buffer_points, 2)
        sl_points = expected_entry_price - sl_price
        # Trailing step = 1x potential loss
        trailing_step = sl_points
    else: # SELL
        # Expected entry price is the low minus a small buffer
        expected_entry_price = round(signal_candle_low * (1 - st.session_state.entry_buffer_percent), 2)
        # SL at signal candle high plus buffer
        sl_price = round(signal_candle_high + st.session_state.sl_buffer_points, 2)
        sl_points = sl_price - expected_entry_price
        # Trailing step = 1x potential loss
        trailing_step = sl_points

    if sl_points <= 0.01:
        return 0, sl_price, 0, trailing_step
    
    # Modified quantity calculation: capital * 0.01 / SL_points
    calculated_quantity = int((capital * 0.01) / sl_points)

    # Target = entry + (potential_loss * multiplier)
    target_price = expected_entry_price + (sl_points * st.session_state.target_multiplier) if signal_type == 'BUY' else expected_entry_price - (sl_points * st.session_state.target_multiplier)
    
    return calculated_quantity, sl_price, target_price, trailing_step


def place_intraday_order(
    buy_or_sell, tradingsymbol, quantity, entry_price, api, token
):
    """
    Places a regular Intraday (MIS) order.
    Returns: order_response dictionary
    """
    if quantity <= 0:
        st.warning(f"Cannot place order for {tradingsymbol}: Calculated quantity is zero or negative ({quantity}).")
        logging.warning(f"Order skipped for {tradingsymbol}: Quantity is zero or negative ({quantity}).")
        return {'stat': 'Not_Ok', 'emsg': 'Quantity is zero or negative'}

    ret = api.place_order(
        buy_or_sell=buy_or_sell,
        product_type='I',  # 'I' for Intraday
        exchange=EXCHANGE,
        tradingsymbol=tradingsymbol,
        quantity=quantity,
        discloseqty=0,
        price_type='MKT',  # Market order
        price=0,
        trigger_price=None,
        retention='DAY',
        token=token
    )
    
    order_response = ret
    try:
        if order_response and order_response.get('stat') == 'Ok':
            st.success(f"Order placed successfully for {tradingsymbol}: {order_response}")
            logging.info(f"Order placed successfully for {tradingsymbol}: {order_response}")
            return order_response
        else:
            error_msg = order_response.get('emsg', 'Unknown error') if isinstance(order_response, dict) else str(order_response)
            st.error(f"Failed to place order for {tradingsymbol}: {error_msg}")
            logging.error(f"Failed to place order for {tradingsymbol}: {error_msg}. Full response: {order_response}")
            return order_response
    except Exception as e:
        st.error(f"An exception occurred while placing order for {tradingsymbol}: {e}")
        logging.error(f"An exception occurred while placing order for {tradingsymbol}: {e}", exc_info=True)
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def exit_position(exchange, tradingsymbol, product_type, netqty, api, token):
    """
    Exits an existing position.
    """
    try:
        # Determine action (BUY or SELL) based on netqty
        buy_or_sell = 'S' if netqty > 0 else 'B'
        
        # Place a market order to close the position
        order_response = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type=product_type,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=abs(netqty),
            discloseqty=0,
            price_type='MKT',
            price=0,
            trigger_price=None,
            retention='DAY',
            token=token
        )
        
        if order_response and order_response.get('stat') == 'Ok':
            st.success(f"Successfully placed exit order for {tradingsymbol}. Order No: {order_response['norenordno']}")
            logging.info(f"Successfully placed exit order for {tradingsymbol}. Order No: {order_response['norenordno']}")
            return True
        else:
            error_msg = order_response.get('emsg', 'Unknown error')
            st.error(f"Failed to place exit order for {tradingsymbol}: {error_msg}")
            logging.error(f"Failed to place exit order for {tradingsymbol}: {error_msg}")
            return False
            
    except Exception as e:
        st.error(f"An exception occurred while exiting position for {tradingsymbol}: {e}")
        logging.error(f"Exception during exit for {tradingsymbol}: {e}", exc_info=True)
        return False
        
def fetch_daily_data(api, token, tsym):
    """
    Fetches the previous day's data using get_daily_price_series.
    """
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    # The API takes dates in milliseconds
    yesterday_ms = int(yesterday.timestamp() * 1000)
    
    try:
        daily_data = api.get_daily_price_series(exchange=EXCHANGE, token=token, startdate=yesterday_ms)
        if daily_data and daily_data.get('stat') == 'Ok' and 'values' in daily_data:
            # The last element should be the previous day's data
            prev_day_data = daily_data['values'][-1]
            return prev_day_data
        else:
            logging.warning(f"Could not fetch daily data for {tsym}: {daily_data}")
            return None
    except Exception as e:
        logging.error(f"Error fetching daily data for {tsym}: {e}")
        return None

# Updated check_indicator_conditions function
def check_indicator_conditions(historical_data, tsym, required_candles):
    """
    Validates historical candle data and checks if indicator conditions are met.
    
    Args:
        historical_data (list or dict): The data returned from the API call.
        tsym (str): The trading symbol.
        required_candles (int): The number of candles needed for calculations.
        
    Returns:
        bool: True if indicator conditions are met, False otherwise.
    """
    try:
        # Step 1: Handle API error responses (which are dictionaries)
        if isinstance(historical_data, dict) and historical_data.get('stat') == 'Not_Ok':
            error_msg = historical_data.get('emsg', 'Unknown API error')
            logging.warning(f"Flattrade API returned an error for {tsym}: {error_msg}")
            return False

        # Step 2: Check for insufficient candle data (now we know it's a list or None)
        if not isinstance(historical_data, list) or len(historical_data) < required_candles:
            logging.warning(f"Insufficient candle data for {tsym}: Expected {required_candles}, got {len(historical_data) if historical_data else 0}")
            return False

        # Step 3: Perform indicator calculations on the list
        latest_candle = historical_data[-1]
        latest_close = float(latest_candle.get('c'))
        
        previous_close = float(historical_data[-2].get('c'))
        
        if latest_close > previous_close:
            logging.info(f"Price is trending up for {tsym}. Conditions met.")
            return True
        else:
            logging.info(f"Price is not trending up for {tsym}. Conditions not met.")
            return False

    except Exception as e:
        logging.error(f"Error processing candle data for {tsym}: {e}", exc_info=True)
        return False

def screen_stock(stock_info_dict, api_instance, all_symbols_map):
    """
    Modified screening function with new conditions:
    1. Check if symbol already traded today
    2. Open-close range instead of high-low range
    3. New traded value condition
    
    Args:
        stock_info_dict (dict): A dictionary containing 'tsym', 'token', 'exchange'.
        api_instance (NorenApiPy): The initialized NorenApiPy instance.
        all_symbols_map (dict): A map of all tradingsymbols to their tokens.

    Returns:
        tuple: (tradingsymbol, signal_type, reason, signal_high, signal_low, signal, signal_time)
               or (tradingsymbol, 'NEUTRAL', error_message, None, None, None, None) on error
    """
    tradingsymbol = stock_info_dict['tsym']
    token = stock_info_dict['token']
    exchange = stock_info_dict['exchange'] # Assuming 'exchange' is always 'NSE' in your context


    # Check if already traded today
    if tradingsymbol in st.session_state.daily_traded_symbols:
        return tradingsymbol, 'NEUTRAL', 'Already traded today', None, None, None, None

    try:
        # Fetch minute-level candle data - REMOVED 'count' ARGUMENT
        candles_data = api_instance.get_time_price_series(
            exchange=exchange,
            token=token,
            interval=CANDLE_INTERVAL,
            # Removed: count=REQUIRED_CANDLES # This caused the AttributeError
        )

        # Handle API error responses (which are dictionaries)
        if isinstance(candles_data, dict) and candles_data.get('stat') == 'Not_Ok':
            error_msg = candles_data.get('emsg', 'Unknown API error')
            logging.warning(f"Flattrade API returned an error for {tradingsymbol}: {error_msg}")
            return tradingsymbol, 'NEUTRAL', f'API Error: {error_msg}', None, None, None, None
        
        # Check for insufficient candle data (now we know it's a list or None)
        if not isinstance(candles_data, list) or len(candles_data) < REQUIRED_CANDLES:
            logging.warning(f"Insufficient candle data for {tradingsymbol}: Expected {REQUIRED_CANDLES}, got {len(candles_data) if candles_data else 0}")
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None, None, None

        # Sort candles by time (just in case)
        # Note: 'ssboe' for start of candle epoch, 'time' is end of candle epoch
        candles_data.sort(key=lambda x: float(x.get('ssboe', 0))) 

        # Get the latest candle and previous candles for calculations
        latest_candle = candles_data[-1]
        previous_candles_for_avg = candles_data[-REQUIRED_CANDLES:-1] # Get 'REQUIRED_CANDLES - 1' candles before the latest
        
        # Extract current candle data
        current_open_price = float(latest_candle.get('into', 0))
        current_high_price = float(latest_candle.get('inth', 0))
        current_low_price = float(latest_candle.get('intl', 0))
        current_close_price = float(latest_candle.get('intc', 0))
        current_volume = float(latest_candle.get('intv', 0))
        current_traded_value = current_volume * current_close_price # Calculate current_traded_value

        # Check for invalid prices
        if any(p <= 0 for p in [current_open_price, current_high_price, current_low_price, current_close_price]):
             return tradingsymbol, 'NEUTRAL', 'Invalid current candle data', None, None, None, None

        # Get signal candle timestamp for display (using IST)
        # Assuming 'ssboe' is epoch in seconds
        candle_timestamp_s = float(latest_candle.get('ssboe', 0)) 
        signal_candle_time_obj_utc = datetime.datetime.fromtimestamp(candle_timestamp_s, tz=datetime.timezone.utc)
        
        # Convert to IST for display
        ist_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        signal_candle_time_display = signal_candle_time_obj_utc.astimezone(ist_timezone).strftime('%H:%M:%S')

        # --- Candle forming check (simplified) ---
        current_time_ist = datetime.datetime.now(ist_timezone)
        # If the latest candle's end time is in the future, it's still forming.
        # Assuming CANDLE_INTERVAL is in minutes, ssboe + interval gives candle end time.
        candle_end_time_ist = signal_candle_time_obj_utc + datetime.timedelta(minutes=int(CANDLE_INTERVAL))
        candle_end_time_ist = candle_end_time_ist.astimezone(ist_timezone) # Convert to IST for comparison

        if current_time_ist < candle_end_time_ist:
            logging.debug(f"DEBUG: {tradingsymbol} - Latest candle still forming. Skipping. Current IST: {current_time_ist.strftime('%H:%M:%S')}, Candle End IST: {candle_end_time_ist.strftime('%H:%M:%S')}")
            return tradingsymbol, 'NEUTRAL', 'Latest candle still forming', None, None, None, signal_candle_time_display

        # --- 1. Average Volume Check ---
        previous_volumes = [float(c.get('intv', 0)) for c in previous_candles_for_avg if float(c.get('intv', 0)) > 0]
        average_volume = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0

        if not (current_volume > (average_volume * st.session_state.volume_multiplier)):
            return tradingsymbol, 'NEUTRAL', 'Volume condition not met', None, None, None, signal_candle_time_display

        # --- 2. Traded Value Check ---
        prev_day_data = fetch_daily_data(api_instance, token, tradingsymbol) # Use api_instance
        if not prev_day_data:
            return tradingsymbol, 'NEUTRAL', 'Could not get previous day data', None, None, None, signal_candle_time_display
        
        prev_day_close = float(prev_day_data.get('intc', 0))
        prev_day_volume = float(prev_day_data.get('intv', 0))

        if prev_day_close <= 0 or prev_day_volume <= 0:
            return tradingsymbol, 'NEUTRAL', 'Invalid previous day data', None, None, None, signal_candle_time_display

        previous_day_traded_value = prev_day_close * prev_day_volume
        
        if not (current_traded_value > (previous_day_traded_value / st.session_state.traded_value_divisor)):
            return tradingsymbol, 'NEUTRAL', 'Traded value condition not met', None, None, None, signal_candle_time_display

        # --- 3. Open-Close Range Check ---
        current_open_close_diff = abs(current_close_price - current_open_price)
        if current_open_close_diff <= 0:
            return tradingsymbol, 'NEUTRAL', 'Current open-close difference invalid', None, None, None, signal_candle_time_display

        previous_open_close_diffs = []
        for c in previous_candles_for_avg:
            open_price = float(c.get('into', 0))
            close_price = float(c.get('intc', 0))
            if open_price > 0 and close_price > 0:
                diff = abs(close_price - open_price)
                if diff > 0:
                    previous_open_close_diffs.append(diff)
        
        if not previous_open_close_diffs:
            return tradingsymbol, 'NEUTRAL', 'No valid open-close diff data in previous 20 candles', None, None, None, signal_candle_time_display

        average_open_close_diff_last_20 = sum(previous_open_close_diffs) / len(previous_open_close_diffs)

        if not (current_open_close_diff > (average_open_close_diff_last_20 * st.session_state.high_low_diff_multiplier)):
            return tradingsymbol, 'NEUTRAL', 'Open-close range not met', None, None, None, signal_candle_time_display

        # --- 4. Identify Signal Type (BUY or SELL) ---
        signal_type = 'NEUTRAL'
        signal_reason = 'No signal'

        if current_close_price > current_open_price: # Green candle
            signal_type = 'BUY'
            signal_reason = 'Green candle signal'
            # You can add more complex BUY conditions here if needed
        elif current_close_price < current_open_price: # Red candle
            signal_type = 'SELL'
            signal_reason = 'Red candle signal'
            # You can add more complex SELL conditions here if needed
        else:
            signal_reason = 'Current candle is Doji (Open == Close)'


        if signal_type != 'NEUTRAL':
            return tradingsymbol, signal_type, signal_reason, current_high_price, current_low_price, signal_type, signal_candle_time_display
        else:
            return tradingsymbol, 'NEUTRAL', signal_reason, None, None, None, signal_candle_time_display

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}", exc_info=True)
        return tradingsymbol, 'NEUTRAL', f'Error during screening: {e}', None, None, None, None


def is_symbol_already_tracked(tsym):
    """
    Checks if a symbol is already in the pending or open trades list.
    """
    if tsym in st.session_state.pending_entries:
        return True
    if tsym in st.session_state.open_tracked_trades:
        # A symbol can be 'OPEN' but its status can be 'CLOSING'
        # We only want to screen if it's not currently being tracked for an active trade
        if st.session_state.open_tracked_trades[tsym]['status'] == 'OPEN':
            return True
    return False

# Updated screening logic - use this in your main screening loop
def get_eligible_symbols_for_screening(all_symbols_map):
    """
    Returns only symbols that are not already being tracked by the app or traded today
    """
    eligible_for_screening = []
    for tsym, token in all_symbols_map.items():
        if not is_symbol_already_tracked(tsym) and tsym not in st.session_state.daily_traded_symbols:
            eligible_for_screening.append({'tsym': tsym, 'token': token, 'exchange': EXCHANGE})
    return eligible_for_screening


# Additional database cleanup function
def cleanup_stale_database_entries():
    """
    Clean up any stale entries in the database that don't match current session state
    Call this during app startup
    """
    try:
        # Get all entries from database
        response = supabase.from_('app_tracked_trades').select('*').execute()
        if response.data:
            for db_entry in response.data:
                tsym = db_entry['tsym']
                db_status = db_entry['status']
                # Check if this entry exists in current session state
                exists_in_session = False
                if db_status == 'PENDING' and tsym in st.session_state.pending_entries:
                    exists_in_session = True
                elif db_status == 'OPEN' and tsym in st.session_state.open_tracked_trades:
                    exists_in_session = True
                
                # If it doesn't exist in session state, it might be stale
                if not exists_in_session:
                    logging.warning(f"Found potentially stale database entry for {tsym} with status {db_status}")
                    # You can choose to delete it or leave it for manual review
                    # delete_trade_from_supabase(tsym)
    except Exception as e:
        logging.error(f"Error during database cleanup: {e}", exc_info=True)

# Main app layout
st.set_page_config(layout="wide", page_title="Intraday Screener & Trading Bot")

# Global placeholders for dynamic updates
status_placeholder = st.empty()
screener_placeholder = st.empty()
tracked_trades_placeholder = st.empty()
order_response_placeholder = st.empty()

# Sidebar for controls and configuration
st.sidebar.title("Configuration")

# General Controls
screen_interval = st.sidebar.number_input(
    "Screener Run Interval (seconds)", min_value=1, max_value=300, value=60
)
screen_limit = st.sidebar.number_input(
    "Symbols to Screen per Run", min_value=1, max_value=500, value=25
)
run_screener = st.sidebar.checkbox("Start Live Screener", value=False)
enable_trading = st.sidebar.checkbox("Enable Automated Trading", value=False)
st.sidebar.markdown("---")


# Strategy Parameters
st.sidebar.subheader("Strategy Parameters")
st.session_state.volume_multiplier = st.sidebar.slider(
    "Volume Multiplier", min_value=1, max_value=50, value=10, help="Signal candle volume must be > avg volume of last 20 candles * this multiplier."
)

st.session_state.traded_value_divisor = st.sidebar.slider(
    "Traded Value Divisor", min_value=10, max_value=1000, value=100, help="Signal candle traded value must be > previous day's total traded value / this divisor."
)

st.session_state.high_low_diff_multiplier = st.sidebar.slider(
    "Open-Close Range Multiplier", min_value=1, max_value=10, value=4, help="Signal candle open-close range must be > avg open-close range of last 20 candles * this multiplier."
)

st.session_state.target_multiplier = st.sidebar.slider(
    "Target Multiplier", min_value=1, max_value=10, value=4, help="Target price is set at a multiple of the potential loss (SL points)."
)
st.session_state.sl_buffer_points = st.sidebar.number_input(
    "SL Buffer (points)", min_value=0.01, format="%.2f", value=0.25, help="Points buffer for calculating the stop loss."
)
st.session_state.trailing_step_points = st.sidebar.number_input(
    "Trailing SL Step (points)", min_value=0.01, format="%.2f", value=1.00, help="The step size in points for trailing the stop loss."
)
st.session_state.entry_buffer_percent = st.sidebar.number_input(
    "Entry Buffer (%)", min_value=0.0, max_value=1.0, format="%.4f", value=0.0005, help="Percentage buffer for entry price calculation based on signal candle high/low."
)

st.sidebar.markdown("---")

# Capital and Risk Management
st.sidebar.subheader("Capital & Risk Management")
st.session_state.capital = st.sidebar.number_input(
    "Trading Capital (₹)", min_value=1000, value=10000
)
calculated_risk_amount = st.session_state.capital * RISK_PERCENTAGE_OF_CAPITAL
st.sidebar.markdown(f"**Risk Amount per Trade:** ₹{calculated_risk_amount:,.2f}")
st.sidebar.markdown("---")


st.sidebar.subheader("Tradetron API Settings (Experimental)")
st.sidebar.info("Paste your active browser cookie and user-agent here. This will expire and needs manual updates for continued use.")
st.session_state.tradetron_cookie = st.sidebar.text_area(
    "Tradetron Cookie String", value=st.session_state.tradetron_cookie, height=150
)
st.session_state.tradetron_user_agent = st.sidebar.text_input(
    "Tradetron User-Agent Header", value=st.session_state.tradetron_user_agent
)

# Main content area setup for fixed and dynamic sections
st.header("Account Information")
account_info_col, refresh_btn_col = st.columns([0.7, 0.3])
with account_info_col:
    st.info("Fetching account limits...")
    cash_margin = None
    try:
        limits = api.get_limits()
        if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
            if 'cash' in limits and limits['cash'] is not None:
                try:
                    cash_margin = float(limits['cash'])
                except ValueError:
                    logging.error(f"Could not convert top-level 'cash' to float: {limits['cash']}")

            if cash_margin is None and 'prange' in limits and isinstance(limits['prange'], list):
                for item in limits['prange']:
                    if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                        try:
                            cash_margin = float(item['cash'])
                            break
                        except ValueError:
                            continue
            
            if cash_margin is not None:
                st.session_state.current_account_balance = cash_margin  # Store in session state
                
                # Calculate current exposure and show leverage info
                current_exposure = calculate_current_exposure(api)
                max_allowed_exposure = cash_margin * 4.5
                current_utilization = (current_exposure / max_allowed_exposure) if max_allowed_exposure > 0 else 0
                
                st.success(f"**Available Cash:** ₹{cash_margin:,.2f}")
                
                # Show leverage information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Exposure", f"₹{current_exposure:,.0f}")
                with col2:
                    st.metric("Max Allowed (4.5x)", f"₹{max_allowed_exposure:,.0f}")
                with col3:
                    utilization_color = "red" if current_utilization > 0.8 else "orange" if current_utilization > 0.6 else "green"
                    st.metric("Leverage Used", f"{current_utilization:.1%}")
                    
                if current_utilization > 0.9:
                    st.warning("⚠️ High leverage utilization! New orders may be rejected.")
                elif current_utilization > 0.8:
                    st.info("🔶 Moderate leverage utilization. Monitor closely.")
                    
            else:
                st.error(f"Could not retrieve cash margin from API. Full response: {limits}")
                logging.error(f"Could not retrieve cash margin from API. Full response: {limits}")
        else:
            error_msg = limits.get('emsg', 'Unknown error') if isinstance(limits, dict) else str(limits)
            st.error(f"Failed to fetch account limits: {error_msg}")
            logging.error(f"Failed to fetch account limits: {error_msg}. Full response: {limits}")
    except Exception as e:
        st.error(f"An exception occurred while fetching account limits: {e}")
        logging.error(f"An exception occurred while fetching account limits: {e}", exc_info=True)


if refresh_btn_col.button("Refresh Account Info", type="primary"):
    st.rerun()

# --- Market Watch Section (New) ---
st.subheader("Market Watch (Live Data)")
st.session_state.market_watch_source = st.selectbox(
    "Select Live Data Source", 
    ["Flattrade (NorenApiPy)", "Tradetron (Experimental)"],
    index=0
)
all_symbols_map = get_nifty500_symbols()
selected_symbols_for_mw = st.multiselect(
    "Add Symbols to Market Watch",
    options=list(all_symbols_map.keys()),
    default=st.session_state.market_watch_symbols,
    placeholder="Select symbols..."
)
st.session_state.market_watch_symbols = selected_symbols_for_mw
if st.sidebar.button("🔄 Refresh LTP (Tradetron)", type="secondary"):
    fetch_and_update_ltp()
    st.success("LTP data refreshed from Tradetron API!")

market_watch_data = []
for mw_tsym in st.session_state.market_watch_symbols:
    try:
        if st.session_state.market_watch_source == "Flattrade (NorenApiPy)":
            mw_token = all_symbols_map.get(mw_tsym)
            if mw_token:
                quote_resp = api.get_quotes(exchange=EXCHANGE, token=mw_token)
                if quote_resp and quote_resp.get('stat') == 'Ok' and quote_resp.get('values'):
                    ltp_value = float(quote_resp['values'][0]['lp'])
                else:
                    ltp_value = 'No Data'
            else:
                ltp_value = 'Token Missing'
        elif st.session_state.market_watch_source == "Tradetron (Experimental)":
            ltp_from_tradetron = get_tradetron_ltp(
                mw_tsym, st.session_state.tradetron_cookie, st.session_state.tradetron_user_agent
            )
            ltp_value = ltp_from_tradetron
            if isinstance(ltp_value, float):
                ltp_value = f"{ltp_value:.2f}"
            
        market_watch_data.append({
            'Symbol': mw_tsym,
            'LTP': ltp_value
        })
    except Exception as e:
        logging.error(f"Error fetching LTP for {mw_tsym}: {e}", exc_info=True)
        ltp_value = 'Error'
        market_watch_data.append({
            'Symbol': mw_tsym,
            'LTP': ltp_value
        })

if market_watch_data:
    st.dataframe(pd.DataFrame(market_watch_data))
else:
    st.info("Select symbols to view live data.")

# --- Live Screener Section ---
st.subheader("Live Screener Results")
screened_data = []

# This loop runs only if the checkbox is checked
if run_screener:
    # First, check if we should continue screening based on current leverage
    should_stop, stop_reason = should_stop_screening(api, st.session_state.current_account_balance)
    
    if should_stop:
        status_placeholder.warning(f"🛑 Screening stopped: {stop_reason}")
        st.error(f"Automatic screening has been paused due to leverage limits. Current exposure is too high relative to available balance.")
    else:
        status_placeholder.info(f"Screener is running... {stop_reason}")
        
        # Get only symbols not currently being tracked
        eligible_symbols = get_eligible_symbols_for_screening(all_symbols_map)
        
        # Randomize symbols to avoid API rate limiting issues on same symbols
        import random
        random.shuffle(eligible_symbols)

        for i, stock_info in enumerate(eligible_symbols):
            if i >= screen_limit:
                break # Stop after screening the defined limit
                
            tsym = stock_info['tsym']
            logging.info(f"Screening {tsym}...")
            
            # CORRECTED CALL: Pass the NorenApiPy instance 'api' explicitly
            tradingsymbol, signal_type, reason, signal_high, signal_low, signal, signal_time = screen_stock(stock_info, api, all_symbols_map)
            
            # Display the screening result
            screened_data.append({
                'Symbol': tradingsymbol,
                'Signal': signal,
                'Signal Reason': reason,
                'Signal Time': signal_time if signal_time else 'N/A' # signal_time is already formatted string
            })
            
            if signal_type == 'BUY' or signal_type == 'SELL':
                if enable_trading:
                    # Calculate required quantity and SL/TP
                    capital_to_use = st.session_state.capital
                    calculated_quantity, sl_price, target_price, trailing_step = calculate_quantity_sl_tp(
                        signal_high, signal_low, signal_type, capital_to_use
                    )
                    
                    if calculated_quantity > 0:
                        # Get current LTP for balance check
                        current_ltp_for_check = None
                        try:
                            quote_resp = api.get_quotes(exchange=EXCHANGE, token=stock_info['token'])
                            if quote_resp and quote_resp.get('stat') == 'Ok' and quote_resp.get('values'):
                                current_ltp_for_check = float(quote_resp['values'][0]['lp'])
                        except Exception as e:
                            logging.error(f"Error getting LTP for balance check {tsym}: {e}")
                            
                        if current_ltp_for_check:
                            # BALANCE CHECK BEFORE ADDING TO PENDING
                            can_place, balance_reason, current_exp, new_order_val, total_exp, max_allowed = check_balance_before_order(
                                api, calculated_quantity, current_ltp_for_check, st.session_state.current_account_balance
                            )
                            
                            if can_place:
                                logging.info(f"Balance check passed for {tsym}: {balance_reason}")
                                
                                # Prepare pending entry for tracking
                                st.session_state.pending_entries[tsym] = {
                                    'buy_or_sell': signal_type,
                                    'signal_candle_high': signal_high,
                                    'signal_candle_low': signal_low,
                                    'calculated_quantity': calculated_quantity,
                                    'initial_sl_price': sl_price,
                                    'initial_tp_price': target_price,
                                    'status': 'PENDING',
                                    'token': stock_info['token'],
                                    'current_ltp': None # Will be updated by fetch_and_update_ltp
                                }
                                
                                # Add to daily traded list to avoid re-screening
                                st.session_state.daily_traded_symbols.add(tsym)
                                
                                # Store in Supabase
                                supabase_payload = {
                                    'tsym': tsym,
                                    'exchange': EXCHANGE,
                                    'token': stock_info['token'],
                                    'buy_or_sell': signal_type,
                                    'quantity': calculated_quantity,
                                    'entry_price': None,
                                    'sl_price': sl_price,
                                    'target_price': target_price,
                                    'status': 'PENDING',
                                    'highest_price_seen': None,
                                    'lowest_price_seen': None,
                                    'signal_candle_high': signal_high,
                                    'signal_candle_low': signal_low
                                }
                                upsert_trade_to_supabase(supabase_payload)
                                
                                st.success(f"✅ Signal detected for {tsym}! Balance check passed. Added to pending entries.")
                                st.info(f"📊 Balance Check Details: Current Exp: ₹{current_exp:,.0f}, New Order: ₹{new_order_val:,.0f}, Total: ₹{total_exp:,.0f}/₹{max_allowed:,.0f}")
                                
                            else:
                                # Balance check failed - log and display warning
                                logging.warning(f"Balance check failed for {tsym}: {balance_reason}")
                                st.warning(f"⚠️ Signal detected for {tsym} but BALANCE CHECK FAILED: {balance_reason}")
                                st.info(f"📊 Exposure Details: Current: ₹{current_exp:,.0f}, New Order: ₹{new_order_val:,.0f}, Would be: ₹{total_exp:,.0f}, Max Allowed: ₹{max_allowed:,.0f}")
                                
                                # If balance check fails, we might want to stop screening entirely
                                # to avoid generating more signals that can't be traded
                                if total_exp > max_allowed * 0.95:  # If very close to limit
                                    st.error("🛑 Stopping further screening due to leverage limits being reached.")
                                    break  # Exit the screening loop
                        else:
                            st.warning(f"Could not get LTP for balance check on {tsym}. Skipping signal.")
                    else:
                        st.warning(f"Skipping signal for {tsym} due to low calculated quantity ({calculated_quantity}).")

    if screened_data:
        screener_placeholder.dataframe(pd.DataFrame(screened_data))
    else:
        screener_placeholder.info("No screener results to display.")

    # --- Check pending entries and execute orders if conditions met ---
    entries_to_execute = []
    trades_to_close = []
    
    # First, get the latest prices for all tracked trades using Tradetron
    fetch_and_update_ltp()

    # Process Pending Entries
    for tsym, entry_info in list(st.session_state.pending_entries.items()):
        current_ltp = entry_info.get('current_ltp')
        if not current_ltp or not isinstance(current_ltp, (float, int)):
            logging.warning(f"Skipping pending entry check for {tsym}: LTP not available or invalid.")
            continue
            
        signal_candle_high = entry_info['signal_candle_high']
        signal_candle_low = entry_info['signal_candle_low']
        buy_or_sell = entry_info['buy_or_sell']
        quantity = entry_info['calculated_quantity']

        # Determine effective SL and TP (manual override or calculated) for pending entry
        manual_sl_override = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
        manual_tp_override = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
        effective_sl_price = manual_sl_override if manual_sl_override is not None and manual_sl_override > 0 else entry_info['initial_sl_price']
        effective_target_price = manual_tp_override if manual_tp_override is not None and manual_tp_override > 0 else entry_info['initial_tp_price']
        
        st.markdown(f"**Pending {tsym} ({buy_or_sell}):** LTP={current_ltp:.2f}, Signal High={signal_candle_high:.2f}, Signal Low={signal_candle_low:.2f}")

        if buy_or_sell == 'B': # BUY entry
            trigger_price = round(signal_candle_high * (1 + st.session_state.entry_buffer_percent), 2)
            if current_ltp >= trigger_price:
                st.success(f"BUY Entry Triggered for {tsym}! LTP {current_ltp} >= Trigger {trigger_price}")
                entries_to_execute.append({
                    'tsym': tsym,
                    'quantity': quantity,
                    'action': 'B',
                    'entry_price': current_ltp,
                    'token': entry_info['token'],
                    'sl_price': effective_sl_price,
                    'target_price': effective_target_price,
                    'signal_high': signal_candle_high,
                    'signal_low': signal_candle_low
                })
        elif buy_or_sell == 'S': # SELL entry
            trigger_price = round(signal_candle_low * (1 - st.session_state.entry_buffer_percent), 2)
            if current_ltp <= trigger_price:
                st.success(f"SELL Entry Triggered for {tsym}! LTP {current_ltp} <= Trigger {trigger_price}")
                entries_to_execute.append({
                    'tsym': tsym,
                    'quantity': quantity,
                    'action': 'S',
                    'entry_price': current_ltp,
                    'token': entry_info['token'],
                    'sl_price': effective_sl_price,
                    'target_price': effective_target_price,
                    'signal_high': signal_candle_high,
                    'signal_low': signal_candle_low
                })

    # Execute and manage entries
    for entry in entries_to_execute:
        tsym = entry['tsym']
        if tsym in st.session_state.pending_entries:
            # FINAL BALANCE CHECK BEFORE ACTUAL ORDER PLACEMENT
            final_check_passed, final_reason, curr_exp, order_val, tot_exp, max_exp = check_balance_before_order(
                api, entry['quantity'], entry['entry_price'], st.session_state.current_account_balance
            )
            
            if not final_check_passed:
                st.error(f"🚫 Final balance check failed for {tsym}: {final_reason}")
                st.info(f"Order cancelled due to insufficient balance/leverage limits.")
                # Keep the entry in pending state rather than executing
                continue
                
            st.session_state.pending_entries[tsym]['status'] = 'EXECUTING'
            
            st.info(f"✅ Final balance check passed for {tsym}. Placing order...")
            st.info(f"📊 Final Check: Current: ₹{curr_exp:,.0f}, Order: ₹{order_val:,.0f}, Total: ₹{tot_exp:,.0f}/₹{max_exp:,.0f}")
            
            order_response = place_intraday_order(
                buy_or_sell=entry['action'],
                tradingsymbol=tsym,
                quantity=entry['quantity'],
                entry_price=entry['entry_price'],
                api=api,
                token=entry['token']
            )
            
            if order_response and order_response.get('stat') == 'Ok':
                # Remove from pending, add to open tracked trades
                del st.session_state.pending_entries[tsym]
                
                # Check for manual overrides
                manual_sl_price = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
                manual_target_price = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
                
                trade_info = {
                    'tsym': tsym,
                    'exchange': EXCHANGE,
                    'token': entry['token'],
                    'buy_or_sell': entry['action'],
                    'entry_price': entry['entry_price'],
                    'quantity': entry['quantity'],
                    'sl_price': manual_sl_price if manual_sl_price is not None and manual_sl_price > 0 else entry['sl_price'],
                    'target_price': manual_target_price if manual_target_price is not None and manual_target_price > 0 else entry['target_price'],
                    'signal_candle_high': entry['signal_high'],
                    'signal_candle_low': entry['signal_low'],
                    'highest_price_seen': entry['entry_price'],
                    'lowest_price_seen': entry['entry_price'],
                    'status': 'OPEN',
                    'order_no': order_response.get('norenordno'),
                    'current_ltp': entry['entry_price']
                }
                st.session_state.open_tracked_trades[tsym] = trade_info
                
                # Update Supabase with the new OPEN trade
                upsert_trade_to_supabase(trade_info)
                
                # Update account balance in session state (approximate)
                if st.session_state.current_account_balance:
                    # This is an approximation - the actual balance might be different
                    # due to margins, but gives a rough idea for immediate checks
                    order_value = entry['quantity'] * entry['entry_price']
                    # For intraday, we typically need only margin, but this is conservative
                    st.session_state.current_account_balance -= (order_value / 4.5)  # Assuming 4.5x leverage
                    
            else:
                st.error(f"Failed to place entry order for {tsym}. Reverting status to PENDING.")
                if tsym in st.session_state.pending_entries:
                    st.session_state.pending_entries[tsym]['status'] = 'PENDING'
                logging.error(f"Failed to place entry order for {tsym}. Order Response: {order_response}")


    # Process Open Trades for SL/TP hits
    for tsym, trade_info in list(st.session_state.open_tracked_trades.items()):
        current_ltp = trade_info.get('current_ltp')
        if trade_info['status'] != 'OPEN' or not current_ltp or not isinstance(current_ltp, (float, int)):
            continue

        # Check for manual overrides first
        manual_sl_override = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
        manual_tp_override = st.session_state.manual_overrides.get(tsym, {}).get('target_price')

        effective_sl_price = manual_sl_override if manual_sl_override is not None else trade_info['sl_price']
        effective_target_price = manual_tp_override if manual_tp_override is not None else trade_info['target_price']
        
        buy_or_sell = trade_info['buy_or_sell']
        quantity = trade_info['quantity']
        token = trade_info['token']
        entry_price = trade_info['entry_price']

        # --- Trailing SL Logic ---
        # NOTE: Trailing SL is based on a fixed trailing step, not dynamic based on PVI.
        # This simplifies the logic. Trailing step is defined in sidebar.
        trailing_step = st.session_state.trailing_step_points
        new_sl = effective_sl_price

        if buy_or_sell == 'B': # BUY position
            if current_ltp > trade_info['highest_price_seen']:
                trade_info['highest_price_seen'] = current_ltp
                
            # If LTP has moved up by a step since last SL adjustment, raise the SL
            if current_ltp - effective_sl_price > trailing_step:
                new_sl = current_ltp - trailing_step
                # Make sure the new SL is higher than the current SL
                if new_sl > effective_sl_price:
                    trade_info['sl_price'] = new_sl
                    logging.info(f"Trailing SL for {tsym} (BUY) updated from {effective_sl_price:.2f} to {new_sl:.2f}")
                    # Update Supabase
                    upsert_trade_to_supabase(trade_info)
        
        elif buy_or_sell == 'S': # SELL position
            if current_ltp < trade_info['lowest_price_seen']:
                trade_info['lowest_price_seen'] = current_ltp
                
            # If LTP has moved down by a step since last SL adjustment, lower the SL
            if effective_sl_price - current_ltp > trailing_step:
                new_sl = current_ltp + trailing_step
                # Make sure the new SL is lower than the current SL
                if new_sl < effective_sl_price:
                    trade_info['sl_price'] = new_sl
                    logging.info(f"Trailing SL for {tsym} (SELL) updated from {effective_sl_price:.2f} to {new_sl:.2f}")
                    # Update Supabase
                    upsert_trade_to_supabase(trade_info)
        
        # Re-get effective SL price after potential trailing update
        effective_sl_price = trade_info['sl_price']

        # Now, check for stop loss or target hit with effective prices
        if buy_or_sell == 'B': # Long position
            if effective_sl_price is not None and current_ltp <= effective_sl_price:
                st.warning(f"Stoploss HIT for BUY {tsym}! LTP {current_ltp} <= SL {effective_sl_price}")
                trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'SELL', 'token': token, 'status_reason': 'CLOSING_SL'})
                trade_info['status'] = 'CLOSING_SL'
            elif effective_target_price is not None and current_ltp >= effective_target_price:
                st.success(f"Target HIT for BUY {tsym}! LTP {current_ltp} >= Target {effective_target_price}")
                trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'SELL', 'token': token, 'status_reason': 'CLOSING_TP'})
                trade_info['status'] = 'CLOSING_TP'
        elif buy_or_sell == 'S': # Short position
            if effective_sl_price is not None and current_ltp >= effective_sl_price:
                st.warning(f"Stoploss HIT for SELL {tsym}! LTP {current_ltp} >= SL {effective_sl_price}")
                trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'BUY', 'token': token, 'status_reason': 'CLOSING_SL'})
                trade_info['status'] = 'CLOSING_SL'
            elif effective_target_price is not None and current_ltp <= effective_target_price:
                st.success(f"Target HIT for SELL {tsym}! LTP {current_ltp} >= Target {effective_target_price}")
                trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'BUY', 'token': token, 'status_reason': 'CLOSING_TP'})
                trade_info['status'] = 'CLOSING_TP'
        
        # Update Supabase with the new status
        if trade_info['status'].startswith('CLOSING'):
            upsert_trade_to_supabase({'tsym': tsym, 'status': trade_info['status']})

    # Execute market close orders
    for trade in trades_to_close:
        tsym = trade['tsym']
        if enable_trading:
            st.info(f"Placing market exit order for {tsym} due to {trade['status_reason']}...")
            order_response = place_intraday_order(
                buy_or_sell=trade['action'],
                tradingsymbol=tsym,
                quantity=trade['quantity'],
                entry_price=None, # Not needed for MKT order
                api=api,
                token=trade['token']
            )

            if order_response and order_response.get('stat') == 'Ok':
                # Remove from session state and database after successful exit
                if tsym in st.session_state.open_tracked_trades:
                    del st.session_state.open_tracked_trades[tsym]
                delete_trade_from_supabase(tsym)
            else:
                # If exit order fails, log it and keep the trade in a 'CLOSING' state for re-attempt
                st.error(f"Failed to place exit order for {tsym}. Please close manually.")
                logging.error(f"Failed to place exit order for {tsym}. Keeping in 'CLOSING' state.")
                
    # EOD (End of Day) Exit Logic
    now_ist = datetime.datetime.now() # Assume server runs in IST
    market_close_time = now_ist.replace(hour=15, minute=15, second=0, microsecond=0) # 3:15 PM IST
    if now_ist >= market_close_time and enable_trading:
        # Get positions from the broker API, as they might not be tracked in our app
        try:
            broker_positions = api.get_positions()
            if broker_positions and isinstance(broker_positions, list):
                for position in broker_positions:
                    net_qty = int(position.get('netqty', 0))
                    if net_qty != 0:
                        tsym = position.get('tsym')
                        product_type = position.get('prdtype')
                        if tsym and product_type:
                            st.info(f"Auto-exiting open position for {tsym} ({net_qty}) due to EOD rule...")
                            current_token = all_symbols_map.get(tsym)
                            if current_token:
                                exit_position(
                                    exchange=position.get('exch'),
                                    tradingsymbol=tsym,
                                    product_type=product_type,
                                    netqty=net_qty,
                                    api=api,
                                    token=current_token
                                )
        except Exception as e:
            st.error(f"Error fetching broker positions for EOD exit: {e}")
            logging.error(f"Error fetching broker positions for EOD exit: {e}", exc_info=True)

    # Balance and Exposure Summary
    st.subheader("💰 Balance & Exposure Summary")
    if st.session_state.current_account_balance:
        current_exposure = calculate_current_exposure(api)
        max_allowed = st.session_state.current_account_balance * 4.5
        utilization = (current_exposure / max_allowed) if max_allowed > 0 else 0
        remaining_capacity = max_allowed - current_exposure
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Available Cash", f"₹{st.session_state.current_account_balance:,.0f}")
        with col2:
            st.metric("Current Exposure", f"₹{current_exposure:,.0f}")
        with col3:
            st.metric("Remaining Capacity", f"₹{remaining_capacity:,.0f}")
        with col4:
            st.metric("Leverage Utilization", f"{utilization:.1%}")
            
        # Progress bar for leverage utilization
        progress_color = "red" if utilization > 0.8 else "orange" if utilization > 0.6 else "normal"
        st.progress(min(utilization, 1.0))
        
        if utilization > 0.9:
            st.error("🚨 High leverage alert! Consider reducing exposure or stopping new trades.")
        elif utilization > 0.8:
            st.warning("⚠️ Moderate leverage usage. Monitor positions closely.")
    else:
        st.warning("Balance information not available. Please refresh account info.")

    # Add manual refresh button for balance
    if st.button("🔄 Refresh Balance & Exposure", type="secondary"):
        # Force refresh of account balance
        try:
            limits = api.get_limits()
            if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
                cash_margin = None
                if 'cash' in limits and limits['cash'] is not None:
                    try:
                        cash_margin = float(limits['cash'])
                    except ValueError:
                        pass

                if cash_margin is None and 'prange' in limits and isinstance(limits['prange'], list):
                    for item in limits['prange']:
                        if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                            try:
                                cash_margin = float(item['cash'])
                                break
                            except ValueError:
                                continue
                
                if cash_margin is not None:
                    st.session_state.current_account_balance = cash_margin
                    st.success("Balance refreshed successfully!")
                else:
                    st.error("Could not refresh balance.")
            else:
                st.error("Failed to refresh account limits.")
        except Exception as e:
            st.error(f"Error refreshing balance: {e}")
        st.rerun()

# --- Monitor and Display Tracked Trades (new section) ---
with tracked_trades_placeholder.container():
    st.subheader("App-Tracked Trades (Pending & Open) 🚀")
    tracked_data = []
    
    # Adjusted columns for LTP and new manual SL/TP columns
    cols_header = st.columns([0.15, 0.04, 0.04, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.05, 0.05, 0.08])
    cols_header[0].write("**Symbol**")
    cols_header[1].write("**Dir**")
    cols_header[2].write("**Qty**")
    cols_header[3].write("**Status**")
    cols_header[4].write("**LTP**")
    cols_header[5].write("**Entry**")
    cols_header[6].write("**SL**")
    cols_header[7].write("**Target**")
    cols_header[8].write("**Sig. High**")
    cols_header[9].write("**Sig. Low**")
    cols_header[10].write("**M. SL**")
    cols_header[11].write("**M. TP**")
    cols_header[12].write("**Actions**")
    
    all_tracked_tsyms_ordered = sorted(
        list(st.session_state.pending_entries.keys()) + list(st.session_state.open_tracked_trades.keys())
    )
    
    for tsym in all_tracked_tsyms_ordered:
        cols = st.columns([0.15, 0.04, 0.04, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.05, 0.05, 0.08])

        if tsym in st.session_state.pending_entries:
            entry_info = st.session_state.pending_entries[tsym]
            cols[0].write(tsym)
            cols[1].write(entry_info['buy_or_sell'])
            cols[2].write(entry_info['calculated_quantity'])
            cols[3].write("PENDING")
            cols[4].write(f"{entry_info['current_ltp']:.2f}" if isinstance(entry_info['current_ltp'], float) else 'N/A')
            cols[5].write('N/A')
            cols[6].write(f"{entry_info['initial_sl_price']:.2f}")
            cols[7].write(f"{entry_info['initial_tp_price']:.2f}")
            cols[8].write(f"{entry_info['signal_candle_high']:.2f}")
            cols[9].write(f"{entry_info['signal_candle_low']:.2f}")
            # Manual SL/TP columns for display
            manual_sl_val = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
            manual_tp_val = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
            cols[10].write(f"{manual_sl_val:.2f}" if manual_sl_val else 'N/A')
            cols[11].write(f"{manual_tp_val:.2f}" if manual_tp_val else 'N/A')
            
            with cols[12]:
                if st.button("Cancel", key=f"cancel_pending_{tsym}"):
                    del st.session_state.pending_entries[tsym]
                    delete_trade_from_supabase(tsym)
                    st.rerun()

        elif tsym in st.session_state.open_tracked_trades:
            trade_info = st.session_state.open_tracked_trades[tsym]
            cols[0].write(tsym)
            cols[1].write(trade_info['buy_or_sell'])
            cols[2].write(trade_info['quantity'])
            cols[3].write(trade_info['status'])
            
            pnl = (trade_info['current_ltp'] - trade_info['entry_price']) * trade_info['quantity'] if trade_info['buy_or_sell'] == 'B' else (trade_info['entry_price'] - trade_info['current_ltp']) * trade_info['quantity']
            
            cols[4].write(f"{trade_info['current_ltp']:.2f}" if isinstance(trade_info['current_ltp'], float) else 'N/A')
            cols[5].write(f"{trade_info['entry_price']:.2f}")
            
            # Use effective SL and TP for display
            manual_sl_override = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
            manual_tp_override = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
            
            effective_sl = manual_sl_override if manual_sl_override is not None else trade_info['sl_price']
            effective_tp = manual_tp_override if manual_tp_override is not None else trade_info['target_price']
            
            cols[6].write(f"{effective_sl:.2f}")
            cols[7].write(f"{effective_tp:.2f}")
            cols[8].write(f"{trade_info['signal_candle_high']:.2f}")
            cols[9].write(f"{trade_info['signal_candle_low']:.2f}")
            
            # Manual SL/TP columns
            manual_sl_val = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
            manual_tp_val = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
            cols[10].write(f"{manual_sl_val:.2f}" if manual_sl_val else 'N/A')
            cols[11].write(f"{manual_tp_val:.2f}" if manual_tp_val else 'N/A')

            with cols[12]:
                if st.button("Exit", key=f"exit_open_{tsym}"):
                    exit_response = exit_position(
                        exchange=trade_info['exchange'],
                        tradingsymbol=tsym,
                        product_type='I',
                        netqty=trade_info['quantity'] if trade_info['buy_or_sell'] == 'B' else -trade_info['quantity'],
                        api=api,
                        token=trade_info['token']
                    )
                    if exit_response:
                        del st.session_state.open_tracked_trades[tsym]
                        delete_trade_from_supabase(tsym)
                        st.rerun()
        
    if not all_tracked_tsyms_ordered:
        st.info("No active or pending trades being tracked by the app.")

    
    # Wait for the specified interval before the next screening cycle
    if run_screener:
        st.info(f"Next full screening cycle in {screen_interval} seconds...")
        time.sleep(screen_interval)

if not run_screener:
    status_placeholder.info("Screener is not running. Check the 'Start Live Screener' box in the sidebar to begin.")
