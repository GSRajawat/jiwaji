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

# Add the parent directory to the sys.path to import api_helper
# This assumes api_helper.py is in the parent directory of this script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# Import the OLAELECStrategy class from ola.py
try:
    from ola import OLAELECStrategy
except ImportError:
    st.error("Error: Could not import OLAELECStrategy. Please ensure ola.py is in the same directory.")
    st.stop()

# --- Configuration ---
# Set logging level to DEBUG to see all detailed messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "a36fe10399fcc8d580ae35c795d8593b7676a1fde2ce2a80073dfa23d6430bbb")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL","https://zybakxpyibubzjhzdcwl.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 21 # Latest candle + previous 20 for calculations

# --- Entry/Exit Buffers and Risk Parameters (Constants) ---
ENTRY_BUFFER_PERCENT = 0.0005 # 0.05% buffer for crossing high/low for entry
RISK_PERCENTAGE_OF_CAPITAL = 0.01 # 1% of capital risked per trade


# --- Initialize ALL session state variables first and foremost ---
if 'simple_counter' not in st.session_state:
    st.session_state.simple_counter = 0
if 'widget_key_tracker' not in st.session_state:
    st.session_state.widget_key_tracker = {}
if 'volume_multiplier' not in st.session_state:
    st.session_state.volume_multiplier = 10
if 'traded_value_threshold' not in st.session_state:
    st.session_state.traded_value_threshold = 10000000
if 'high_low_diff_multiplier' not in st.session_state:
    st.session_state.high_low_diff_multiplier = 4
if 'capital' not in st.session_state:
    st.session_state.capital = 1000
if 'target_multiplier' not in st.session_state:
    st.session_state.target_multiplier = 4
if 'sl_buffer_points' not in st.session_state:
    st.session_state.sl_buffer_points = 0.25
if 'trailing_step_points' not in st.session_state:
    st.session_state.trailing_step_points = 1.00

if 'pending_entries' not in st.session_state:
    st.session_state.pending_entries = {}
if 'open_tracked_trades' not in st.session_state:
    st.session_state.open_tracked_trades = {}
if 'manual_overrides' not in st.session_state:
    st.session_state.manual_overrides = {}
if 'market_watch_symbols' not in st.session_state:
    st.session_state.market_watch_symbols = []
if 'supabase_loaded' not in st.session_state:
    st.session_state.supabase_loaded = False
if 'exit_all_triggered_today' not in st.session_state:
    st.session_state.exit_all_triggered_today = False
if 'last_run_date' not in st.session_state:
    st.session_state.last_run_date = datetime.date.today()
if 'tradetron_cookie' not in st.session_state:
    st.session_state.tradetron_cookie = ""
if 'tradetron_user_agent' not in st.session_state:
    st.session_state.tradetron_user_agent = ""
if 'market_watch_source' not in st.session_state:
    st.session_state.market_watch_source = "Flattrade (NorenApiPy)"
if 'daily_traded_symbols' not in st.session_state:
    st.session_state.daily_traded_symbols = set()
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.date.today()
if 'strategy' not in st.session_state:
    st.session_state.strategy = OLAELECStrategy()

# Reset daily traded symbols at start of new day
current_date = datetime.date.today()
if st.session_state.last_reset_date != current_date:
    st.session_state.daily_traded_symbols = set()
    st.session_state.last_reset_date = current_date
    st.session_state.strategy.reset_daily_flags(current_date)

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

# SOLUTION 1: Define current_manual_sl before using it
def fix_manual_sl_input_v1(cols, tsym):
    """
    Fix by defining current_manual_sl from session state
    """
    current_manual_sl = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
    if current_manual_sl is None or current_manual_sl <= 0:
        current_manual_sl = 0.0

    unique_timestamp = int(time.time() * 1000000)

    new_manual_sl = cols[8].number_input(
        "Manual SL",
        value=current_manual_sl,
        step=0.01,
        format="%.2f",
        key=f'manual_sl_{tsym}_pending_{unique_timestamp}_{hash(tsym) % 1000}'
    )
    return new_manual_sl

def display_trades_with_unique_keys():
    """
    Display trades with guaranteed unique keys
    """
    if hasattr(st.session_state, 'pending_trades'):
        display_trades_safely(st.session_state.pending_trades, "pending")
        st.subheader("📋 Pending Trades")
        for tsym, trade_data in st.session_state.pending_trades.items():
            cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            cols[0].write(tsym)
            cols[1].write(trade_data.get('buy_or_sell', 'N/A'))
            cols[2].write(str(trade_data.get('quantity', 0)))
            current_manual_sl = st.session_state.manual_overrides.get(tsym, {}).get('sl_price', 0.0)
            if current_manual_sl is None or current_manual_sl < 0:
                current_manual_sl = 0.0
            st.session_state.simple_counter += 1
            unique_key = f"manual_sl_{tsym}_pending_{st.session_state.simple_counter}"
            new_manual_sl = cols[8].number_input(
                "Manual SL",
                value=current_manual_sl,
                step=0.01,
                format="%.2f",
                key=unique_key
            )
    if hasattr(st.session_state, 'open_tracked_trades'):
        display_trades_safely(st.session_state.open_tracked_trades, "open")
        st.subheader("🔄 Open Trades")
        for tsym, trade_info in st.session_state.open_tracked_trades.items():
            cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            cols[0].write(tsym)
            cols[1].write(trade_info.get('buy_or_sell', 'N/A'))
            cols[2].write(str(trade_info.get('quantity', 0)))
            current_manual_sl = st.session_state.manual_overrides.get(tsym, {}).get('sl_price', 0.0)
            if current_manual_sl is None or current_manual_sl < 0:
                current_manual_sl = trade_info.get('sl_price', 0.0)
            st.session_state.simple_counter += 1
            unique_key = f"manual_sl_{tsym}_open_{st.session_state.simple_counter}"
            new_manual_sl = cols[8].number_input(
                "Manual SL",
                value=current_manual_sl,
                step=0.01,
                format="%.2f",
                key=unique_key
            )
            if new_manual_sl != current_manual_sl:
                if tsym not in st.session_state.manual_overrides:
                    st.session_state.manual_overrides[tsym] = {}
                st.session_state.manual_overrides[tsym]['sl_price'] = new_manual_sl

# --- Supabase Database Operations ---
def upsert_trade_to_supabase(trade_data):
    """Inserts or updates a trade record in Supabase."""
    tsym = trade_data['tsym']
    for key, value in trade_data.items():
        if value is None:
            trade_data[key] = None
        elif isinstance(value, float) and (value == float('inf') or value == float('-inf') or pd.isna(value)):
             trade_data[key] = None
    try:
        response = supabase.from_('app_tracked_trades').select('id').eq('tsym', tsym).limit(1).execute()
        if response.data:
            trade_id = response.data[0]['id']
            data_to_update = {k: v for k, v in trade_data.items() if k not in ['id', 'created_at']}
            updated_response = supabase.from_('app_tracked_trades').update(data_to_update).eq('id', trade_id).execute()
            if updated_response.data:
                logging.info(f"Updated trade {tsym} in Supabase: {updated_response.data}")
                return True
            else:
                logging.error(f"Failed to update trade {tsym} in Supabase: {updated_response.status_code} - {updated_response.get('error', 'No error message')}")
                return False
        else:
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
        today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        response = supabase.from_('app_tracked_trades').select('*').gt('created_at', today_start).execute()
        if response.data:
            logging.info(f"Loaded {len(response.data)} trades from Supabase for today.")
            for trade_record in response.data:
                tsym = trade_record['tsym']
                status = trade_record['status']
                for key in ['entry_price', 'sl_price', 'target_price', 'highest_price_seen',
                            'lowest_price_seen', 'signal_candle_high', 'signal_candle_low',
                            'manual_sl_price', 'manual_target_price']:
                    if key in trade_record and trade_record[key] is not None:
                        try:
                            trade_record[key] = float(trade_record[key])
                        except (ValueError, TypeError):
                            trade_record[key] = None
                if status == 'PENDING':
                    st.session_state.pending_entries[tsym] = {
                        'buy_or_sell': trade_record['buy_or_sell'],
                        'signal_candle_high': trade_record['signal_candle_high'],
                        'signal_candle_low': trade_record['signal_candle_low'],
                        'calculated_quantity': trade_record['quantity'],
                        'initial_sl_price': trade_record['sl_price'],
                        'initial_tp_price': trade_record['target_price'],
                        'status': 'PENDING',
                        'token': trade_record['token'],
                        'current_ltp': None
                    }
                    if trade_record['manual_sl_price'] is not None or trade_record['manual_target_price'] is not None:
                         st.session_state.manual_overrides.setdefault(tsym, {})['sl_price'] = trade_record['manual_sl_price']
                         st.session_state.manual_overrides.setdefault(tsym, {})['target_price'] = trade_record['manual_target_price']
                elif status == 'OPEN':
                    st.session_state.open_tracked_trades[tsym] = {
                        'order_no': trade_record.get('order_no'),
                        'entry_price': trade_record['entry_price'],
                        'quantity': trade_record['quantity'],
                        'sl_price': trade_record['sl_price'],
                        'target_price': trade_record['target_price'],
                        'buy_or_sell': trade_record['buy_or_sell'],
                        'status': 'OPEN',
                        'token': trade_record['token'],
                        'highest_price_seen': trade_record['highest_price_seen'],
                        'lowest_price_seen': trade_record['lowest_price_seen'],
                        'current_ltp': None
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

if not st.session_state.supabase_loaded:
    if load_tracked_trades_from_supabase():
        st.session_state.supabase_loaded = True
    else:
        st.session_state.supabase_loaded = True

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
        logging.critical(f"An exception occurred during API session setup: {e}", exc_info=True)
        return None

api = get_api_instance(USER_ID, USER_SESSION)

if api is None:
    st.stop()

@st.cache_data
def load_symbols_from_csv(file_path="NSE_Equity.csv"):
    """Loads stock symbols and tokens from the provided CSV file."""
    try:
        df = pd.read_csv(file_path)
        if all(col in df.columns for col in ['Exchange', 'Token', 'Tradingsymbol', 'Instrument']):
            equity_symbols = df[df['Instrument'] == 'EQ'][['Exchange', 'Token', 'Tradingsymbol']].copy()
            symbols_map = {row['Tradingsymbol']: str(row['Token']) for index, row in equity_symbols.iterrows()}
            st.success(f"Loaded {len(symbols_map)} equity symbols from {file_path}.")
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

def fetch_and_update_ltp():
    """Fetches the live LTP for all pending and open trades and updates the session state."""
    all_tracked_symbols = set(list(st.session_state.pending_entries.keys()) + list(st.session_state.open_tracked_trades.keys()))
    if not all_tracked_symbols:
        logging.info("No tracked symbols to update LTP.")
        return
    logging.info(f"Fetching LTP for {len(all_tracked_symbols)} symbols: {all_tracked_symbols}")
    for tsym in all_tracked_symbols:
        if tsym in st.session_state.pending_entries:
            trade_data = st.session_state.pending_entries[tsym]
            current_ltp = get_tradetron_ltp_simple(
                tsym.replace('-EQ', ''),
                st.session_state.tradetron_cookie,
                st.session_state.tradetron_user_agent
            )
            if isinstance(current_ltp, (int, float)):
                trade_data['current_ltp'] = current_ltp
                logging.info(f"Updated LTP for PENDING trade {tsym} to {current_ltp}")
            else:
                logging.warning(f"Failed to get LTP for PENDING trade {tsym}: {current_ltp}")
        elif tsym in st.session_state.open_tracked_trades:
            trade_data = st.session_state.open_tracked_trades[tsym]
            current_ltp = get_tradetron_ltp_simple(
                tsym.replace('-EQ', ''),
                st.session_state.tradetron_cookie,
                st.session_state.tradetron_user_agent
            )
            if isinstance(current_ltp, (int, float)):
                trade_data['current_ltp'] = current_ltp
                logging.info(f"Updated LTP for OPEN trade {tsym} to {current_ltp}")
            else:
                logging.warning(f"Failed to get LTP for OPEN trade {tsym}: {current_ltp}")

def get_nifty500_symbols():
    """Uses the load_symbols_from_csv function to get the actual symbols."""
    return load_symbols_from_csv()

def get_tradetron_ltp(symbol, tradetron_cookie, tradetron_user_agent):
    """Fetches the Last Traded Price (LTP) for a given symbol using the Tradetron API."""
    if not tradetron_cookie or not tradetron_user_agent:
        logging.warning(f"Tradetron cookie or user-agent not provided for {symbol}. Cannot fetch LTP.")
        return "Auth Missing"
    clean_symbol = symbol.replace('-EQ', '').strip()
    current_time = datetime.datetime.now()
    etime_ms = int(current_time.timestamp() * 1000)
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
        print(f"DEBUG - Symbol: {symbol}, Clean Symbol: {clean_symbol}")
        print(f"DEBUG - Full API Response: {data}")
        print(f"DEBUG - Response Type: {type(data)}")
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

def get_current_manual_sl(tsym, trade_info=None):
    """Safely get current manual SL with fallbacks"""
    try:
        manual_sl = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
        if manual_sl is not None and manual_sl > 0:
            return float(manual_sl)
        if trade_info and 'sl_price' in trade_info:
            trade_sl = trade_info.get('sl_price')
            if trade_sl is not None and trade_sl > 0:
                return float(trade_sl)
        return 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0

def get_tradetron_ltp_simple(symbol, tradetron_cookie, tradetron_user_agent):
    """ Simplified version for production use after debugging """
    if not tradetron_cookie or not tradetron_user_agent:
        return "Auth Missing"
    current_time = datetime.datetime.now()
    time_ranges = [15, 30, 60, 120]
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
                if (data and data.get('success') is True and 'Data' in data and isinstance(data['Data'], list) and len(data['Data']) > 0 and isinstance(data['Data'][0], dict) and 'close' in data['Data'][0]):
                    return float(data['Data'][0]['close'])
        except Exception as e:
            logging.error(f"Error fetching Tradetron data for {symbol} with {minutes}min range: {e}")
            continue
    return "No Data"

if __name__ == "__main__":
    st.title("Trade Monitor Dashboard")
    with st.sidebar:
        st.header("Tradetron API Credentials")
        st.session_state.tradetron_cookie = st.text_input("Tradetron Cookie", type="password")
        st.session_state.tradetron_user_agent = st.text_input("User-Agent", type="password")
        if st.button("🔄 Refresh LTP (Tradetron)", type="secondary"):
            fetch_and_update_ltp()
            st.success("LTP data refreshed from Tradetron API!")

    # Main dashboard view
    st.subheader("📊 Live Positions & P&L")
    st.markdown("---")

    # The main trade logic loop
    if st.button("Run Trade Logic"):
        # Reset daily flags for the strategy if a new day has started
        st.session_state.strategy.reset_daily_flags(datetime.date.today())

        # Placeholder for data fetching - replace with your actual data fetching function
        # Example: Fetch 1-minute candle data for a set of symbols
        # For demonstration, we will use mock data
        symbol_token_map = get_nifty500_symbols()
        symbols_to_screen = list(symbol_token_map.keys())[:10] # Screen first 10 for demo

        for tsym in symbols_to_screen:
            # Mock data fetch. In a real app, this would be a live API call
            mock_data = {
                'Open': [100, 105],
                'High': [110, 112],
                'Low': [98, 103],
                'Close': [105, 108],
                'Volume': [1000, 1200]
            }
            mock_df = pd.DataFrame(mock_data)

            # Use the strategy to get signals
            buy_signal, sell_signal = st.session_state.strategy.get_signals(mock_df)
            
            # Check for entry conditions
            if buy_signal and st.session_state.strategy.current_position is None:
                st.session_state.strategy.current_position = 'long'
                st.session_state.strategy.entry_price = mock_df.iloc[-1]['Close']
                st.info(f"Buy Signal for {tsym} at {st.session_state.strategy.entry_price}")
            
            if sell_signal and st.session_state.strategy.current_position is None:
                st.session_state.strategy.current_position = 'short'
                st.session_state.strategy.entry_price = mock_df.iloc[-1]['Close']
                st.info(f"Sell Signal for {tsym} at {st.session_state.strategy.entry_price}")

            # Check for exit conditions
            if st.session_state.strategy.current_position:
                current_ltp = mock_df.iloc[-1]['Close'] # Using mock data, replace with live LTP
                if st.session_state.strategy.check_exit_conditions(current_ltp):
                    st.success(f"Exit Signal for {tsym} at {current_ltp}")
                    st.session_state.strategy.current_position = None
                    st.session_state.strategy.entry_price = None

    display_trades_with_unique_keys()
