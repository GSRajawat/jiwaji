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
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "87157f5ad3c28cefab77351b4bdbc49644b53a91de70fdba30497c2abccb1a43")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL","https://zybakxpyibubzjhzdcwl.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFrc3B5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU")

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 21 # Latest candle + previous 20 for calculations

# --- Entry/Exit Buffers and Risk Parameters (Constants) ---
ENTRY_BUFFER_PERCENT = 0.0005 # 0.05% buffer for crossing high/low for entry
RISK_PERCENTAGE_OF_CAPITAL = 0.01 # 1% of capital risked per trade


# --- Initialize ALL session state variables first and foremost ---
if 'volume_multiplier' not in st.session_state:
    st.session_state.volume_multiplier = 10
if 'traded_value_threshold' not in st.session_state:
    st.session_state.traded_value_threshold = 10000000
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
    Filters for 'EQ' (Equity) instruments only. Assumes 'Token' and 'Tradingsymbol' are present.
    Returns a dictionary mapping tradingsymbol (tsym) to its token.
    """
    try:
        df = pd.read_csv(file_path)
        if all(col in df.columns for col in ['Exchange', 'Token', 'Tradingsymbol', 'Instrument']):
            equity_symbols = df[df['Instrument'] == 'EQ'][['Exchange', 'Token', 'Tradingsymbol']].copy()
            
            # Create a dictionary mapping tsym to token
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
                    st.write("Please get a fresh cookie from your browser")
                    
                elif response.status_code == 403:
                    st.error("❌ 403 Forbidden - Access denied")
                    
                else:
                    st.error(f"❌ HTTP {response.status_code}")
                    st.write(f"Response: {response.text[:200]}...")
                    
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out")
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection error")
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
    tradetron_cookie = st.session_state.get('tradetron_cookie')
    tradetron_user_agent = st.session_state.get('tradetron_user_agent')

    if not tradetron_cookie or not tradetron_user_agent:
        st.warning("Tradetron API credentials (cookie and user-agent) are missing. Please enter them in the sidebar to get live LTP updates.")
        return

    # Create a unified list of all symbols to check
    all_tracked_symbols = set(list(st.session_state.pending_entries.keys()) + list(st.session_state.open_tracked_trades.keys()))

    if not all_tracked_symbols:
        logging.info("No tracked symbols to update LTP.")
        return

    logging.info(f"Fetching LTP for {len(all_tracked_symbols)} symbols: {all_tracked_symbols}")

    for tsym in all_tracked_symbols:
        # Clean up symbol format for Tradetron API
        clean_symbol = tsym.replace('-EQ', '').strip()
        
        # Use the same function that works for Market Watch
        ltp = get_tradetron_ltp(clean_symbol, tradetron_cookie, tradetron_user_agent)
        
        # Check if the symbol is in pending entries
        if tsym in st.session_state.pending_entries:
            st.session_state.pending_entries[tsym]['current_ltp'] = ltp
            logging.info(f"Updated LTP for PENDING trade {tsym} to {ltp}")

        # Check if the symbol is in open tracked trades
        if tsym in st.session_state.open_tracked_trades:
            st.session_state.open_tracked_trades[tsym]['current_ltp'] = ltp
            logging.info(f"Updated LTP for OPEN trade {tsym} to {ltp}")

def calculate_pvi_and_ema(candle_data, period=180):
    """
    Calculate Positive Volume Index (PVI) and its 180-period EMA
    PVI increases when volume is higher than previous period, stays same when lower
    """
    if len(candle_data) < period + 1:
        return None, None
    
    # Calculate PVI
    pvi_values = [100]  # Start PVI at 100
    
    for i in range(1, len(candle_data)):
        current_volume = float(candle_data[i].get('intv', 0))
        previous_volume = float(candle_data[i-1].get('intv', 0))
        current_close = float(candle_data[i].get('intc', 0))
        previous_close = float(candle_data[i-1].get('intc', 0))
        
        if current_volume > previous_volume and previous_close > 0:
            # Volume increased, adjust PVI
            price_change_ratio = (current_close - previous_close) / previous_close
            new_pvi = pvi_values[-1] * (1 + price_change_ratio)
        else:
            # Volume decreased or same, PVI stays same
            new_pvi = pvi_values[-1]
        
        pvi_values.append(new_pvi)
    
    # Calculate 180-period EMA of PVI
    if len(pvi_values) >= period:
        # Simple moving average for first EMA value
        ema_multiplier = 2 / (period + 1)
        ema_values = [sum(pvi_values[:period]) / period]
        
        # Calculate EMA for remaining values
        for i in range(period, len(pvi_values)):
            ema_value = (pvi_values[i] * ema_multiplier) + (ema_values[-1] * (1 - ema_multiplier))
            ema_values.append(ema_value)
        
        return pvi_values[-1], ema_values[-1]  # Return latest PVI and its EMA
    
    return None, None

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
    
    # Calculate potential loss (this will be used for trailing SL)
    potential_loss = sl_points  # 1x potential loss = SL distance
    
    return calculated_quantity, sl_points, potential_loss


def place_trade_with_new_logic(api, tsym, buy_or_sell, entry_price, sl_price, target_price, capital, token):
    """
    Place a trade using the new quantity calculation and store potential loss for trailing SL.
    
    Args:
        api: Trading API instance
        tsym (str): Trading symbol
        buy_or_sell (str): 'B' for buy, 'S' for sell
        entry_price (float): Entry price
        sl_price (float): Stop loss price
        target_price (float): Target price
        capital (float): Available capital
        token (str): Trading token
    
    Returns:
        dict: Order response
    """
    # Calculate quantity using new formula
    quantity, sl_points, potential_loss = calculate_quantity_and_sl_new(entry_price, sl_price, capital) # Use the new function
    
    if quantity <= 0:
        st.error(f"Invalid quantity calculated: {quantity}. Check SL distance.")
        return None
    
    st.info(f"Calculated Quantity: {quantity} (Capital: {capital}, SL Points: {sl_points:.2f}, Potential Loss: {potential_loss:.2f})")
    
    # Place the order
    order_response = api.place_order(
        buy_or_sell=buy_or_sell,
        product_type='I',  # Intraday
        exchange=EXCHANGE,
        tradingsymbol=tsym,
        quantity=int(quantity),
        discloseqty=0,
        price_type='MKT',  # Market order
        price=0,
        trigger_price=None,
        retention='DAY',
        remarks=f'Auto_Entry_{buy_or_sell}'
    )
    
    if order_response and order_response.get('stat') == 'Ok':
        # Store trade info with potential loss for trailing SL
        trade_info = {
            'tsym': tsym,
            'buy_or_sell': buy_or_sell,
            'quantity': quantity,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'target_price': target_price,
            'potential_loss': potential_loss,  # NEW: Store for trailing SL calculation
            'sl_points': sl_points,
            'status': 'OPEN',
            'order_id': order_response.get('norenordno'),
            'current_ltp': entry_price
        }
        
        # Initialize highest/lowest price seen for trailing
        if buy_or_sell == 'B':
            trade_info['highest_price_seen'] = entry_price
        else:
            trade_info['lowest_price_seen'] = entry_price
        
        # Store in session state and database
        st.session_state.open_tracked_trades[tsym] = trade_info
        upsert_trade_to_supabase(trade_info)
        
        st.success(f"Order placed successfully: {order_response}")
        return order_response
    else:
        st.error(f"Order failed: {order_response.get('emsg', 'Unknown error')}")
        return None

if st.sidebar.button("🔄 Refresh LTP (Tradetron)", type="secondary"):
    # This is a good place to call a function that updates LTP for all tracked trades
    fetch_and_update_ltp()
    st.success("LTP data refreshed from Tradetron API!")

def screen_stock(stock_info, api, all_symbols_map):
    """
    Modified screening function with new conditions:
    1. Signal candle timestamp display
    2. Open-close range instead of high-low range
    3. PVI condition with 180 EMA
    4. Check if symbol already traded today
    """
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']
    
    # Check if already traded today
    if tradingsymbol in st.session_state.daily_traded_symbols:
        return tradingsymbol, 'NEUTRAL', 'Already traded today', None, None, None, None
    
    # Calculate start time for fetching candles (need more for PVI calculation)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=220)  # Extra candles for PVI calculation
    
    try:
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=int(start_time.timestamp()),
            endtime=int(end_time.timestamp()),
            interval=CANDLE_INTERVAL
        )
        
        if not candle_data or len(candle_data) < 200:  # Need more candles for PVI
            logging.warning(f"Not enough candle data for {tradingsymbol}. Needed: 200+, Got: {len(candle_data) if candle_data else 0}")
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None, None, None
        
        current_candle = candle_data[0]  # Most recent candle
        previous_20_candles = candle_data[1:21]  # Previous 20 candles for average calculation
        
        # Extract values from current candle
        current_volume = float(current_candle.get('intv', 0))
        current_close_price = float(current_candle.get('intc', 0))
        current_open_price = float(current_candle.get('into', 0))
        current_high = float(current_candle.get('inth', 0))
        current_low = float(current_candle.get('intl', 0))
        
        # Get signal candle timestamp
        signal_candle_time = current_candle.get('time', 'Unknown')
        if isinstance(signal_candle_time, str) and signal_candle_time.isdigit():
            signal_candle_time = datetime.datetime.fromtimestamp(int(signal_candle_time)).strftime('%H:%M:%S')
        
        if current_volume == 0 or current_close_price == 0 or current_high == 0 or current_low == 0:
            return tradingsymbol, 'NEUTRAL', 'Current candle data is zero/invalid', None, None, None, signal_candle_time

        # --- 1. Volume Check ---
        previous_volumes = [float(c.get('intv', 0)) for c in previous_20_candles if float(c.get('intv', 0)) > 0]
        if not previous_volumes:
            return tradingsymbol, 'NEUTRAL', 'No valid volume data in previous 20 candles', None, None, None, signal_candle_time
        
        average_volume_last_20 = sum(previous_volumes) / len(previous_volumes)
        if not (current_volume > st.session_state.volume_multiplier * average_volume_last_20):
            return tradingsymbol, 'NEUTRAL', 'Volume condition not met', None, None, None, signal_candle_time

        # --- 2. Traded Value Check ---
        current_traded_value = current_volume * current_close_price
        if not (current_traded_value > st.session_state.traded_value_threshold):
            return tradingsymbol, 'NEUTRAL', 'Traded value condition not met', None, None, None, signal_candle_time

        # --- 3. Open-Close Range Check (Modified) ---
        current_open_close_diff = abs(current_close_price - current_open_price)
        if current_open_close_diff <= 0:
            return tradingsymbol, 'NEUTRAL', 'Current open-close difference invalid', None, None, None, signal_candle_time
        
        previous_open_close_diffs = []
        for c in previous_20_candles:
            open_price = float(c.get('into', 0))
            close_price = float(c.get('intc', 0))
            if open_price > 0 and close_price > 0:
                diff = abs(close_price - open_price)
                if diff > 0:
                    previous_open_close_diffs.append(diff)
        
        if not previous_open_close_diffs:
            return tradingsymbol, 'NEUTRAL', 'No valid open-close diff data in previous 20 candles', None, None, None, signal_candle_time

        average_open_close_diff_last_20 = sum(previous_open_close_diffs) / len(previous_open_close_diffs)
        if not (current_open_close_diff > st.session_state.high_low_diff_multiplier * average_open_close_diff_last_20):
            return tradingsymbol, 'NEUTRAL', 'Open-close diff condition not met', None, None, None, signal_candle_time

        # --- 4. PVI Condition Check ---
        current_pvi, pvi_ema_180 = calculate_pvi_and_ema(candle_data, period=180)
        if current_pvi is None or pvi_ema_180 is None:
            return tradingsymbol, 'NEUTRAL', 'Insufficient data for PVI calculation', None, None, None, signal_candle_time

        # --- 5. Candle Color Check with PVI condition ---
        if current_close_price > current_open_price:
            # Green candle - check if PVI > PVI EMA for BUY
            if current_pvi > pvi_ema_180:
                return tradingsymbol, 'BUY', 'All conditions met: Green candle + PVI above EMA', current_close_price, current_high, current_low, signal_candle_time
            else:
                return tradingsymbol, 'NEUTRAL', 'Green candle but PVI below EMA', None, None, None, signal_candle_time
        elif current_close_price < current_open_price:
            # Red candle - check if PVI < PVI EMA for SELL
            if current_pvi < pvi_ema_180:
                return tradingsymbol, 'SELL', 'All conditions met: Red candle + PVI below EMA', current_close_price, current_high, current_low, signal_candle_time
            else:
                return tradingsymbol, 'NEUTRAL', 'Red candle but PVI above EMA', None, None, None, signal_candle_time
        else:
            return tradingsymbol, 'NEUTRAL', 'Current candle is Doji (Open == Close)', None, None, None, signal_candle_time

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}", exc_info=True)
        return tradingsymbol, 'NEUTRAL', f'Error during screening: {e}', None, None, None, None
def calculate_quantity_and_sl(signal_type, signal_candle_high, signal_candle_low, expected_entry_price, capital):
    """
    Modified quantity calculation: capital * 0.01 / SL_points
    SL adjustment: 1x potential loss instead of fixed points
    """
    if signal_type == 'BUY':
        # SL at signal candle low minus buffer
        sl_price = round(signal_candle_low - st.session_state.sl_buffer_points, 2)
        sl_points = expected_entry_price - sl_price
        
        # Trailing step = 1x potential loss
        trailing_step = sl_points
        
    else:  # SELL
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
    buy_or_sell,
    tradingsymbol,
    quantity,
    entry_price,
    api,
    token
):
    """
    Places a regular Intraday (MIS) order.
    Returns: order_response dictionary
    """
    if quantity <= 0:
        st.warning(f"Cannot place order for {tradingsymbol}: Calculated quantity is zero or negative ({quantity}).")
        logging.warning(f"Order skipped for {tradingsymbol}: Quantity is zero or negative ({quantity}).")
        return {'stat': 'Not_Ok', 'emsg': 'Quantity is zero or negative'}

    # --- NEW: Check for existing open positions before placing order ---
    try:
        broker_positions = api.get_positions()
        if isinstance(broker_positions, list):
            for pos in broker_positions:
                if pos.get('tsym') == tradingsymbol and pos.get('netqty', 0) != 0:
                    st.warning(f"Order skipped for {tradingsymbol}: An existing open position with Net Qty {pos.get('netqty')} is already found in your broker account. To avoid duplicate orders, this new order will not be placed.")
                    logging.warning(f"Order skipped for {tradingsymbol}: Duplicate position detected in broker account.")
                    return {'stat': 'Not_Ok', 'emsg': 'Duplicate position in broker account'}
    except Exception as e:
        st.error(f"Error checking broker positions for {tradingsymbol}: {e}. Proceeding with order placement, but this might lead to duplicates if positions exist.")
        logging.error(f"Error checking broker positions before placing order for {tradingsymbol}: {e}", exc_info=True)
    # --- END NEW CHECK ---


    st.info(f"Attempting to place {buy_or_sell} Intraday order for {tradingsymbol}: Qty={int(quantity)}, Price={entry_price:.2f}")
    logging.info(f"Placing order for {tradingsymbol}. Action: {buy_or_sell}, Qty: {quantity}, Entry Ref Price: {entry_price:.2f}, Product: I, Exchange: {EXCHANGE}")
    
    try:
        order_response = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type='I', # 'I' for Intraday (MIS)
            exchange=EXCHANGE,
            tradingsymbol=tradingsymbol,
            quantity=int(quantity),
            discloseqty=0,
            price_type='MKT', # Use Market order for entry once triggered
            price=0, # Not relevant for MKT order
            trigger_price=None,
            retention='DAY',
            remarks='Automated_Screener_Trade_Manual_SL_TP'
        )
        
        if order_response and order_response.get('stat') == 'Ok':
            st.success(f"Order placed successfully for {tradingsymbol}. Order ID: {order_response.get('norenordno')}")
            logging.info(f"Order placed successfully for {tradingsymbol}: {order_response}")
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
    st.info(f"Attempting to exit position for {tradingsymbol} (Qty: {netqty}, Product: {product_type})...")
    try:
        response = api.exit_order(
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            product_type=product_type,
            quantity=abs(int(netqty)) # Ensure positive quantity for exit
        )
        if response and response.get('stat') == 'Ok':
            st.success(f"Position for {tradingsymbol} exited successfully: {response}")
            logging.info(f"Position for {tradingsymbol} exited successfully: {response}")
            
            # Mark the tracked trade as closed and delete from Supabase
            if tradingsymbol in st.session_state.open_tracked_trades:
                st.session_state.open_tracked_trades[tradingsymbol]['status'] = 'CLOSED'
                delete_trade_from_supabase(tsym=tradingsymbol) # Delete from DB

        else:
            error_msg = response.get('emsg', 'Unknown error') if isinstance(response, dict) else str(response)
            st.error(f"Failed to exit position for {tradingsymbol}: {error_msg}")
            logging.error(f"Failed to exit position for {tradingsymbol}: {error_msg}. Full response: {response}")
            return response
            
    except Exception as e:
        st.error(f"An error occurred while exiting position for {tradingsymbol}: {e}")
        logging.error(f"An error occurred while exiting position for {tradingsymbol}: {e}", exc_info=True)
        return {'stat': 'Not_Ok', 'emsg': str(e)}


def monitor_open_trades(api, all_symbols_map):
    """
    Monitors actively tracked trades for stoploss/target conditions and applies trailing stop-loss.
    Uses Tradetron API for live quotes and triggers closing orders if conditions are met.
    
    Key Changes:
    - Trailing SL based on 1x potential loss amount instead of fixed points
    - Dynamic calculation: BUY SL = highest_price - potential_loss, SELL SL = lowest_price + potential_loss
    
    Returns:
        list: A list of trades that need to be closed.
    """
    trades_to_close = []
    
    tracked_tsyms_for_quotes = []
    for tsym, trade_info in st.session_state.open_tracked_trades.items():
        if trade_info['status'].startswith('OPEN') or trade_info['status'].startswith('CLOSING'):
            tracked_tsyms_for_quotes.append(tsym)

    if not tracked_tsyms_for_quotes:
        logging.debug("No open trades to monitor.")
        return [] # Return empty list

    # Get Tradetron credentials
    tradetron_cookie = st.session_state.get('tradetron_cookie')
    tradetron_user_agent = st.session_state.get('tradetron_user_agent')

    if not tradetron_cookie or not tradetron_user_agent:
        st.warning("Tradetron credentials missing. Cannot fetch live prices for monitoring.")
        return []

    for tsym in tracked_tsyms_for_quotes:
        trade_info = st.session_state.open_tracked_trades[tsym]
        token = all_symbols_map.get(tsym) # Get token from the map for order placement
        
        if not token:
            logging.warning(f"Token not found for {tsym}. Skipping monitoring for this trade.")
            trade_info['current_ltp'] = "Token Missing"
            continue

        logging.debug(f"Fetching Tradetron LTP for open trade {tsym}")
        try:
            # Clean up symbol format for Tradetron
            clean_symbol = tsym.replace('-EQ', '').strip()
            
            # Fetch LTP using Tradetron API
            current_ltp = get_tradetron_ltp(clean_symbol, tradetron_cookie, tradetron_user_agent)
            
            # Check if we got a valid numeric LTP
            if isinstance(current_ltp, (int, float)):
                trade_info['current_ltp'] = current_ltp # Store LTP for display
                
                buy_or_sell = trade_info['buy_or_sell']
                quantity = trade_info['quantity']

                # Get potential loss from trade data (calculated during order placement)
                potential_loss = trade_info.get('potential_loss', 0)
                if potential_loss <= 0:
                    logging.warning(f"No potential loss found for {tsym}. Cannot apply trailing SL.")
                    continue

                # --- Determine effective SL and TP (manual override or calculated/trailing) ---
                manual_sl_override = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
                manual_tp_override = st.session_state.manual_overrides.get(tsym, {}).get('target_price')

                effective_sl_price = manual_sl_override if manual_sl_override is not None and manual_sl_override > 0 else trade_info['sl_price']
                effective_target_price = manual_tp_override if manual_tp_override is not None and manual_tp_override > 0 else trade_info['target_price']

                # Apply trailing stop only if NO manual SL override is active
                if manual_sl_override is None or manual_sl_override <= 0:
                    if buy_or_sell == 'B': # Long position
                        if 'highest_price_seen' not in trade_info or current_ltp > trade_info['highest_price_seen']:
                            trade_info['highest_price_seen'] = current_ltp
                            upsert_trade_to_supabase({
                                'tsym': tsym,
                                'highest_price_seen': trade_info['highest_price_seen']
                            })
                        
                        # NEW: Trailing SL = highest_price_seen - potential_loss (instead of fixed points)
                        new_potential_sl = round(trade_info['highest_price_seen'] - potential_loss, 2)
                        if effective_sl_price is None or new_potential_sl > effective_sl_price: # Only move SL up
                            trade_info['sl_price'] = new_potential_sl # Update internal TSL
                            effective_sl_price = new_potential_sl # Use updated TSL for monitoring
                            st.info(f"Trailing SL for BUY {tsym} updated to {effective_sl_price:.2f} (Highest: {trade_info['highest_price_seen']:.2f} - Loss: {potential_loss:.2f})")
                            upsert_trade_to_supabase({'tsym': tsym, 'sl_price': effective_sl_price})

                    elif buy_or_sell == 'S': # Short position
                        if 'lowest_price_seen' not in trade_info or current_ltp < trade_info['lowest_price_seen']:
                            trade_info['lowest_price_seen'] = current_ltp
                            upsert_trade_to_supabase({
                                'tsym': tsym,
                                'lowest_price_seen': trade_info['lowest_price_seen']
                            })

                        # NEW: Trailing SL = lowest_price_seen + potential_loss (instead of fixed points)
                        new_potential_sl = round(trade_info['lowest_price_seen'] + potential_loss, 2)
                        if effective_sl_price is None or new_potential_sl < effective_sl_price: # Only move SL down
                            trade_info['sl_price'] = new_potential_sl # Update internal TSL
                            effective_sl_price = new_potential_sl # Use updated TSL for monitoring
                            st.info(f"Trailing SL for SELL {tsym} updated to {effective_sl_price:.2f} (Lowest: {trade_info['lowest_price_seen']:.2f} + Loss: {potential_loss:.2f})")
                            upsert_trade_to_supabase({'tsym': tsym, 'sl_price': effective_sl_price})
                
                st.markdown(f"**Monitoring {tsym}:** LTP={current_ltp:.2f}, SL={effective_sl_price if effective_sl_price is not None else 'N/A':.2f}, Target={effective_target_price if effective_target_price is not None else 'N/A':.2f}, PotLoss={potential_loss:.2f}")

                # Check for SL or Target hit with effective prices
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
                        st.success(f"Target HIT for SELL {tsym}! LTP {current_ltp} <= Target {effective_target_price}")
                        trades_to_close.append({'tsym': tsym, 'quantity': quantity, 'action': 'BUY', 'token': token, 'status_reason': 'CLOSING_TP'})
                        trade_info['status'] = 'CLOSING_TP' 
            else:
                trade_info['current_ltp'] = current_ltp  # Store the error message
                logging.warning(f"Failed to get Tradetron LTP for {tsym}: {current_ltp}")

        except Exception as e:
            trade_info['current_ltp'] = "Error"
            logging.error(f"Error monitoring open trade {tsym}: {e}", exc_info=True)
    
    return trades_to_close


def monitor_pending_entries(api, all_symbols_map):
    """
    Monitors pending entries for their entry conditions (crossing signal candle high/low).
    Uses Tradetron API for live quotes and places initial orders if conditions are met.
    
    Returns:
        list: A list of entries that need to be executed and moved to open trades.
    """
    entries_to_execute = []
    
    pending_tsyms_for_quotes = []
    for tsym, entry_info in st.session_state.pending_entries.items():
        if entry_info['status'] == 'PENDING':
            pending_tsyms_for_quotes.append(tsym)

    if not pending_tsyms_for_quotes:
        logging.debug("No pending entries to monitor.")
        return []

    # Get Tradetron credentials
    tradetron_cookie = st.session_state.get('tradetron_cookie')
    tradetron_user_agent = st.session_state.get('tradetron_user_agent')

    if not tradetron_cookie or not tradetron_user_agent:
        st.warning("Tradetron credentials missing. Cannot fetch live prices for pending entries.")
        return []

    for tsym in pending_tsyms_for_quotes:
        entry_info = st.session_state.pending_entries[tsym]
        token = all_symbols_map.get(tsym) # Get token from the map for order placement
        
        if not token:
            logging.warning(f"Token not found for {tsym}. Skipping monitoring for this pending entry.")
            entry_info['current_ltp'] = "Token Missing"
            continue

        logging.debug(f"Fetching Tradetron LTP for pending entry {tsym}")
        try:
            # Clean up symbol format for Tradetron
            clean_symbol = tsym.replace('-EQ', '').strip()
            
            # Fetch LTP using Tradetron API
            current_ltp = get_tradetron_ltp(clean_symbol, tradetron_cookie, tradetron_user_agent)
            
            # Check if we got a valid numeric LTP
            if isinstance(current_ltp, (int, float)):
                entry_info['current_ltp'] = current_ltp # Store LTP for display
                
                signal_candle_high = entry_info['signal_candle_high']
                signal_candle_low = entry_info['signal_candle_low']
                buy_or_sell = entry_info['buy_or_sell']
                quantity = entry_info['calculated_quantity']
                
                # --- Determine effective SL and TP (manual override or calculated) for pending entry ---
                manual_sl_override = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
                manual_tp_override = st.session_state.manual_overrides.get(tsym, {}).get('target_price')

                effective_sl_price = manual_sl_override if manual_sl_override is not None and manual_sl_override > 0 else entry_info['initial_sl_price']
                effective_target_price = manual_tp_override if manual_tp_override is not None and manual_tp_override > 0 else entry_info['initial_tp_price']

                st.markdown(f"**Pending {tsym} ({buy_or_sell}):** LTP={current_ltp:.2f}, Signal High={signal_candle_high:.2f}, Signal Low={signal_candle_low:.2f}")

                if buy_or_sell == 'B': # BUY entry
                    trigger_price = round(signal_candle_high * (1 + ENTRY_BUFFER_PERCENT), 2)
                    if current_ltp >= trigger_price:
                        st.success(f"BUY Entry Triggered for {tsym}! LTP {current_ltp} >= Trigger {trigger_price}")
                        entries_to_execute.append({
                            'tsym': tsym,
                            'quantity': quantity,
                            'action': 'B',
                            'entry_price': current_ltp, # Use current LTP as actual entry price
                            'sl_price': effective_sl_price, # Use effective SL for execution
                            'target_price': effective_target_price, # Use effective TP for execution
                            'token': token,
                            'signal_candle_high': signal_candle_high,  # Preserve signal data
                            'signal_candle_low': signal_candle_low     # Preserve signal data
                        })
                        entry_info['status'] = 'EXECUTING_BUY' # Mark for execution
                elif buy_or_sell == 'S': # SELL entry
                    trigger_price = round(signal_candle_low * (1 - ENTRY_BUFFER_PERCENT), 2)
                    if current_ltp <= trigger_price:
                        st.success(f"SELL Entry Triggered for {tsym}! LTP {current_ltp} <= Trigger {trigger_price}")
                        entries_to_execute.append({
                            'tsym': tsym,
                            'quantity': quantity,
                            'action': 'S',
                            'entry_price': current_ltp, # Use current LTP as actual entry price
                            'sl_price': effective_sl_price, # Use effective SL for execution
                            'target_price': effective_target_price, # Use effective TP for execution
                            'token': token,
                            'signal_candle_high': signal_candle_high,  # Preserve signal data
                            'signal_candle_low': signal_candle_low     # Preserve signal data
                        })
                        entry_info['status'] = 'EXECUTING_SELL' # Mark for execution
            else:
                entry_info['current_ltp'] = current_ltp  # Store the error message
                logging.warning(f"Failed to get Tradetron LTP for {tsym}: {current_ltp}")

        except Exception as e:
            entry_info['current_ltp'] = "Error"
            logging.error(f"Error monitoring pending entry {tsym}: {e}", exc_info=True)
    
    return entries_to_execute


# Additional fix: Add this function to prevent duplicate screening of already tracked positions
def is_symbol_already_tracked(tsym):
    """
    Check if a symbol is already being tracked by the app (either PENDING or OPEN)
    Returns True if already tracked, False if available for new signals
    """
    # Check if it's in pending entries with PENDING status
    if tsym in st.session_state.pending_entries:
        if st.session_state.pending_entries[tsym]['status'] == 'PENDING':
            return True
    
    # Check if it's in open tracked trades with OPEN status
    if tsym in st.session_state.open_tracked_trades:
        if st.session_state.open_tracked_trades[tsym]['status'] == 'OPEN':
            return True
    
    return False


# Updated screening logic - use this in your main screening loop
def get_eligible_symbols_for_screening(all_symbols_map):
    """
    Returns only symbols that are not already being tracked by the app
    """
    eligible_for_screening = []
    
    for tsym, token in all_symbols_map.items():
        if not is_symbol_already_tracked(tsym):
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
        logging.error(f"Error during database cleanup: {e}")


# Call this in your app initialization (after loading from Supabase)
# cleanup_stale_database_entries()
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
    "Available Capital (INR) for Qty Calc",
    min_value=1000,
    max_value=10000,
    value=int(st.session_state.capital),
    step=1000
)

st.sidebar.markdown(f"**Risk Percentage per Trade:** {RISK_PERCENTAGE_OF_CAPITAL * 100:.2f}%")

st.session_state.sl_buffer_points = st.sidebar.number_input(
    "SL Buffer Points (below signal low for BUY / above signal high for SELL)",
    min_value=0.01, value=st.session_state.sl_buffer_points, step=0.01, format="%.2f"
)

st.session_state.trailing_step_points = st.sidebar.number_input(
    "Trailing Stop Step Points",
    min_value=0.01, value=st.session_state.trailing_step_points, step=0.01, format="%.2f"
)

st.session_state.target_multiplier = st.sidebar.number_input(
    "Target Multiplier (Potential Loss * Multiplier = Target Profit)",
    min_value=1, value=int(st.session_state.target_multiplier), step=1
)

calculated_risk_amount = st.session_state.capital * RISK_PERCENTAGE_OF_CAPITAL
st.sidebar.markdown(f"**Risk Amount per Trade:** ₹{calculated_risk_amount:,.2f}")

st.sidebar.markdown("---")
st.sidebar.subheader("Tradetron API Settings (Experimental)")
st.sidebar.info("Paste your active browser cookie and user-agent here. This will expire and needs manual updates for continued use.")
st.session_state.tradetron_cookie = st.sidebar.text_area(
    "Tradetron Cookie String", 
    value=st.session_state.tradetron_cookie, 
    height=150
)
st.session_state.tradetron_user_agent = st.sidebar.text_input(
    "Tradetron User-Agent Header", 
    value=st.session_state.tradetron_user_agent
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
                            logging.error(f"Could not convert 'prange' item 'cash' to float: {item['cash']}")
        
        if cash_margin is not None:
            st.success(f"**Available Cash Margin:** ₹{cash_margin:,.2f}")
        else:
            st.warning("Could not fetch account limits: 'cash' data not found in expected locations or invalid format.")
            logging.warning(f"Account limits response (final check failed): {limits}") 
    except Exception as e:
        st.error(f"Error fetching account limits: {e}")
        logging.error(f"Error fetching account limits: {e}", exc_info=True)

with refresh_btn_col:
    if st.button("Refresh Data", key="refresh_all_data"):
        st.rerun()

st.markdown("---") # Separator

st.header("Broker Open Positions 💼 (For manual management or verification)")
with st.container():
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
                st.dataframe(df_positions)
                st.markdown("---")
                st.markdown("### Manage / Adopt Broker Positions")

                for _, row in df_positions.iterrows():
                    tsym = row['Symbol']
                    net_qty = row['Net Qty']
                    product_type = row['Product Type']

                    # Check if this position is already tracked by the app
                    is_tracked = tsym in st.session_state.open_tracked_trades or tsym in st.session_state.pending_entries

                    if is_tracked:
                        st.info(f"**{tsym}** (Qty: {net_qty}, Product: {product_type}) - *Already managed by the app.*")
                    else:
                        st.warning(f"**{tsym}** (Qty: {net_qty}, Product: {product_type}) - Not currently managed by the app.")
                        
                        expand_adopt = st.expander(f"Adopt {tsym} into App Management", expanded=False)
                        with expand_adopt:
                            st.markdown(
                                """
                                **Warning:** By adopting this position, the app will start actively monitoring and managing its Stop Loss and Target Profit.
                                It might place exit orders on your behalf when conditions are met. Ensure you understand this before proceeding.
                                """
                            )
                            with st.form(key=f"adopt_form_{tsym}"):
                                # Infer buy_or_sell from net_qty
                                adopted_buy_or_sell = 'B' if net_qty > 0 else 'S'
                                st.write(f"Inferred Action: **{adopted_buy_or_sell}**")
                                
                                adopted_entry_price = st.number_input(
                                    f"Original Entry Price for {tsym}", 
                                    min_value=0.01, format="%.2f", key=f"adopt_entry_price_{tsym}"
                                )
                                adopted_signal_high = st.number_input(
                                    f"Signal Candle High for {tsym} (Optional, for auto SL/TP)",
                                    min_value=0.0, format="%.2f", key=f"adopt_signal_high_{tsym}"
                                )
                                adopted_signal_low = st.number_input(
                                    f"Signal Candle Low for {tsym} (Optional, for auto SL/TP)",
                                    min_value=0.0, format="%.2f", key=f"adopt_signal_low_{tsym}"
                                )
                                
                                adopt_submitted = st.form_submit_button(f"Confirm Adopt {tsym}")

                                if adopt_submitted:
                                    if adopted_entry_price <= 0:
                                        st.error("Please provide a valid original entry price to adopt this position.")
                                    else:
                                        current_token = None
                                        # Use the all_symbols_map to get the token
                                        current_token = get_nifty500_symbols().get(tsym) 

                                        if not current_token:
                                            st.error(f"Cannot adopt {tsym}: Token not found in loaded symbols. Please ensure it's in NSE_Equity.csv.")
                                        else:
                                            # Calculate initial SL/TP and quantity based on strategy, or use manual if provided
                                            calculated_sl = None
                                            calculated_tp = None
                                            calculated_qty = abs(net_qty) # Use broker's quantity

                                            if adopted_signal_high > 0 and adopted_signal_low > 0:
                                                # Use strategy to calculate SL/TP if signal candle info provided
                                                if adopted_buy_or_sell == 'B':
                                                    initial_sl_price = round(adopted_signal_low - st.session_state.sl_buffer_points, 2)
                                                    potential_loss_per_share = adopted_entry_price - initial_sl_price
                                                    if potential_loss_per_share <= 0.01: potential_loss_per_share = 0.01 # Prevent zero/negative loss
                                                    initial_tp_price = round(adopted_entry_price + (potential_loss_per_share * st.session_state.target_multiplier), 2)
                                                else: # SELL
                                                    initial_sl_price = round(adopted_signal_high + st.session_state.sl_buffer_points, 2)
                                                    potential_loss_per_share = initial_sl_price - adopted_entry_price
                                                    if potential_loss_per_share <= 0.01: potential_loss_per_share = 0.01
                                                    initial_tp_price = round(adopted_entry_price - (potential_loss_per_share * st.session_state.target_multiplier), 2)
                                                calculated_sl = initial_sl_price
                                                calculated_tp = initial_tp_price
                                            else:
                                                st.info("Signal Candle High/Low not provided. Please set Manual SL/Target in 'App-Tracked Trades' after adoption.")

                                            # Add to open_tracked_trades and Supabase
                                            new_tracked_trade = {
                                                'order_no': None, # No order_no for adopted positions
                                                'entry_price': adopted_entry_price,
                                                'quantity': calculated_qty,
                                                'sl_price': calculated_sl,
                                                'target_price': calculated_tp,
                                                'buy_or_sell': adopted_buy_or_sell,
                                                'status': 'OPEN',
                                                'token': current_token,
                                                'highest_price_seen': adopted_entry_price if adopted_buy_or_sell == 'B' else None,
                                                'lowest_price_seen': adopted_entry_price if adopted_buy_or_sell == 'S' else None,
                                                'signal_candle_high': adopted_signal_high if adopted_signal_high > 0 else None,
                                                'signal_candle_low': adopted_signal_low if adopted_signal_low > 0 else None,
                                                'current_ltp': None # Initialize current_ltp for adopted trades
                                            }
                                            st.session_state.open_tracked_trades[tsym] = new_tracked_trade
                                            
                                            supabase_payload = {
                                                'tsym': tsym,
                                                'exchange': EXCHANGE,
                                                'token': current_token,
                                                'buy_or_sell': adopted_buy_or_sell,
                                                'quantity': calculated_qty,
                                                'entry_price': adopted_entry_price,
                                                'sl_price': calculated_sl,
                                                'target_price': calculated_tp,
                                                'status': 'OPEN',
                                                'highest_price_seen': new_tracked_trade['highest_price_seen'],
                                                'lowest_price_seen': new_tracked_trade['lowest_price_seen'],
                                                'signal_candle_high': new_tracked_trade['signal_candle_high'],
                                                'signal_candle_low': new_tracked_trade['signal_candle_low']
                                            }
                                            if upsert_trade_to_supabase(supabase_payload):
                                                st.success(f"Position {tsym} successfully adopted and added to app tracking!")
                                                st.rerun() # Refresh to update tracked trades display
                                            else:
                                                st.error(f"Failed to save adopted position {tsym} to database.")
                                    
                    col1_pos, col2_pos = st.columns([0.7, 0.3])
                    # Always show manual exit for broker positions, even if adopted, for quick manual closure
                    col1_pos.write(f"**{row['Symbol']}** (Qty: {row['Net Qty']}, Product: {row['Product Type']})")
                    if col2_pos.button(f"Exit {row['Symbol']}", key=f"exit_broker_{row['Symbol']}_{row['Product Type']}_{row['Net Qty']}"):
                        current_token = get_nifty500_symbols().get(row['Symbol'])

                        if current_token:
                            exit_response = exit_position(
                                exchange=row['Exchange'],
                                tradingsymbol=row['Symbol'],
                                product_type=row['Product Type'],
                                netqty=row['Net Qty'],
                                api=api,
                                token=current_token
                            )
                            if exit_response and exit_response.get('stat') == 'Ok':
                                st.success(f"Exit order sent for {row['Symbol']}. Refreshing positions...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed to send exit order for {row['Symbol']}.")
                        else:
                            st.error(f"Cannot exit {row['Symbol']}: Token not found in loaded symbols.")
            else:
                st.info("No open positions found in broker account.")
        else:
            st.info("No open positions found in broker account or failed to retrieve positions.")
    except Exception as e:
        st.error(f"Error fetching positions from broker: {e}")
        logging.error(f"Error fetching positions from broker: {e}", exc_info=True)

st.markdown("---") # Separator

# --- Market Watch Section ---
st.header("Market Watch 📊")

all_symbols_map = get_nifty500_symbols() # Get the map of tsym to token
nifty500_symbols_names = sorted(list(all_symbols_map.keys()))

selected_market_watch_symbols = st.multiselect(
    "Select symbols for Market Watch",
    options=nifty500_symbols_names,
    default=st.session_state.market_watch_symbols
)

# Debug: Confirm selected symbols
st.info(f"**Market Watch Symbols Selected:** {selected_market_watch_symbols}")

if st.session_state.market_watch_symbols != selected_market_watch_symbols:
    st.session_state.market_watch_symbols = selected_market_watch_symbols
    st.rerun() # Rerun to update the market watch immediately

# NEW: Choose data source for Market Watch
market_watch_source_selection = st.radio(
    "Choose Market Watch Data Source:",
    ("Flattrade (NorenApiPy)", "Tradetron (Experimental)"),
    index=0 if st.session_state.market_watch_source == "Flattrade (NorenApiPy)" else 1,
    key="market_watch_source_radio"
)
if st.session_state.market_watch_source != market_watch_source_selection:
    st.session_state.market_watch_source = market_watch_source_selection
    st.rerun()


market_watch_placeholder = st.empty()

st.markdown("---") # Separator


st.header("Signals & App-Tracked Trades")

signal_placeholder = st.empty()
tracked_trades_placeholder = st.empty()
status_placeholder = st.empty() # For continuous screening status messages

# No need to call get_nifty500_symbols() here again, use all_symbols_map

if not all_symbols_map:
    st.warning("No equity symbols loaded from CSV. Please ensure 'NSE_Equity.csv' is correct and accessible.")
    st.stop()

# Main continuous screening loop
while run_screener:
    with status_placeholder.container():
        st.info(f"Running screening cycle at {datetime.datetime.now().strftime('%H:%M:%S')}...")
    
    # End-of-day exit condition (3:10 PM IST)
    now = datetime.datetime.now()
    market_close_time = now.replace(hour=15, minute=10, second=0, microsecond=0) # 3:10 PM IST

    # Reset the flag at the beginning of a new trading day (e.g., after midnight or before market open)
    # This also handles cases where the app runs across midnight
    if st.session_state.last_run_date != now.date():
        st.session_state.exit_all_triggered_today = False
        st.session_state.last_run_date = now.date()

    if now >= market_close_time and not st.session_state.exit_all_triggered_today:
        st.warning("Market close time (3:10 PM) reached. Initiating exit for all open positions.")
        positions_to_exit = list(st.session_state.open_tracked_trades.items()) # Make a copy to iterate
        for tsym, trade_info in positions_to_exit:
            if trade_info['status'] == 'OPEN':
                st.info(f"Auto-exiting {tsym} due to EOD rule...")
                
                current_lp = 0
                token_for_eod = all_symbols_map.get(tsym)
                if token_for_eod:
                    try:
                        quote_resp = api.get_quotes(exchange=EXCHANGE, token=token_for_eod) # Individual quote fetch
                        if quote_resp and quote_resp.get('stat') == 'Ok' and quote_resp.get('values'):
                            current_lp = float(quote_resp['values'][0]['lp'])
                        else:
                            logging.warning(f"Could not fetch live quote for {tsym} during EOD exit: {quote_resp.get('emsg', 'No error message')}. Using 0.")
                    except Exception as e:
                        logging.error(f"Error fetching quote for {tsym} during EOD exit: {e}")
                        current_lp = 0 # Default to 0 if quote fails
                else:
                    logging.warning(f"Token not found for {tsym} during EOD exit. Using 0.")

                exit_response = place_intraday_order(
                    buy_or_sell=('S' if trade_info['buy_or_sell'] == 'B' else 'B'), # Opposite action
                    tradingsymbol=tsym,
                    quantity=trade_info['quantity'],
                    entry_price=current_lp, # Use current_lp as reference, MKT will take actual
                    api=api,
                    token=trade_info['token']
                )

                if exit_response and exit_response.get('stat') == 'Ok':
                    st.success(f"EOD Exit order placed for {tsym}. Marking as CLOSED.")
                    st.session_state.open_tracked_trades[tsym]['status'] = 'CLOSED'
                    delete_trade_from_supabase(tsym=tsym)
                    if tsym in st.session_state.manual_overrides:
                        del st.session_state.manual_overrides[tsym]
                else:
                    st.error(f"Failed to place EOD exit order for {tsym}: {exit_response.get('emsg', 'Unknown error')}. Please close manually.")
        st.session_state.exit_all_triggered_today = True
        st.rerun() # Rerun to update the UI with closed positions
    
    # --- Update Market Watch ---
    with market_watch_placeholder.container():
        st.subheader("Live Market Watch")
        if st.session_state.market_watch_symbols:
            market_watch_data = []
            
            for mw_tsym in st.session_state.market_watch_symbols:
                ltp_value = "N/A" # Default
                
                if st.session_state.market_watch_source == "Flattrade (NorenApiPy)":
                    token = all_symbols_map.get(mw_tsym)
                    if not token:
                        logging.warning(f"Token not found for Market Watch symbol {mw_tsym}. Skipping.")
                        ltp_value = 'Token Missing'
                    else:
                        logging.debug(f"Fetching Flattrade MW quote for {mw_tsym} with token: {token}")
                        try:
                            mw_quotes = api.get_quotes(exchange=EXCHANGE, token=token)
                            if mw_quotes and mw_quotes.get('stat') == 'Ok' and mw_quotes.get('values'):
                                ltp_value = float(mw_quotes['values'][0]['lp'])
                            else:
                                logging.warning(f"Could not fetch live quote for Flattrade MW symbol {mw_tsym}: {mw_quotes.get('emsg', 'No error message')}")
                                ltp_value = 'API Error'
                        except Exception as e:
                            logging.error(f"Error fetching Flattrade MW quotes for {mw_tsym}: {e}", exc_info=True)
                            ltp_value = 'Error'
                
                elif st.session_state.market_watch_source == "Tradetron (Experimental)":
                    ltp_from_tradetron = get_tradetron_ltp(
                        mw_tsym, 
                        st.session_state.tradetron_cookie, 
                        st.session_state.tradetron_user_agent
                    )
                    # Now ltp_from_tradetron will be a string (JSON or error)
                    ltp_value = ltp_from_tradetron 

                market_watch_data.append({
                    'Symbol': mw_tsym,
                    # Display the full JSON string or the error message
                    'LTP': ltp_value
                })
            
            if market_watch_data:
                st.dataframe(pd.DataFrame(market_watch_data))
            else:
                st.info("No live data available for selected Market Watch symbols.")
        else:
            st.info("Select symbols in the multiselect above to add them to your Market Watch.")


    # --- Phase 1: Monitor OPEN trades for SL/TP ---
    st.markdown("---")
    st.subheader("Monitoring Open Trades...")
    # This now returns a list of trades to be closed, not closing them directly
    trades_to_close_in_batch = monitor_open_trades(api, all_symbols_map) 

    # --- Phase 2: Monitor PENDING entries for trigger ---
    st.markdown("---")
    st.subheader("Monitoring Pending Entries...")
    # This now returns a list of entries to be executed, not executing them directly
    entries_to_execute_in_batch = monitor_pending_entries(api, all_symbols_map)

    # --- Phase 3: Screen for new signals (only non-tracked/non-pending stocks) ---
    st.markdown("---")
    st.subheader("Screening for New Signals...")
    buy_signals_data = []
    sell_signals_data = []

    eligible_for_screening = get_eligible_symbols_for_screening(all_symbols_map)

    if not eligible_for_screening:
        status_placeholder.info("All eligible symbols are either being tracked, pending entry, or no new symbols to screen.")
        st.empty() # Clear progress bar if any
    else:
        progress_bar = st.progress(0, text="Screening eligible stocks for new signals...")
        for i, stock in enumerate(eligible_for_screening):
            symbol_info, signal, reason, current_ltp_for_signal, signal_high, signal_low, signal_candle_time = screen_stock(stock, api, all_symbols_map) # Added signal_candle_time
            
            if signal in ['BUY', 'SELL'] and current_ltp_for_signal is not None and signal_high is not None and signal_low is not None:
                current_stock_token = stock.get('token')
                if not current_stock_token:
                    logging.warning(f"Cannot process signal for {symbol_info}: Missing valid API Token.")
                    continue
                
                if signal == 'BUY':
                    expected_entry_price = round(signal_high * (1 + ENTRY_BUFFER_PERCENT), 2)
                    initial_sl_price = round(signal_low - st.session_state.sl_buffer_points, 2)
                    
                    potential_loss_per_share = expected_entry_price - initial_sl_price
                    
                    if potential_loss_per_share <= 0.01:
                        logging.warning(f"Invalid or too small potential loss ({potential_loss_per_share}) for BUY {symbol_info}. Skipping.")
                        continue
                    
                    calculated_quantity = int( (st.session_state.capital * RISK_PERCENTAGE_OF_CAPITAL) / potential_loss_per_share )
                    initial_tp_price = round(expected_entry_price + (potential_loss_per_share * st.session_state.target_multiplier), 2)

                    buy_signals_data.append({
                        'Symbol': symbol_info,
                        'Signal': 'BUY',
                        'Reason': reason,
                        'Price': f"{current_ltp_for_signal:,.2f}",
                        'Signal High': f"{signal_high:,.2f}",
                        'Signal Low': f"{signal_low:,.2f}",
                        'Est. Qty': calculated_quantity,
                        'Est. SL': f"{initial_sl_price:,.2f}",
                        'Est. TP': f"{initial_tp_price:,.2f}",
                        'Time': signal_candle_time # Display signal candle time
                    })
                    
                    if calculated_quantity > 0:
                        st.session_state.pending_entries[symbol_info] = {
                            'buy_or_sell': 'B',
                            'signal_candle_high': signal_high,
                            'signal_candle_low': signal_low,
                            'calculated_quantity': calculated_quantity,
                            'initial_sl_price': initial_sl_price,
                            'initial_tp_price': initial_tp_price,
                            'status': 'PENDING',
                            'token': current_stock_token
                        }
                        # Save pending entry to Supabase
                        supabase_payload = {
                            'tsym': symbol_info,
                            'exchange': EXCHANGE,
                            'token': current_stock_token,
                            'buy_or_sell': 'B',
                            'quantity': calculated_quantity,
                            'entry_price': None, # No entry price yet for pending
                            'sl_price': initial_sl_price,
                            'target_price': initial_tp_price,
                            'status': 'PENDING',
                            'highest_price_seen': None,
                            'lowest_price_seen': None,
                            'signal_candle_high': signal_high,
                            'signal_candle_low': signal_low,
                            'manual_sl_price': st.session_state.manual_overrides.get(symbol_info, {}).get('sl_price'),
                            'manual_target_price': st.session_state.manual_overrides.get(symbol_info, {}).get('target_price')
                        }
                        upsert_trade_to_supabase(supabase_payload)
                        status_placeholder.info(f"Added BUY signal for {symbol_info} to pending entries. Qty: {calculated_quantity}, SL: {initial_sl_price:.2f}, TP: {initial_tp_price:.2f}")
                    else:
                        status_placeholder.warning(f"Calculated quantity for BUY {symbol_info} is zero. Not adding to pending entries.")


                elif signal == 'SELL':
                    expected_entry_price = round(signal_low * (1 - ENTRY_BUFFER_PERCENT), 2)
                    initial_sl_price = round(signal_high + st.session_state.sl_buffer_points, 2)
                    
                    potential_loss_per_share = initial_sl_price - expected_entry_price

                    if potential_loss_per_share <= 0.01:
                        logging.warning(f"Invalid or too small potential loss ({potential_loss_per_share}) for SELL {symbol_info}. Skipping.")
                        continue

                    calculated_quantity = int( (st.session_state.capital * RISK_PERCENTAGE_OF_CAPITAL) / potential_loss_per_share )
                    initial_tp_price = round(expected_entry_price - (potential_loss_per_share * st.session_state.target_multiplier), 2)

                    sell_signals_data.append({
                        'Symbol': symbol_info,
                        'Signal': 'SELL',
                        'Reason': reason,
                        'Price': f"{current_ltp_for_signal:,.2f}",
                        'Signal High': f"{signal_high:,.2f}",
                        'Signal Low': f"{signal_low:,.2f}",
                        'Est. Qty': calculated_quantity,
                        'Est. SL': f"{initial_sl_price:,.2f}",
                        'Est. TP': f"{initial_tp_price:,.2f}",
                        'Time': signal_candle_time # Display signal candle time
                    })

                    if calculated_quantity > 0:
                        st.session_state.pending_entries[symbol_info] = {
                            'buy_or_sell': 'S',
                            'signal_candle_high': signal_high,
                            'signal_candle_low': signal_low,
                            'calculated_quantity': calculated_quantity,
                            'initial_sl_price': initial_sl_price,
                            'initial_tp_price': initial_tp_price,
                            'status': 'PENDING',
                            'token': current_stock_token
                        }
                        # Save pending entry to Supabase
                        supabase_payload = {
                            'tsym': symbol_info,
                            'exchange': EXCHANGE,
                            'token': current_stock_token,
                            'buy_or_sell': 'S',
                            'quantity': calculated_quantity,
                            'entry_price': None, # No entry price yet for pending
                            'sl_price': initial_sl_price,
                            'target_price': initial_tp_price,
                            'status': 'PENDING',
                            'highest_price_seen': None,
                            'lowest_price_seen': None,
                            'signal_candle_high': signal_high,
                            'signal_candle_low': signal_low,
                            'manual_sl_price': st.session_state.manual_overrides.get(symbol_info, {}).get('sl_price'),
                            'manual_target_price': st.session_state.manual_overrides.get(symbol_info, {}).get('target_price')
                        }
                        upsert_trade_to_supabase(supabase_payload)
                        status_placeholder.info(f"Added SELL signal for {symbol_info} to pending entries. Qty: {calculated_quantity}, SL: {initial_sl_price:.2f}, TP: {initial_tp_price:.2f}")
                    else:
                        status_placeholder.warning(f"Calculated quantity for SELL {symbol_info} is zero. Not adding to pending entries.")
            
            progress_bar.progress((i + 1) / len(eligible_for_screening), text=f"Screening {stock['tsym']}...")
        progress_bar.empty() # Clear progress bar

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
            
    # --- Process collected state transitions (NEW BATCH PROCESSING) ---
    if trades_to_close_in_batch or entries_to_execute_in_batch:
        st.info("Processing pending trade transitions...")
        rerun_needed = False

        # Process trades to close
        for trade in trades_to_close_in_batch:
            tsym = trade['tsym']
            action = trade['action']
            quantity = trade['quantity']
            token = trade['token']
            status_reason = trade['status_reason'] # Added for logging
            
            st.info(f"Placing closing {action} order for {tsym} (Qty: {quantity}) due to {status_reason}...")
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
                remarks=f'Automated_Exit_{status_reason}'
            )
            if close_response and close_response.get('stat') == 'Ok':
                st.success(f"Closing order for {tsym} successful: {close_response}")
                # Update session state after successful order
                if tsym in st.session_state.open_tracked_trades:
                    st.session_state.open_tracked_trades[tsym]['status'] = 'CLOSED' 
                    delete_trade_from_supabase(tsym=tsym) # Delete from DB
                    if tsym in st.session_state.manual_overrides:
                        del st.session_state.manual_overrides[tsym] # Clean up manual overrides
                rerun_needed = True
            else:
                st.error(f"Failed to place closing order for {tsym}: {close_response.get('emsg', 'Unknown error')}")
                logging.error(f"Failed to place closing order for {tsym}: {close_response}")

        # Process entries to execute
        for entry in entries_to_execute_in_batch:
            tsym = entry['tsym']
            action = entry['action']
            quantity = entry['quantity']
            entry_price = entry['entry_price']
            sl_price = entry['sl_price']
            target_price = entry['target_price']
            token = entry['token']
            signal_candle_high = entry.get('signal_candle_high')
            signal_candle_low = entry.get('signal_candle_low')

            st.info(f"Placing initial {action} order for {tsym} (Qty: {quantity}) at LTP {entry_price:.2f}...")
            order_response = place_intraday_order(
                buy_or_sell=action,
                tradingsymbol=tsym,
                quantity=quantity,
                entry_price=entry_price, 
                api=api,
                token=token
            )
            
            if order_response and order_response.get('stat') == 'Ok':
                # Update session state after successful order
                if tsym in st.session_state.pending_entries:
                    del st.session_state.pending_entries[tsym]
                    logging.info(f"Successfully removed {tsym} from pending_entries after order placement")
                
                new_tracked_trade = {
                    'order_no': order_response.get('norenordno'),
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'sl_price': sl_price,
                    'target_price': target_price,
                    'buy_or_sell': action,
                    'status': 'OPEN',
                    'token': token,
                    'highest_price_seen': entry_price if action == 'B' else None,
                    'lowest_price_seen': entry_price if action == 'S' else None,
                    'current_ltp': entry_price 
                }
                st.session_state.open_tracked_trades[tsym] = new_tracked_trade
                logging.info(f"Successfully moved pending entry {tsym} to open tracked trades with SL: {sl_price:.2f}, TP: {target_price:.2f}")
                
                try:
                    delete_trade_from_supabase(tsym)
                    logging.info(f"Deleted PENDING entry for {tsym} from Supabase")
                    
                    supabase_payload = {
                        'tsym': tsym,
                        'exchange': EXCHANGE,
                        'token': token,
                        'buy_or_sell': action,
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'target_price': target_price,
                        'status': 'OPEN',
                        'order_no': order_response.get('norenordno'),
                        'highest_price_seen': new_tracked_trade['highest_price_seen'],
                        'lowest_price_seen': new_tracked_trade['lowest_price_seen'],
                        'signal_candle_high': signal_candle_high,
                        'signal_candle_low': signal_candle_low,
                        'manual_sl_price': st.session_state.manual_overrides.get(tsym, {}).get('sl_price'),
                        'manual_target_price': st.session_state.manual_overrides.get(tsym, {}).get('target_price')
                    }
                    
                    if upsert_trade_to_supabase(supabase_payload):
                        logging.info(f"Successfully saved OPEN trade {tsym} to Supabase")
                    else:
                        logging.error(f"Failed to save OPEN trade {tsym} to Supabase")
                        
                except Exception as db_error:
                    logging.error(f"Database operation error for {tsym}: {db_error}")
                    st.error(f"Database error for {tsym}: {db_error}")

                if tsym in st.session_state.manual_overrides:
                    del st.session_state.manual_overrides[tsym]
                    logging.info(f"Cleared manual overrides for {tsym} after trade execution")
                rerun_needed = True
                    
            else:
                st.error(f"Failed to place entry order for {tsym}. Reverting status to PENDING.")
                if tsym in st.session_state.pending_entries:
                    st.session_state.pending_entries[tsym]['status'] = 'PENDING' 
                logging.error(f"Failed to place entry order for {tsym}. Order Response: {order_response}")
        
        # Trigger a single rerun if any state changes occurred
        if rerun_needed:
            st.rerun()


    # --- Monitor and Display Tracked Trades (new section) ---
    with tracked_trades_placeholder.container():
        st.subheader("App-Tracked Trades (Pending & Open) 🚀")
        tracked_data = []

        # Adjusted columns for LTP
        cols_header = st.columns([0.15, 0.04, 0.04, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.05])
        cols_header[0].write("**Symbol**")
        cols_header[1].write("**Act**")
        cols_header[2].write("**Qty**")
        cols_header[3].write("**LTP**") # New LTP column header
        cols_header[4].write("**Entry Ref Price**")
        cols_header[5].write("**SL Price**")
        cols_header[6].write("**Target Price**")
        cols_header[7].write("**Status**")
        cols_header[8].write("**Manual SL**")
        cols_header[9].write("**Manual Target**")
        cols_header[10].write("**Clear**") # Corrected this line to use `cols_header[10]`

        # Explicitly build and prioritize the list of tracked symbols for rendering
        all_tracked_tsyms_ordered = []
        
        # Add pending entries first
        for tsym in st.session_state.pending_entries.keys():
            if st.session_state.pending_entries[tsym]['status'] == 'PENDING':
                all_tracked_tsyms_ordered.append(tsym)
        
        # Add open trades that are NOT already in pending
        for tsym in st.session_state.open_tracked_trades.keys():
            if tsym not in all_tracked_tsyms_ordered and \
               (st.session_state.open_tracked_trades[tsym]['status'] == 'OPEN' or \
                st.session_state.open_tracked_trades[tsym]['status'].startswith('CLOSING')):
                all_tracked_tsyms_ordered.append(tsym)
        
        # Now iterate over the strictly ordered and unique list
        for tsym in all_tracked_tsyms_ordered:
            # Strictly prioritize pending entries for rendering
            if tsym in st.session_state.pending_entries and st.session_state.pending_entries[tsym]['status'] == 'PENDING':
                trade = st.session_state.pending_entries[tsym]
                
                current_sl_display = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
                if current_sl_display is None or current_sl_display <= 0:
                    current_sl_display = trade['initial_sl_price']
                current_sl_display = f"{current_sl_display:,.2f}" if current_sl_display is not None else 'N/A'

                current_tp_display = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
                if current_tp_display is None or current_tp_display <= 0:
                    current_tp_display = trade['initial_tp_price']
                current_tp_display = f"{current_tp_display:,.2f}" if current_tp_display is not None else 'N/A'
                
                manual_sl_val = st.session_state.manual_overrides.get(tsym, {}).get('sl_price', None)
                manual_tp_val = st.session_state.manual_overrides.get(tsym, {}).get('target_price', None)

                cols = st.columns([0.15, 0.04, 0.04, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.05])
                cols[0].write(tsym)
                cols[1].write(trade['buy_or_sell'])
                cols[2].write(trade['calculated_quantity'])
                # Use .get() for current_ltp and check if it's numeric before formatting
                ltp_value = trade.get('current_ltp')
                if isinstance(ltp_value, (float, int)):
                    cols[3].write(f"{ltp_value:,.2f}")
                else:
                    cols[3].write("Fetching...")
                cols[4].write(f"{trade['signal_candle_high'] if trade['buy_or_sell'] == 'B' else trade['signal_candle_low']:,.2f}")
                cols[5].write(current_sl_display)
                cols[6].write(current_tp_display)
                cols[7].write('PENDING ENTRY')
                
                # --- CORRECTED: Stable unique keys for number_input widgets ---
                new_manual_sl = cols[8].number_input(
                    label="Manual SL", # Simplified label
                    value=manual_sl_val if manual_sl_val is not None and manual_sl_val > 0 else 0.0,
                    step=0.01, format="%.2f", 
                    key=f"manual_sl_input_{tsym}_pending", # Stable key
                    label_visibility="collapsed"
                )
                new_manual_tp = cols[9].number_input(
                    label="Manual TP", # Simplified label
                    value=manual_tp_val if manual_tp_val is not None and manual_tp_val > 0 else 0.0,
                    step=0.01, format="%.2f", 
                    key=f"manual_tp_input_{tsym}_pending", # Stable key
                    label_visibility="collapsed"
                )
                
                if new_manual_sl is not None and (manual_sl_val is None or new_manual_sl != manual_sl_val):
                    st.session_state.manual_overrides.setdefault(tsym, {})['sl_price'] = new_manual_sl
                    upsert_trade_to_supabase({'tsym': tsym, 'manual_sl_price': new_manual_sl})
                    # Removed st.rerun()
                if new_manual_tp is not None and (manual_tp_val is None or new_manual_tp != manual_tp_val):
                    st.session_state.manual_overrides.setdefault(tsym, {})['target_price'] = new_manual_tp
                    upsert_trade_to_supabase({'tsym': tsym, 'manual_target_price': new_manual_tp})
                    # Removed st.rerun()

                if cols[10].button("Clear", key=f"clear_manual_{tsym}_pending_button"):
                    if tsym in st.session_state.manual_overrides:
                        if 'sl_price' in st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]['sl_price']
                        if 'target_price' in st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]['target_price']
                        if not st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]
                        upsert_trade_to_supabase({'tsym': tsym, 'manual_sl_price': None, 'manual_target_price': None})
                        # Removed st.rerun()

            # Else, if it's an OPEN or CLOSING tracked trade
            elif tsym in st.session_state.open_tracked_trades and (st.session_state.open_tracked_trades[tsym]['status'] == 'OPEN' or st.session_state.open_tracked_trades[tsym]['status'].startswith('CLOSING')):
                trade = st.session_state.open_tracked_trades[tsym]
                
                current_sl_display = st.session_state.manual_overrides.get(tsym, {}).get('sl_price')
                if current_sl_display is None or current_sl_display <= 0:
                    current_sl_display = trade['sl_price']
                current_sl_display = f"{current_sl_display:,.2f}" if current_sl_display is not None else 'N/A'

                current_tp_display = st.session_state.manual_overrides.get(tsym, {}).get('target_price')
                if current_tp_display is None or current_tp_display <= 0:
                    current_tp_display = trade['target_price']
                current_tp_display = f"{current_tp_display:,.2f}" if current_tp_display is not None else 'N/A'

                manual_sl_val = st.session_state.manual_overrides.get(tsym, {}).get('sl_price', None)
                manual_tp_val = st.session_state.manual_overrides.get(tsym, {}).get('target_price', None)
                
                cols = st.columns([0.15, 0.04, 0.04, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.05])
                cols[0].write(tsym)
                cols[1].write(trade['buy_or_sell'])
                cols[2].write(trade['quantity'])
                # Use .get() for current_ltp and check if it's numeric before formatting
                ltp_value = trade.get('current_ltp')
                if isinstance(ltp_value, (float, int)):
                    cols[3].write(f"{ltp_value:,.2f}")
                else:
                    cols[3].write("Fetching...")
                cols[4].write(f"{trade['entry_price']:,.2f}")
                cols[5].write(current_sl_display)
                cols[6].write(current_tp_display)
                cols[7].write(trade['status'])

                # --- CORRECTED: Stable unique keys for number_input widgets ---
                new_manual_sl = cols[8].number_input(
                    label="Manual SL", # Simplified label
                    value=manual_sl_val if manual_sl_val is not None and manual_sl_val > 0 else 0.0,
                    step=0.01, format="%.2f", 
                    key=f"manual_sl_input_{tsym}_open", # Stable key
                    label_visibility="collapsed"
                )
                new_manual_tp = cols[9].number_input(
                    label="Manual TP", # Simplified label
                    value=manual_tp_val if manual_tp_val is not None and manual_tp_val > 0 else 0.0,
                    step=0.01, format="%.2f", 
                    key=f"manual_tp_input_{tsym}_open", # Stable key
                    label_visibility="collapsed"
                )

                if new_manual_sl is not None and (manual_sl_val is None or new_manual_sl != manual_sl_val): # Corrected comparison
                    st.session_state.manual_overrides.setdefault(tsym, {})['sl_price'] = new_manual_sl
                    upsert_trade_to_supabase({'tsym': tsym, 'manual_sl_price': new_manual_sl})
                    # Removed st.rerun()
                if new_manual_tp is not None and (manual_tp_val is None or new_manual_tp != manual_tp_val):
                    st.session_state.manual_overrides.setdefault(tsym, {})['target_price'] = new_manual_tp
                    upsert_trade_to_supabase({'tsym': tsym, 'manual_target_price': new_manual_tp})
                    # Removed st.rerun()
                
                if cols[10].button("Clear", key=f"clear_manual_{tsym}_open_button"):
                    if tsym in st.session_state.manual_overrides:
                        if 'sl_price' in st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]['sl_price']
                        if 'target_price' in st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]['target_price']
                        if not st.session_state.manual_overrides[tsym]:
                            del st.session_state.manual_overrides[tsym]
                        upsert_trade_to_supabase({'tsym': tsym, 'manual_sl_price': None, 'manual_target_price': None})
                        # Removed st.rerun()
        
        if not all_tracked_tsyms_ordered:
            st.info("No active or pending trades being tracked by the app.")

    
    # Wait for the specified interval before the next screening cycle
    if run_screener:
        st.info(f"Next full screening cycle in {screen_interval} seconds...")
        time.sleep(screen_interval)

if not run_screener:
    status_placeholder.info("Screener is paused. Check 'Run Screener Continuously' in sidebar to start.")
