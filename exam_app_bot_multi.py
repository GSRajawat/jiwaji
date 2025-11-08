import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time
import time
import json
from io import StringIO

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NSE Enhanced Trading Strategy",
    page_icon="ðŸ“Š",
    layout="wide"
)

# ==================== SESSION STATE (MERGED) ====================
# Initialize session state for both auth and strategy
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'otp_sent' not in st.session_state:
    st.session_state.otp_sent = False
if 'api_credentials' not in st.session_state:
    st.session_state.api_credentials = {}
if 'api_instance' not in st.session_state:
    st.session_state.api_instance = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

# State from new strategy
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trailing_stops' not in st.session_state:
    st.session_state.trailing_stops = {}

# ==================== DEMO DATA GENERATOR ====================
def generate_demo_data():
    """Generate realistic demo data when NSE API is unavailable"""
    
    stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", 
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT",
        "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN", "BAJFINANCE",
        "NESTLEIND", "WIPRO", "TECHM", "HCLTECH", "TATAMOTORS", "ONGC",
        "NTPC", "POWERGRID", "ADANIPORTS", "JSWSTEEL", "TATASTEEL", "HINDALCO",
        "DIVISLAB", "BRITANNIA", "DRREDDY", "APOLLOHOSP", "CIPLA", "EICHERMOT",
        "GRASIM", "HEROMOTOCO", "HINDZINC", "INDUSINDBK", "ADANIENT", "COALINDIA"
    ]
    
    all_stocks = []
    
    np.random.seed(42)
    
    for i, symbol in enumerate(stocks):
        base_price = np.random.uniform(100, 2000)
        open_price = base_price * np.random.uniform(0.98, 1.02)
        
        # Random positive or negative change
        pct_change = np.random.uniform(-8.0, 8.0)
        ltp = open_price * (1 + pct_change/100)
        
        if pct_change > 0:
            day_high = ltp * 1.005
            day_low = open_price * 0.995
        else:
            day_high = open_price * 1.005
            day_low = ltp * 0.995
        
        current_traded_value = np.random.uniform(50000000, 5000000000)
        prev_traded_value = current_traded_value * np.random.uniform(0.7, 1.3)
        
        all_stocks.append({
            'symbol': symbol,
            'open': round(open_price, 2),
            'ltp': round(ltp, 2),
            'high': round(day_high, 2),
            'low': round(day_low, 2),
            'pChange': round(pct_change, 2),
            'tradedValue': current_traded_value,
            'prevTradedValue': prev_traded_value
        })
    
    return {'data': all_stocks}

# ==================== NSE DATA FETCHER ====================
class NSEDataFetcher:
    """Fetch live data from NSE India"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_cookies(self):
        """Get cookies by visiting NSE homepage"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Cookie fetch error: {str(e)}")
            return False
    
    def fetch_all_securities(self):
        """Fetch all securities data from NSE"""
        try:
            # Get cookies first
            self.get_cookies()
            time.sleep(1)
            
            # Try to fetch all securities in F&O
            try:
                url = f"{self.base_url}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("âœ… Fetched all F&O securities")
                    return data
            except Exception as e:
                st.warning(f"F&O endpoint failed: {str(e)}")
            
            # Try broader market endpoint
            try:
                url = f"{self.base_url}/api/allIndices"
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    st.info("âœ… Using all indices endpoint")
                    return data
            except Exception as e:
                st.warning(f"All indices endpoint failed: {str(e)}")
            
            st.error("âŒ All API endpoints failed. Using demo mode.")
            return None
                
        except Exception as e:
            st.error(f"Error fetching NSE data: {str(e)}")
            return None
    
    def parse_nse_data(self, nse_data):
        """Parse NSE JSON data into dataframe"""
        try:
            if not nse_data:
                return pd.DataFrame()
            
            # Debug: Show raw data structure
            with st.expander("ðŸ” Debug: Raw API Response", expanded=False):
                st.write("**Data Keys:**", list(nse_data.keys()) if nse_data else "None")
                if 'data' in nse_data and isinstance(nse_data['data'], list) and len(nse_data['data']) > 0:
                    st.write("**Sample Record:**", nse_data['data'][0])
            
            # Parse data
            df = pd.DataFrame()
            
            if 'data' in nse_data and isinstance(nse_data['data'], list):
                df = pd.DataFrame(nse_data['data'])
            elif isinstance(nse_data, list):
                df = pd.DataFrame(nse_data)
            
            if df.empty:
                st.warning("âš ï¸ No data found in API response")
                return pd.DataFrame()
            
            # Normalize column names - comprehensive mapping
            column_mapping = {
                'open_price': 'open',
                'high_price': 'high', 
                'low_price': 'low',
                'prev_price': 'prev_close',
                'net_price': 'change',
                'trade_quantity': 'volume',
                'turnover': 'tradedValue',
                'lastPrice': 'ltp',
                'last': 'ltp',
                'previousClose': 'prev_close',
                'dayHigh': 'high',
                'dayLow': 'low',
                'totalTradedVolume': 'volume',
                'totalTradedValue': 'tradedValue',
                'pChange': 'pChange',
                'perChange': 'pChange',
                'percentChange': 'pChange'
            }
            df = df.rename(columns=column_mapping)
            
            # Show available columns for debugging
            with st.expander("ðŸ” Available Columns", expanded=False):
                st.write("**All columns:**", df.columns.tolist())
            
            # Handle missing columns - create with defaults
            if 'ltp' not in df.columns:
                if 'lastPrice' in df.columns:
                    df['ltp'] = df['lastPrice']
                elif 'last' in df.columns:
                    df['ltp'] = df['last']
                else:
                    st.error("âŒ Cannot find price column (ltp/lastPrice/last)")
                    return pd.DataFrame()
            
            if 'open' not in df.columns:
                if 'open_price' in df.columns:
                    df['open'] = df['open_price']
                else:
                    df['open'] = df['ltp']  # Fallback
            
            if 'high' not in df.columns:
                if 'high_price' in df.columns:
                    df['high'] = df['high_price']
                elif 'dayHigh' in df.columns:
                    df['high'] = df['dayHigh']
                else:
                    df['high'] = df['ltp']  # Fallback
            
            if 'low' not in df.columns:
                if 'low_price' in df.columns:
                    df['low'] = df['low_price']
                elif 'dayLow' in df.columns:
                    df['low'] = df['dayLow']
                else:
                    df['low'] = df['ltp']  # Fallback
            
            if 'tradedValue' not in df.columns:
                if 'turnover' in df.columns:
                    df['tradedValue'] = df['turnover']
                elif 'totalTradedValue' in df.columns:
                    df['tradedValue'] = df['totalTradedValue']
                else:
                    st.error("âŒ Cannot find traded value column")
                    return pd.DataFrame()
            
            if 'symbol' not in df.columns:
                st.error("âŒ Symbol column not found")
                return pd.DataFrame()
            
            # Convert all numeric columns to float, handling strings
            numeric_cols = ['ltp', 'open', 'high', 'low', 'tradedValue']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate pChange if not present
            if 'pChange' not in df.columns:
                if 'prev_close' in df.columns:
                    df['prev_close'] = pd.to_numeric(df['prev_close'], errors='coerce')
                    df['pChange'] = ((df['ltp'] - df['prev_close']) / df['prev_close']) * 100
                elif 'open' in df.columns:
                    df['pChange'] = ((df['ltp'] - df['open']) / df['open']) * 100
                else:
                    df['pChange'] = 0
            else:
                df['pChange'] = pd.to_numeric(df['pChange'], errors='coerce')
            
            # Convert traded value from lakhs to actual value if needed
            max_value = df['tradedValue'].max()
            if not pd.isna(max_value) and max_value < 1000000:  # Likely in lakhs
                df['tradedValue'] = df['tradedValue'] * 100000
            
            # Add previous day traded value (simulated - in real scenario, fetch from historical data)
            # For demo, we'll use 80-120% of current value as previous day value
            np.random.seed(42)
            df['prevTradedValue'] = df['tradedValue'] * np.random.uniform(0.8, 1.2, len(df))
            
            # Clean data - remove rows with invalid values
            df = df.dropna(subset=['ltp', 'open', 'high', 'low', 'tradedValue'])
            df = df[df['ltp'] > 0]
            df = df[df['tradedValue'] > 0]
            
            # Fill any remaining NaN values in pChange
            df['pChange'] = df['pChange'].fillna(0)
            
            with st.expander("ðŸ“Š Parsed Data Info", expanded=False):
                st.write(f"âœ… Total stocks: {len(df)} rows")
                st.write("Columns:", df.columns.tolist())
                st.write("Data types:", df.dtypes.to_dict())
                st.dataframe(df.head(5))
            
            return df
            
        except Exception as e:
            st.error(f"âŒ Error parsing data: {str(e)}")
            st.write("**Error details:**", str(e))
            
            import traceback
            with st.expander("ðŸ› Full Error Debug", expanded=True):
                st.code(traceback.format_exc())
                st.json({
                    "data_sample": str(nse_data)[:1000] if nse_data else "None"
                })
            
            return pd.DataFrame()

# ==================== DEFINEDGE API SETUP (FIXED ENDPOINT) ====================
class DefinedgeAPI:
    """Wrapper for Definedge Securities API"""
    
    def __init__(self):
        # Definedge Securities actual API endpoints
        self.auth_base_url = "https://signin.definedgesecurities.com/auth/realms/debroking/dsbpkc"
        
        # FIX: Setting the base URL to the root for the Dart API. 
        # We will manually prepend '/dart/v1' to the endpoints if required.
        self.api_base_url = "https://integrate.definedgesecurities.com" 
        
        self.user_id = None
        self.api_token = None
        self.api_secret = None
        self.otp_token = None # This is the short-lived token for OTP verification
        self.access_token = None # This is the main session key
    
    def set_credentials(self, user_id, api_token, api_secret):
        """Set user credentials"""
        self.user_id = user_id
        self.api_token = api_token
        self.api_secret = api_secret
    
    def generate_otp(self):
        """Request OTP from Definedge using actual API"""
        try:
            url = f"{self.auth_base_url}/login/{self.api_token}"
            headers = {
                "api_secret": self.api_secret,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            st.info(f"ðŸ“¡ Sending OTP request to Definedge API...")
            response = requests.get(url, headers=headers, timeout=10)
            
            st.info(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                # Store otp_token from response
                self.otp_token = data.get('otp_token')
                message = data.get('message', 'OTP sent successfully')
                
                if self.otp_token:
                    st.success(f"âœ… OTP Token received.")
                    return True, message, self.otp_token
                else:
                    return False, "OTP token not found in response", None
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}", None
            
        except requests.exceptions.Timeout:
            return False, "Request timeout. Please check your internet connection.", None
        except Exception as e:
            return False, f"Error generating OTP: {str(e)}", None
    
    def verify_otp_and_authenticate(self, otp, otp_token):
        """Verify OTP and complete authentication"""
        try:
            self.otp_token = otp_token
            if not self.otp_token:
                return False, "OTP token not found. Please generate OTP first."
            
            url = f"{self.auth_base_url}/token" # Corrected endpoint
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "otp_token": self.otp_token,
                "otp": otp
            }
            
            st.info(f"ðŸ“¡ Verifying OTP with Definedge API...")
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            st.info(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                # Extract session token (api_session_key)
                self.access_token = data.get('api_session_key') or data.get('susertoken')
                
                if self.access_token:
                    st.success(f"âœ… Session Key acquired!")
                    return True, self.access_token
                else:
                    return False, "Session key not found in response"
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
            
        except Exception as e:
            return False, f"Error verifying OTP: {str(e)}"

    def get_profile(self):
        """Fetch user profile to validate session"""
        if not self.access_token:
            return None, "Not authenticated"
        try:
            # Reconstruct the full path
            url = f"{self.api_base_url}/dart/v1/user/profile"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return None, f"Exception fetching profile: {str(e)}"

    def place_order(self, symbol, side, quantity, exchange="NSE", product="INTRADAY", order_type="MARKET", price=0):
        """Place an order via Definedge API"""
        if not self.access_token:
            return None, "Not authenticated"
            
        try:
            # FIX ATTEMPT 2: Based on 404 error, we are reconstructing the path manually.
            # The previous URL was: self.api_base_url + /orders/regular (where base includes /v1)
            # New URL structure: self.api_base_url/dart/v1/orders/regular
            url = f"{self.api_base_url}/dart/v1/orders/regular"
            
            st.info(f"Targeting Order URL: {url}")
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Definedge API payload structure - Ensure quantity is integer
            payload = {
                "exchange": exchange,
                "symbol": symbol,
                "side": side,  # "BUY" or "SELL"
                "quantity": int(quantity), # CRITICAL: Ensure integer
                "product": product, # "INTRADAY", "CNC", "NORMAL"
                "order_type": order_type, # "MARKET", "LIMIT"
                "price": float(price) if order_type == "LIMIT" else 0,
                "validity": "DAY",
            }
            
            st.info(f"Placing {side} order for {int(quantity)} of {symbol}...")
            st.json({"Payload Sent": payload}) # DEBUG: Show payload
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            # CRITICAL DEBUG STEP: Read response regardless of status code
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                # This often happens when 404 or 500 error returns HTML instead of JSON
                data = {"raw_response": response.text, "status_code": response.status_code, "error_details": "JSON decode failed, likely non-JSON error response."}
                
            st.json({"API Response": data}) # DEBUG: Show full response
            
            if response.status_code == 200:
                if data.get('status') == 'success' and data.get('data', {}).get('order_id'):
                    return data, f"Order placed successfully! Order ID: {data['data']['order_id']}"
                else:
                    return data, f"Order placement failed: {data.get('message', 'Unknown error')}. Check full response above."
            else:
                return None, f"API Error {response.status_code}: Order failed. Check full response above."

        except Exception as e:
            return None, f"Exception placing order: {str(e)}"

# ==================== AUTHENTICATION UI (RE-INTRODUCED) ====================
def show_authentication_page():
    """Display authentication interface"""
    st.title("ðŸ” Definedge Securities Authentication")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Connect to Broker
        Login to your Definedge Securities account to enable signal execution.
        
        **Steps:**
        1. Enter your API credentials (saved from your Definedge dashboard).
        2. Click "Save Credentials".
        3. Click "Generate OTP".
        4. Enter the OTP sent to your device and click "Verify & Login".
        """)
        
        if st.button("ðŸ§ª Continue with Demo Mode (No Trading)", type="secondary"):
            st.session_state.authenticated = True
            st.session_state.api_instance = None # No API instance in demo
            st.session_state.api_credentials = {'user_id': 'DEMO_USER'}
            st.info("âœ… Demo mode activated! Trading buttons will be disabled.")
            time.sleep(1)
            st.rerun()

    with col2:
        with st.form("credentials_form"):
            st.markdown("### ðŸ“‹ API Credentials")
            user_id = st.text_input("User ID", help="Your Definedge User ID")
            api_token = st.text_input("API Token", type="password", help="API Token from Definedge")
            api_secret = st.text_input("API Secret", type="password", help="API Secret from Definedge")
            
            submit_creds = st.form_submit_button("ðŸ’¾ Save Credentials")
            
            if submit_creds:
                if user_id and api_token and api_secret:
                    st.session_state.api_credentials = {
                        'user_id': user_id,
                        'api_token': api_token,
                        'api_secret': api_secret
                    }
                    st.success("âœ… Credentials saved! Please generate OTP.")
                else:
                    st.error("Please fill all credential fields")
    
    if st.session_state.api_credentials and not st.session_state.otp_sent:
        st.markdown("---")
        if st.button("ðŸ“¤ Generate OTP", use_container_width=True, type="primary"):
            api = DefinedgeAPI()
            api.set_credentials(
                st.session_state.api_credentials['user_id'],
                st.session_state.api_credentials['api_token'],
                st.session_state.api_credentials['api_secret']
            )
            with st.spinner("Generating OTP..."):
                success, message, otp_token = api.generate_otp()
                if success:
                    st.session_state.otp_sent = True
                    st.session_state.temp_otp_token = otp_token # Store short-lived token
                    st.success(f"âœ… {message}")
                    st.rerun()
                else:
                    st.error(f"âŒ {message}")

    if st.session_state.otp_sent:
        st.markdown("---")
        with st.form("otp_form"):
            st.markdown("### ðŸ”‘ Enter OTP")
            otp = st.text_input("Enter 6-digit OTP")
            
            submit_otp = st.form_submit_button("âœ… Verify & Login", use_container_width=True)
            
            if submit_otp:
                if otp and len(otp) == 6:
                    api = DefinedgeAPI()
                    api.set_credentials(
                        st.session_state.api_credentials['user_id'],
                        st.session_state.api_credentials['api_token'],
                        st.session_state.api_credentials['api_secret']
                    )
                    with st.spinner("Verifying OTP..."):
                        # Pass the stored short-lived token
                        success, token_or_message = api.verify_otp_and_authenticate(
                            otp, 
                            st.session_state.temp_otp_token
                        )
                        
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.access_token = token_or_message
                            api.access_token = token_or_message # Set token in API instance
                            st.session_state.api_instance = api # Save full API instance
                            st.session_state.otp_sent = False
                            st.session_state.temp_otp_token = None
                            st.success("âœ… Authentication successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"âŒ {token_or_message}")
                else:
                    st.error("Please enter a valid 6-digit OTP")

# ==================== STRATEGY ====================
class EnhancedTradingStrategy:
    """Enhanced trading strategy with new rules"""
    
    def __init__(self):
        self.positions = []
        self.trailing_stops = {}
    
    def is_trading_time(self):
        """Check if current time is between 9:25 AM and 2:30 PM"""
        now = datetime.now().time()
        start_time = dt_time(9, 25)
        end_time = dt_time(14, 30)
        
        return start_time <= now <= end_time
    
    def filter_by_traded_value(self, df):
        """Filter stocks where current traded value > previous day traded value"""
        if df.empty:
            return df
        
        # Filter: current traded value > previous traded value
        df_filtered = df[df['tradedValue'] > df['prevTradedValue']].copy()
        
        return df_filtered
    
    def generate_signals(self, df_all):
        """Generate buy/sell signals based on enhanced strategy"""
        signals = []
        
        if df_all.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        # Step 1: Filter by traded value condition
        df_filtered = self.filter_by_traded_value(df_all)
        
        if df_filtered.empty:
            st.warning("âš ï¸ No stocks meet the traded value criteria (current > previous)")
            return signals, pd.DataFrame(), pd.DataFrame()
        
        # Step 2: Separate gainers and losers
        df_gainers = df_filtered[df_filtered['pChange'] > 0].copy()
        df_losers = df_filtered[df_filtered['pChange'] < 0].copy()
        
        # Step 3: Sort by traded value (descending)
        df_gainers = df_gainers.sort_values('tradedValue', ascending=False)
        df_losers = df_losers.sort_values('tradedValue', ascending=False)
        
        # Step 4: Check trading time
        is_trading_time = self.is_trading_time()
        
        # Step 5: Process gainers - BUY when LTP crosses day high
        for _, stock in df_gainers.iterrows():
            symbol = stock['symbol']
            ltp = float(stock['ltp'])
            open_price = float(stock['open'])
            day_high = float(stock['high'])
            day_low = float(stock['low'])
            pct_change = float(stock['pChange'])
            traded_value = float(stock['tradedValue'])
            prev_traded_value = float(stock['prevTradedValue'])
            
            # Check if LTP has crossed day high (indicating breakout)
            if ltp >= day_high and ltp > 0 and open_price > 0:
                initial_sl = day_low
                sl_distance = ltp - initial_sl
                
                signal = {
                    'symbol': symbol,
                    'type': 'BUY',
                    'ltp': ltp,
                    'open': open_price,  # Target is open
                    'target': open_price,  # Target is open
                    'initial_stop_loss': initial_sl,
                    'current_stop_loss': initial_sl,
                    'sl_distance': sl_distance,
                    'day_low': day_low,
                    'day_high': day_high,
                    'pct_change': pct_change,
                    'traded_value': traded_value,
                    'prev_traded_value': prev_traded_value,
                    'traded_value_change': ((traded_value - prev_traded_value) / prev_traded_value) * 100,
                    'entry_time': datetime.now(),
                    'trailing_trigger': initial_sl + (2 * sl_distance),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day High Breakout'
                }
                signals.append(signal)
        
        # Step 6: Process losers - SELL when LTP crosses day low
        for _, stock in df_losers.iterrows():
            symbol = stock['symbol']
            ltp = float(stock['ltp'])
            open_price = float(stock['open'])
            day_high = float(stock['high'])
            day_low = float(stock['low'])
            pct_change = float(stock['pChange'])
            traded_value = float(stock['tradedValue'])
            prev_traded_value = float(stock['prevTradedValue'])
            
            # Check if LTP has crossed day low (indicating breakdown)
            if ltp <= day_low and ltp > 0 and open_price > 0:
                initial_sl = day_high
                sl_distance = initial_sl - ltp
                
                signal = {
                    'symbol': symbol,
                    'type': 'SELL',
                    'ltp': ltp,
                    'open': open_price,  # Target is open
                    'target': open_price,  # Target is open
                    'initial_stop_loss': initial_sl,
                    'current_stop_loss': initial_sl,
                    'sl_distance': sl_distance,
                    'day_low': day_low,
                    'day_high': day_high,
                    'pct_change': pct_change,
                    'traded_value': traded_value,
                    'prev_traded_value': prev_traded_value,
                    'traded_value_change': ((traded_value - prev_traded_value) / prev_traded_value) * 100,
                    'entry_time': datetime.now(),
                    'trailing_trigger': initial_sl - (2 * sl_distance),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day Low Breakdown'
                }
                signals.append(signal)
        
        return signals, df_gainers, df_losers
    
    def update_trailing_stop(self, signal, current_price):
        """Update trailing stop loss - trail for every double move of SL distance"""
        symbol = signal['symbol']
        signal_type = signal['type']
        initial_sl = signal['initial_stop_loss']
        sl_distance = signal['sl_distance']
        entry_price = signal['ltp']
        
        if signal_type == 'BUY':
            # For long: trail stop loss upward when price moves up
            profit = current_price - entry_price
            
            # Check if profit >= 2x SL distance (double move)
            if profit >= (2 * sl_distance):
                # Calculate how many "double moves" have occurred
                num_moves = int(profit / (2 * sl_distance))
                
                # New trailing stop = entry + (num_moves * sl_distance)
                new_sl = entry_price + (num_moves * sl_distance)
                
                # Only update if new SL is higher than current
                if new_sl > signal['current_stop_loss']:
                    signal['current_stop_loss'] = new_sl
                    return True, new_sl
        
        else:  # SELL
            # For short: trail stop loss downward when price moves down
            profit = entry_price - current_price
            
            # Check if profit >= 2x SL distance (double move)
            if profit >= (2 * sl_distance):
                # Calculate how many "double moves" have occurred
                num_moves = int(profit / (2 * sl_distance))
                
                # New trailing stop = entry - (num_moves * sl_distance)
                new_sl = entry_price - (num_moves * sl_distance)
                
                # Only update if new SL is lower than current
                if new_sl < signal['current_stop_loss']:
                    signal['current_stop_loss'] = new_sl
                    return True, new_sl
        
        return False, signal['current_stop_loss']

# ==================== VISUALIZATION (MODIFIED) ====================
def display_signals_table(signals, title, color, api, total_risk):
    """Display signals in a formatted table with order buttons"""
    if not signals:
        st.info(f"No {title.lower()} signals found")
        return
    
    st.markdown(f"### {title}")
    
    # Show trading time status
    now = datetime.now().time()
    start_time = dt_time(9, 25)
    end_time = dt_time(14, 30)
    is_trading_time = start_time <= now <= end_time
    
    if is_trading_time:
        st.success(f"âœ… Trading Time Active ({datetime.now().strftime('%H:%M:%S')})")
    else:
        st.warning(f"âš ï¸ Outside Trading Hours (9:25 AM - 2:30 PM) | Current: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Check if API is connected (not in demo mode)
    is_api_connected = (api is not None)
    
    for i, signal in enumerate(signals[:30]):  # Top 30
        # Color code based on trading time
        expander_status = "ðŸŸ¢" if signal['can_trade'] else "ðŸ”´"
        
        with st.expander(
            f"{expander_status} **{signal['symbol']}** | LTP: â‚¹{signal['ltp']:.2f} | Change: {signal['pct_change']:.2f}% | {signal['breakout_type']}",
            expanded=(i < 5)
        ):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price (LTP)", f"â‚¹{signal['ltp']:.2f}")
                st.metric("% Change", f"{signal['pct_change']:.2f}%", 
                          delta=f"{signal['pct_change']:.2f}%")
            
            with col2:
                st.metric("Open Price (Target)", f"â‚¹{signal['open']:.2f}")
                target_distance = abs(signal['ltp'] - signal['open'])
                target_pct = (target_distance / signal['ltp']) * 100
                st.metric("Target Distance", f"{target_pct:.2f}%")
            
            with col3:
                st.metric("Initial Stop Loss", f"â‚¹{signal['initial_stop_loss']:.2f}")
                st.metric("Current Stop Loss", f"â‚¹{signal['current_stop_loss']:.2f}")
            
            with col4:
                st.metric("Traded Value", f"â‚¹{signal['traded_value']/10000000:.2f} Cr")
                st.metric("Value Change", f"{signal['traded_value_change']:.2f}%",
                          delta=f"{signal['traded_value_change']:.2f}%")
            
            # Trade details
            st.markdown("---")
            col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
            
            with col1:
                st.write(f"**Position Type:** {signal['type']}")
            with col2:
                st.write(f"**Day Low:** â‚¹{signal['day_low']:.2f}")
            with col3:
                st.write(f"**Day High:** â‚¹{signal['day_high']:.2f}")
            with col4:
                sl_pct = abs(signal['ltp'] - signal['initial_stop_loss']) / signal['ltp'] * 100
                st.write(f"**SL Distance:** {sl_pct:.2f}%")
            
            # Additional info
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"ðŸ“Š **Prev Day Value:** â‚¹{signal['prev_traded_value']/10000000:.2f} Cr")
            with col2:
                st.success(f"ðŸ”„ **Trailing:** Activates every 2x SL distance (â‚¹{signal['sl_distance']*2:.2f})")

            # ==================== ORDER PLACEMENT UI ====================
            st.markdown("---")
            
            risk_per_share = signal['sl_distance']
            
            if risk_per_share > 0:
                calculated_quantity = int(total_risk / risk_per_share)
                calculated_quantity = max(1, calculated_quantity) # Min quantity is 1
            else:
                calculated_quantity = 1 # Default if SL distance is zero
            
            st.subheader(f"ðŸš€ Execute Trade (Auto-Qty: {calculated_quantity})")
            
            if not is_api_connected:
                st.warning("Running in Demo Mode. Connect to Definedge API to place trades.")
            else:
                trade_cols = st.columns([1, 2])
                with trade_cols[0]:
                    quantity = st.number_input(
                        f"Quantity (Risk: â‚¹{total_risk})", 
                        key=f"qty_{signal['symbol']}{i}", 
                        min_value=1, 
                        value=calculated_quantity, 
                        step=1,
                        help=f"Auto-calculated for â‚¹{total_risk} risk. (Risk/share: â‚¹{risk_per_share:.2f})"
                    )
                
                with trade_cols[1]:
                    btn_label = f"Execute {signal['type']} for {signal['symbol']}"
                    
                    # Disable button if not in trading time or API not connected
                    is_disabled = not signal['can_trade'] or not is_api_connected
                    
                    if st.button(btn_label, key=f"btn_{signal['symbol']}{i}", use_container_width=True, type="primary", disabled=is_disabled):
                        with st.spinner(f"Placing {signal['type']} order..."):
                            data, message = api.place_order(
                                symbol=signal['symbol'],
                                side=signal['type'],
                                # Ensure quantity is passed as an integer
                                quantity=int(quantity)
                            )
                            if data and data.get('status') == 'success':
                                st.success(message)
                            else:
                                # Display the error message returned by the API
                                st.error(message)

# ==================== MAIN APP (RENAMED) ====================
def show_trading_dashboard():
    """Display the main trading dashboard after authentication"""
    
    api = st.session_state.api_instance
    
    # Top Bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_id = st.session_state.api_credentials.get('user_id', 'DEMO')
        st.success(f"ðŸŸ¢ Connected | User: {user_id}")
    with col2:
        if api:
            if st.button("ðŸ”„ Check Profile", use_container_width=True):
                profile, error = api.get_profile()
                if profile:
                    st.toast(f"Welcome, {profile.get('data', {}).get('user_name', 'User')}!")
                else:
                    st.error(f"Session error: {error}")
    with col3:
        if st.button("ðŸšª Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.otp_sent = False
            st.session_state.api_credentials = {}
            st.session_state.api_instance = None
            st.session_state.access_token = None
            st.rerun()

    st.title("ðŸ“Š NSE Enhanced Trading Strategy")
    st.markdown("**All Securities | Value-Based | Breakout Strategy**")
    
    # Sidebar
    with st.sidebar:
        st.header("âš™ï¸ Enhanced Strategy Rules")
        
        st.markdown("""
        ### ðŸ“‹ Strategy Overview
        
        **Data Source:**
        - âœ… All securities (not limited to Nifty 50)
        - âœ… Sorted by traded value
        
        **Filters:**
        - ðŸ“ˆ Current traded value > Previous day
        - â° Trading time: 9:25 AM - 2:30 PM
        
        **Long (BUY):**
        - Top gainers (by traded value)
        - Entry: LTP crosses day high
        - Target: Open price
        - Stop Loss: Day low
        - Trailing: Every 2x SL distance
        
        **Short (SELL):**
        - Top losers (by traded value)
        - Entry: LTP crosses day low
        - Target: Open price
        - Stop Loss: Day high
        - Trailing: Every 2x SL distance
        """)
        
        st.markdown("---")
        
        # Current time display
        st.metric("Current Time", datetime.now().strftime('%H:%M:%S'))
        
        now = datetime.now().time()
        start_time = dt_time(9, 25)
        end_time = dt_time(14, 30)
        
        if start_time <= now <= end_time:
            st.success("âœ… Trading Hours Active")
        else:
            st.error("âŒ Outside Trading Hours")
        
        st.markdown("---")
        
        max_signals = st.slider("Max Signals per Side", 10, 100, 30)
        
        total_risk = st.number_input("Risk per Trade (Rs)", min_value=10, value=100, step=10, help="Total amount to risk per trade (e.g., Rs 100)")
        
        st.markdown("---")
        
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        scan_button = st.button("ðŸ” Fetch Live Data", use_container_width=True, type="primary")
        demo_button = st.button("ðŸ§ª Use Demo Data", use_container_width=True)
    
    # Fetch data
    if scan_button or auto_refresh or demo_button:
        with st.spinner("Fetching data..."):
            if demo_button:
                st.info("ðŸ“Š Loading demo data...")
                all_data = generate_demo_data()
                fetcher = NSEDataFetcher()
                df_all = fetcher.parse_nse_data(all_data)
            else:
                fetcher = NSEDataFetcher()
                all_data = fetcher.fetch_all_securities()
                
                if all_data is None:
                    st.error("âŒ Failed to fetch data from NSE.")
                    st.info("ðŸ’¡ Try demo mode or check back later.")
                    st.stop()
                else:
                    df_all = fetcher.parse_nse_data(all_data)
    else:
        st.info("ðŸ‘† Click 'Fetch Live Data' to get real-time NSE data or 'Use Demo Data' for sample data")
        st.markdown("""
        ### ðŸŽ¯ Enhanced Strategy Features:
        
        1. **All Securities Coverage** - Not limited to Nifty 50
        2. **Value-Based Sorting** - Arranged by traded value
        3. **Volume Filter** - Only stocks with current value > previous day
        4. **Time Window** - Trades only between 9:25 AM - 2:30 PM
        5. **Breakout Entry** - Buy on day high break, Sell on day low break
        6. **Smart Targets** - Target is open price
        7. **Dynamic Stop Loss** - Day low/high with 2x trailing
        8. **Broker Integration** - Connect to Definedge to execute trades
        """)
        st.stop()
    
    # Generate signals
    strategy = EnhancedTradingStrategy()
    signals, df_gainers, df_losers = strategy.generate_signals(df_all)
    
    # Separate buy and sell signals
    buy_signals = [s for s in signals if s['type'] == 'BUY'][:max_signals]
    sell_signals = [s for s in signals if s['type'] == 'SELL'][:max_signals]
    
    # Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Stocks", len(df_all))
    with col2:
        st.metric("Value Filtered", len(df_all[df_all['tradedValue'] > df_all['prevTradedValue']]))
    with col3:
        st.metric("Gainers", len(df_gainers))
    with col4:
        st.metric("Losers", len(df_losers))
    with col5:
        st.metric("Buy Signals", len(buy_signals))
    with col6:
        st.metric("Sell Signals", len(sell_signals))
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸŸ¢ Buy Signals",
        "ðŸ”´ Sell Signals",
        "ðŸ“Š Market Overview",
        "ðŸ“¥ Export"
    ])
    
    with tab1:
        display_signals_table(buy_signals, "ðŸŸ¢ Long Positions (Day High Breakouts)", "green", api, total_risk)
    
    with tab2:
        display_signals_table(sell_signals, "ðŸ”´ Short Positions (Day Low Breakdowns)", "red", api, total_risk)
    
    with tab3:
        st.subheader("ðŸ“Š Market Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ðŸŸ¢ Top Gainers (by Traded Value)")
            if not df_gainers.empty:
                df_display = df_gainers[['symbol', 'ltp', 'pChange', 'tradedValue', 'prevTradedValue']].head(20).copy()
                df_display['ltp'] = df_display['ltp'].apply(lambda x: f"â‚¹{float(x):.2f}")
                df_display['pChange'] = df_display['pChange'].apply(lambda x: f"{float(x):.2f}%")
                df_display['tradedValue'] = df_display['tradedValue'].apply(lambda x: f"â‚¹{float(x)/10000000:.2f} Cr")
                df_display['prevTradedValue'] = df_display['prevTradedValue'].apply(lambda x: f"â‚¹{float(x)/10000000:.2f} Cr")
                df_display.columns = ['Symbol', 'LTP', 'Change %', 'Current Value', 'Prev Value']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### ðŸ”´ Top Losers (by Traded Value)")
            if not df_losers.empty:
                df_display = df_losers[['symbol', 'ltp', 'pChange', 'tradedValue', 'prevTradedValue']].head(20).copy()
                df_display['ltp'] = df_display['ltp'].apply(lambda x: f"â‚¹{float(x):.2f}")
                df_display['pChange'] = df_display['pChange'].apply(lambda x: f"{float(x):.2f}%")
                df_display['tradedValue'] = df_display['tradedValue'].apply(lambda x: f"â‚¹{float(x)/10000000:.2f} Cr")
                df_display['prevTradedValue'] = df_display['prevTradedValue'].apply(lambda x: f"â‚¹{float(x)/10000000:.2f} Cr")
                df_display.columns = ['Symbol', 'LTP', 'Change %', 'Current Value', 'Prev Value']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("ðŸ“¥ Export Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if buy_signals:
                df_buy_export = pd.DataFrame([{
                    'Symbol': s['symbol'],
                    'Type': s['type'],
                    'LTP': s['ltp'],
                    'Open (Target)': s['open'],
                    'Stop Loss': s['initial_stop_loss'],
                    'Day High': s['day_high'],
                    'Day Low': s['day_low'],
                    '% Change': s['pct_change'],
                    'Traded Value (Cr)': s.get('traded_value', 0)/10000000,
                    'Prev Value (Cr)': s.get('prev_traded_value', 0)/10000000,
                    'Value Change %': s.get('traded_value_change', 0),
                    'Breakout Type': s['breakout_type'],
                    'Can Trade': 'Yes' if s['can_trade'] else 'No'
                } for s in buy_signals])
                
                csv_buy = df_buy_export.to_csv(index=False)
                st.download_button(
                    "ðŸ“¥ Download Buy Signals (CSV)",
                    csv_buy,
                    f"buy_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col2:
            if sell_signals:
                df_sell_export = pd.DataFrame([{
                    'Symbol': s['symbol'],
                    'Type': s['type'],
                    'LTP': s['ltp'],
                    'Open (Target)': s['open'],
                    'Stop Loss': s['initial_stop_loss'],
                    'Day High': s['day_high'],
                    'Day Low': s['day_low'],
                    '% Change': s['pct_change'],
                    'Traded Value (Cr)': s.get('traded_value', 0)/10000000,
                    'Prev Value (Cr)': s.get('prev_traded_value', 0)/10000000,
                    'Value Change %': s.get('traded_value_change', 0),
                    'Breakout Type': s['breakout_type'],
                    'Can Trade': 'Yes' if s['can_trade'] else 'No'
                } for s in sell_signals])
                
                csv_sell = df_sell_export.to_csv(index=False)
                st.download_button(
                    "ðŸ“¥ Download Sell Signals (CSV)",
                    csv_sell,
                    f"sell_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: NSE India | Broker: Definedge Securities")
    st.caption("âš ï¸ This is for educational purposes only. Not financial advice.")

# ==================== MAIN ENTRY POINT (NEW) ====================
def main():
    """Main application entry point: handles auth vs dashboard"""
    if not st.session_state.get('authenticated', False):
        show_authentication_page()
    else:
        show_trading_dashboard()

if __name__ == "__main__":
    main()
