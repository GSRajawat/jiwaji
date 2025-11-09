import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time
import time
import json
import os
from supabase import create_client, Client
from datetime import timedelta
from datetime import timezone


IST = timezone(timedelta(hours=5, minutes=30))

# ==========================================
# SUPABASE DATABASE CREDENTIALS
# ==========================================
# Create a free account at: https://supabase.com/
# Create a new project and get these credentials


try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except KeyError:
    st.error("❌ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY in Streamlit's secrets.")
    supabase = None
# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NSE Enhanced Trading Strategy",
    page_icon="📊",
    layout="wide"
)

# ==================== CSV FALLBACK PATH ====================
# Used for downloading/uploading
LOCAL_CSV_FILE = "trades_database.csv"

# ==================== SUPABASE DB FUNCTIONS ====================

def save_trade_to_db(trade_data):
    """Save a new trade to the Supabase database"""
    if supabase is None:
        st.error("Supabase client not initialized.")
        return

    # Ensure nullable fields are None, not NaN
    for key in ['exit_order_id', 'exit_price', 'exit_time', 'pnl']:
        if key not in trade_data:
            trade_data[key] = None
    
    try:
        response = supabase.table("trades").insert(trade_data).execute()
        if len(response.data) == 0:
            st.error(f"Failed to save trade: {response}")
    except Exception as e:
        st.error(f"Error saving trade to Supabase: {e}")

def update_trade_exit(entry_order_id, exit_order_id, exit_price, exit_time, pnl):
    """Update trade with exit information in Supabase"""
    if supabase is None:
        st.error("Supabase client not initialized.")
        return

    update_data = {
        'exit_order_id': exit_order_id,
        'exit_price': exit_price,
        'exit_time': exit_time,
        'pnl': pnl,
        'status': 'CLOSED'
    }
    
    try:
        response = supabase.table("trades").update(update_data).eq("entry_order_id", entry_order_id).execute()
        if len(response.data) == 0:
            st.warning(f"Trade {entry_order_id} not found or failed to update.")
    except Exception as e:
        st.error(f"Error updating trade in Supabase: {e}")

def get_open_trades():
    """Get all open trades from Supabase"""
    if supabase is None:
        st.error("Supabase client not initialized.")
        return pd.DataFrame()

    try:
        response = supabase.table("trades").select("*").eq("status", "OPEN").execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching open trades: {e}")
        return pd.DataFrame()

def get_all_trades():
    """Get all trades from Supabase"""
    if supabase is None:
        st.error("Supabase client not initialized.")
        return pd.DataFrame()

    try:
        response = supabase.table("trades").select("*").order("entry_time", desc=True).execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching all trades: {e}")
        return pd.DataFrame()

# ==================== SESSION STATE ====================
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
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trailing_stops' not in st.session_state:
    st.session_state.trailing_stops = {}
if 'last_data' not in st.session_state:
    st.session_state.last_data = pd.DataFrame()
if 'auto_exit_enabled' not in st.session_state:
    st.session_state.auto_exit_enabled = False
if 'auto_entry_enabled' not in st.session_state:
    st.session_state.auto_entry_enabled = False
if 'min_traded_value_cr' not in st.session_state:
    st.session_state.min_traded_value_cr = 500

# ==================== NSE DATA FETCHER ====================
class NSEDataFetcher:
    """Fetch live data from NSE India"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_cookies(self):
        try:
            response = self.session.get(self.base_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def fetch_all_securities(self):
        try:
            self.get_cookies()
            time.sleep(1)
            
            url = f"{self.base_url}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                st.success("✅ Fetched F&O securities from NSE")
                return response.json()
            
            st.error(f"❌ API failed ({response.status_code})")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def parse_nse_data(self, nse_data):
        try:
            if not nse_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(nse_data.get('data', []))
            
            if df.empty:
                st.warning("⚠️ No data in response")
                return pd.DataFrame()
            
            column_mapping = {
                'lastPrice': 'ltp',
                'previousClose': 'prev_close',
                'dayHigh': 'high',
                'dayLow': 'low',
                'totalTradedValue': 'tradedValue',
                'perChange': 'pChange'
            }
            df = df.rename(columns=column_mapping)
            
            required = ['symbol', 'ltp', 'open', 'high', 'low', 'tradedValue']
            for col in required:
                if col not in df.columns:
                    if col == 'open' and 'previousClose' in df.columns:
                        df['open'] = df['previousClose']
                    elif col not in df.columns:
                        st.error(f"❌ Missing column: {col}")
                        return pd.DataFrame()
            
            numeric_cols = ['ltp', 'open', 'high', 'low', 'tradedValue']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'pChange' not in df.columns and 'prev_close' in df.columns:
                df['prev_close'] = pd.to_numeric(df['prev_close'], errors='coerce')
                df['pChange'] = ((df['ltp'] - df['prev_close']) / df['prev_close']) * 100
            elif 'pChange' not in df.columns:
                df['pChange'] = ((df['ltp'] - df['open']) / df['open']) * 100
            else:
                df['pChange'] = pd.to_numeric(df['pChange'], errors='coerce')
            
            df = df.dropna(subset=['ltp', 'open', 'high', 'low', 'tradedValue'])
            df = df[df['ltp'] > 0]
            df = df[df['tradedValue'] > 0]
            df['pChange'] = df['pChange'].fillna(0)
            
            st.success(f"✅ Parsed {len(df)} stocks from NSE")
            return df
            
        except Exception as e:
            st.error(f"❌ Parsing error: {str(e)}")
            return pd.DataFrame()

# ==================== DEFINEDGE API ====================
class DefinedgeAPI:
    """Definedge Securities API Wrapper"""
    
    def __init__(self):
        self.auth_base_url = "https://signin.definedgesecurities.com/auth/realms/debroking/dsbpkc"
        self.api_base_url = "https://integrate.definedgesecurities.com"
        self.user_id = None
        self.api_token = None
        self.api_secret = None
        self.otp_token = None
        self.access_token = None
    
    def set_credentials(self, user_id, api_token, api_secret):
        self.user_id = user_id
        self.api_token = api_token
        self.api_secret = api_secret
    
    def generate_otp(self):
        try:
            url = f"{self.auth_base_url}/login/{self.api_token}"
            headers = {
                "api_secret": self.api_secret,
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.otp_token = data.get('otp_token')
                message = data.get('message', 'OTP sent')
                
                if self.otp_token:
                    return True, message, self.otp_token
                return False, "OTP token not found", None
            return False, f"Error {response.status_code}", None
        except Exception as e:
            return False, f"Error: {str(e)}", None
    
    def verify_otp_and_authenticate(self, otp, otp_token):
        try:
            url = f"{self.auth_base_url}/token"
            headers = {"Content-Type": "application/json"}
            payload = {"otp_token": otp_token, "otp": otp}
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('api_session_key') or data.get('susertoken')
                
                if self.access_token:
                    return True, self.access_token
                return False, "Session key not found"
            return False, f"Error {response.status_code}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_quotes(self, symbols):
        """Get live quotes from Definedge API"""
        if not self.access_token:
            return None
        
        try:
            url = f"{self.api_base_url}/dart/v1/quotes"
            headers = {
                "Authorization": self.access_token,
                "Content-Type": "application/json"
            }
            
            formatted_symbols = [self._format_symbol(s, "NSE") for s in symbols]
            
            payload = {
                "exchange": "NSE",
                "tradingsymbols": formatted_symbols
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS':
                    return data.get('quotes', [])
            
            return None
        except Exception as e:
            st.error(f"Error fetching quotes: {str(e)}")
            return None
    
    def place_order(self, symbol, side, quantity, exchange="NSE", product="INTRADAY", order_type="MARKET", price=0, trigger_price=0):
        """Place order with proper symbol formatting"""
        if not self.access_token:
            return None, "❌ Not authenticated"
        
        try:
            formatted_symbol = self._format_symbol(symbol, exchange)
            
            url = f"{self.api_base_url}/dart/v1/placeorder"
            headers = {
                "Authorization": self.access_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "exchange": exchange,
                "tradingsymbol": formatted_symbol,
                "order_type": side,
                "quantity": str(quantity),
                "product_type": product,
                "price_type": order_type,
                "price": str(price) if order_type in ["LIMIT", "SL-LIMIT"] else "0",
                "validity": "DAY"
            }
            
            if order_type in ["SL-LIMIT", "SL-MARKET"]:
                payload["trigger_price"] = str(trigger_price)
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            try:
                data = response.json()
            except:
                data = {"error": "Invalid response", "status_code": response.status_code}
            
            if response.status_code == 200 and data.get('status') == 'SUCCESS':
                return data, f"✅ Order {data.get('order_id')} placed!"
            else:
                return data, f"❌ Failed: {data.get('message', 'Unknown error')}"
        
        except Exception as e:
            return None, f"❌ Exception: {str(e)}"
    
    def _format_symbol(self, symbol, exchange):
        """Format trading symbol for API"""
        clean = symbol.replace("-EQ", "").strip()
        
        if exchange == "NSE":
            is_derivative = any(x in clean.upper() for x in ["FUT", "CE", "PE"]) or any(c.isdigit() for c in clean)
            
            if not is_derivative:
                return f"{clean}-EQ"
        
        return clean

# ==================== DEFINEDGE REPORTS ====================
class DefinedgeReports:
    """Handles fetching orders, trades, and positions"""
    
    def __init__(self, access_token, api_base_url):
        self.access_token = access_token
        self.api_base_url = api_base_url

    def orders(self):
        """Fetches the Order Book"""
        if not self.access_token:
            return False, "Not authenticated"
        
        try:
            url = f"{self.api_base_url}/dart/v1/orders"
            headers = {"Authorization": self.access_token}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS':
                    return True, data.get('orders', [])
                else:
                    return False, f"API Error: {data.get('message', 'Unknown error')}"
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    def trades(self):
        """Fetches the Trade Book"""
        if not self.access_token:
            return False, "Not authenticated"
        
        try:
            url = f"{self.api_base_url}/dart/v1/trades"
            headers = {"Authorization": self.access_token}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS':
                    return True, data.get('trades', [])
                else:
                    return False, f"API Error: {data.get('message', 'Unknown error')}"
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    def positions(self):
        """Fetches current open Positions"""
        if not self.access_token:
            return False, "Not authenticated"
        
        try:
            url = f"{self.api_base_url}/dart/v1/positions"
            headers = {"Authorization": self.access_token}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS':
                    return True, data.get('positions', [])
                else:
                    return False, f"API Error: {data.get('message', 'Unknown error')}"
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def limits(self):
        """Fetches fund and margin limits"""
        if not self.access_token:
            return False, "Not authenticated"
        
        try:
            url = f"{self.api_base_url}/dart/v1/limits"
            headers = {"Authorization": self.access_token}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'SUCCESS':
                    return True, data
                else:
                    return False, f"API Error: {data.get('message', 'Unknown error')}"
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Exception: {str(e)}"

# ==================== AUTO EXIT MONITOR ====================
def monitor_and_exit_positions(api, reports):
    """Monitor open positions and exit based on target/stop loss"""
    if not st.session_state.auto_exit_enabled:
        return
    
    open_trades = get_open_trades()
    
    if open_trades.empty:
        return
    
    success, positions = reports.positions()
    
    if not success or not positions:
        return
    
    position_dict = {}
    for pos in positions:
        symbol = pos.get('tradingsymbol', '')
        ltp = float(pos.get('ltp', 0))
        qty = int(pos.get('net_quantity', 0))
        position_dict[symbol] = {'ltp': ltp, 'qty': qty}
    
    for idx, trade in open_trades.iterrows():
        symbol = trade['symbol']
        side = trade['side']
        target = float(trade['target_price'])
        stop_loss = float(trade['stop_loss'])
        entry_price = float(trade['entry_price'])
        quantity = int(trade['quantity'])
        entry_order_id = trade['entry_order_id']
        
        pos_data = position_dict.get(symbol, {})
        ltp = pos_data.get('ltp', 0)
        pos_qty = pos_data.get('qty', 0)
        
        if ltp == 0 or pos_qty == 0:
            continue
        
        should_exit = False
        exit_reason = ""
        
        if side == 'BUY':
            if ltp >= target:
                should_exit = True
                exit_reason = "Target Hit"
            elif ltp <= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss Hit"
        else:
            if ltp <= target:
                should_exit = True
                exit_reason = "Target Hit"
            elif ltp >= stop_loss:
                should_exit = True
                exit_reason = "Stop Loss Hit"
        
        if should_exit:
            exit_side = 'SELL' if side == 'BUY' else 'BUY'
            
            data, msg = api.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                exchange="NSE",
                order_type="MARKET"
            )
            
            if data and data.get('status') == 'SUCCESS':
                exit_order_id = data.get('order_id')
                pnl = (ltp - entry_price) * quantity if side == 'BUY' else (entry_price - ltp) * quantity
                
                update_trade_exit(
                    entry_order_id=entry_order_id,
                    exit_order_id=exit_order_id,
                    exit_price=ltp,
                    exit_time=datetime.now(IST).isoformat(),
                    pnl=pnl
                )
                
                st.success(f"🎯 Auto Exit: {symbol} - {exit_reason} | Order: {exit_order_id} | P&L: ₹{pnl:.2f}")

def check_order_executed(reports, order_id):
    """Check if order is executed"""
    success, trades = reports.trades()
    
    if not success or not trades:
        return False, 0
    
    for trade in trades:
        if trade.get('order_id') == order_id:
            return True, float(trade.get('fill_price', 0))
    
    return False, 0

# ==================== AUTHENTICATION UI ====================
def show_authentication_page():
    st.title("🔐 Definedge Securities Login")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Quick Start
        1. Enter API credentials
        2. Generate OTP
        3. Enter OTP and login
        """)
    
    with col2:
        with st.form("creds"):
            st.markdown("### 📋 API Credentials")
            user_id = st.text_input("User ID")
            api_token = st.text_input("API Token", type="password")
            api_secret = st.text_input("API Secret", type="password")
            
            if st.form_submit_button("💾 Save"):
                if user_id and api_token and api_secret:
                    st.session_state.api_credentials = {
                        'user_id': user_id,
                        'api_token': api_token,
                        'api_secret': api_secret
                    }
                    st.success("✅ Saved! Generate OTP below.")
                else:
                    st.error("Fill all fields")
    
    if st.session_state.api_credentials and not st.session_state.otp_sent:
        st.markdown("---")
        if st.button("📤 Generate OTP", use_container_width=True, type="primary"):
            api = DefinedgeAPI()
            api.set_credentials(
                st.session_state.api_credentials['user_id'],
                st.session_state.api_credentials['api_token'],
                st.session_state.api_credentials['api_secret']
            )
            
            success, message, token = api.generate_otp()
            if success:
                st.session_state.otp_sent = True
                st.session_state.temp_otp_token = token
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")
    
    if st.session_state.otp_sent:
        st.markdown("---")
        with st.form("otp"):
            st.markdown("### 🔒 Enter OTP")
            otp = st.text_input("6-digit OTP")
            
            if st.form_submit_button("✅ Verify & Login", use_container_width=True):
                if otp and len(str(otp)) >= 4:
                    api = DefinedgeAPI()
                    api.set_credentials(
                        st.session_state.api_credentials['user_id'],
                        st.session_state.api_credentials['api_token'],
                        st.session_state.api_credentials['api_secret']
                    )
                    
                    success, token = api.verify_otp_and_authenticate(otp, st.session_state.temp_otp_token)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.access_token = token
                        api.access_token = token
                        st.session_state.api_instance = api
                        st.session_state.otp_sent = False
                        st.success("✅ Logged in!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {token}")
                else:
                    st.error("Enter a valid OTP")

# ==================== TRADING STRATEGY ====================
class EnhancedTradingStrategy:
    def __init__(self):
        self.positions = []
        self.trailing_stops = {}
    
    def is_trading_time(self):
        now = datetime.now(IST).time()
        return dt_time(9, 15) <= now <= dt_time(23, 59)
    
    def filter_by_traded_value(self, df, min_value_cr):
        if df.empty:
            return df
        min_value = min_value_cr * 1e7
        return df[df['tradedValue'] > min_value].copy()
    
    def generate_signals(self, df_nse, api, min_traded_value_cr):
        """Generate signals using NSE data for filtering and Definedge for prices"""
        signals = []
        
        if df_nse.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        df_filtered = self.filter_by_traded_value(df_nse, min_traded_value_cr)
        
        if df_filtered.empty:
            st.warning(f"⚠️ No stocks with traded value > {min_traded_value_cr} crore")
            return signals, pd.DataFrame(), pd.DataFrame()
        
        symbols = df_filtered['symbol'].tolist()
        quotes = api.get_quotes(symbols)
        
        if not quotes:
            st.warning("⚠️ Could not fetch live prices from Definedge API")
            return signals, pd.DataFrame(), pd.DataFrame()
        
        price_dict = {}
        for quote in quotes:
            symbol = quote.get('tradingsymbol', '').replace('-EQ', '')
            price_dict[symbol] = {
                'ltp': float(quote.get('ltp', 0)),
                'open': float(quote.get('open', 0)),
                'high': float(quote.get('high', 0)),
                'low': float(quote.get('low', 0))
            }
        
        for idx, stock in df_filtered.iterrows():
            symbol = stock['symbol']
            prices = price_dict.get(symbol)
            
            if not prices or prices['ltp'] == 0:
                continue
            
            stock['ltp'] = prices['ltp']
            stock['open'] = prices['open']
            stock['high'] = prices['high']
            stock['low'] = prices['low']
            stock['pChange'] = ((stock['ltp'] - stock['open']) / stock['open']) * 100 if stock['open'] > 0 else 0
            
            df_filtered.loc[idx] = stock
        
        df_gainers = df_filtered[df_filtered['pChange'] > 0].copy().sort_values('tradedValue', ascending=False)
        df_losers = df_filtered[df_filtered['pChange'] < 0].copy().sort_values('tradedValue', ascending=False)
        
        is_trading_time = self.is_trading_time()
        
        for _, stock in df_gainers.iterrows():
            if stock['ltp'] >= stock['high']:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'BUY',
                    'ltp': float(stock['ltp']),
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['low']),
                    'current_stop_loss': float(stock['low']),
                    'sl_distance': float(stock['ltp'] - stock['low']),
                    'day_low': float(stock['low']),
                    'day_high': float(stock['high']),
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day High Breakout'
                })
        
        for _, stock in df_losers.iterrows():
            if stock['ltp'] <= stock['low']:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'SELL',
                    'ltp': float(stock['ltp']),
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['high']),
                    'current_stop_loss': float(stock['high']),
                    'sl_distance': float(stock['high'] - stock['ltp']),
                    'day_low': float(stock['low']),
                    'day_high': float(stock['high']),
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day Low Breakdown'
                })
        
        return signals, df_gainers, df_losers

# ==================== SIGNAL DISPLAY ====================
def display_signals_table(signals, title, api, reports, total_risk, auto_entry):
    if not signals:
        st.info(f"No {title.lower()} found")
        return
    
    st.markdown(f"### {title}")
    
    now = datetime.now(IST).time()
    is_trading_time = dt_time(9, 15) <= now <= dt_time(23, 59)
    
    if is_trading_time:
        st.success(f"✅ Trading Active ({datetime.now(IST).strftime('%H:%M:%S')})")
    else:
        st.warning(f"⚠️ Outside Hours | Current: {datetime.now(IST).strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    success_pos, positions = reports.positions()
    position_dict = {}
    if success_pos and positions:
        for pos in positions:
            symbol = pos.get('tradingsymbol', '')
            qty = int(pos.get('net_quantity', 0))
            position_dict[symbol] = qty
    
    open_trades = get_open_trades()
    
    for i, signal in enumerate(signals[:30]):
        status = "🟢" if signal['can_trade'] else "🔴"
        
        with st.expander(
            f"{status} {signal['symbol']} | ₹{signal['ltp']:.2f} | {signal['pct_change']:.2f}% | {signal['breakout_type']}",
            expanded=(i < 5)
        ):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("LTP", f"₹{signal['ltp']:.2f}")
                st.metric("% Change", f"{signal['pct_change']:.2f}%")
            
            with col2:
                st.metric("Target (Open)", f"₹{signal['open']:.2f}")
                target_pct = abs(signal['ltp'] - signal['open']) / signal['ltp'] * 100
                st.metric("Target Dist", f"{target_pct:.2f}%")
            
            with col3:
                st.metric("Stop Loss", f"₹{signal['initial_stop_loss']:.2f}")
                sl_pct = signal['sl_distance'] / signal['ltp'] * 100
                st.metric("SL Dist", f"{sl_pct:.2f}%")
            
            with col4:
                st.metric("Traded Value", f"₹{signal['traded_value']/1e7:.2f} Cr")
            
            st.markdown("---")
            
            risk_per_share = signal['sl_distance']
            calc_qty = int(total_risk / risk_per_share) if risk_per_share > 0 else 1
            calc_qty = max(1, calc_qty)
            
            formatted_sym = api._format_symbol(signal['symbol'], "NSE")
            
            if auto_entry and signal['can_trade']:
                symbol_in_db = False
                if not open_trades.empty:
                    symbol_in_db = any(open_trades['symbol'] == formatted_sym)
                
                if not symbol_in_db:
                    st.info(f"🤖 Auto-entry enabled. Placing order...")
                    
                    data, msg = api.place_order(
                        symbol=signal['symbol'],
                        side=signal['type'],
                        quantity=calc_qty,
                        exchange="NSE"
                    )
                    
                    if data and data.get('status') == 'SUCCESS':
                        entry_order_id = data.get('order_id')
                        
                        time.sleep(2)
                        is_executed, exec_price = check_order_executed(reports, entry_order_id)
                        
                        if is_executed:
                            trade_data = {
                                'entry_order_id': entry_order_id,
                                'symbol': formatted_sym,
                                'side': signal['type'],
                                'quantity': calc_qty,
                                'entry_price': exec_price if exec_price > 0 else signal['ltp'],
                                'target_price': signal['target'],
                                'stop_loss': signal['initial_stop_loss'],
                                'entry_time': datetime.now(IST).isoformat(),
                                'status': 'OPEN'
                            }
                            
                            save_trade_to_db(trade_data)
                            st.success(f"✅ {msg} - Trade executed and saved to database!")
                        else:
                            st.warning(f"⏳ Order {entry_order_id} placed but not yet executed")
                    else:
                        st.error(f"❌ Auto-entry failed: {msg}")
            
            st.subheader(f"🚀 Manual Trade Controls")
            
            st.info(f"📋 Symbol: `{signal['symbol']}` → API Format: `{formatted_sym}`")
            
            col_qty, col_manual = st.columns([1, 2])
            
            with col_qty:
                qty = st.number_input(
                    f"Qty (Risk: ₹{total_risk})",
                    key=f"q_{title}_{i}",
                    min_value=1,
                    value=calc_qty,
                    help=f"Auto: ₹{total_risk} ÷ ₹{risk_per_share:.2f}"
                )
            
            with col_manual:
                manual = st.checkbox("Override Symbol", key=f"m_{title}_{i}")
                if manual:
                    custom_sym = st.text_input(
                        "Trading Symbol",
                        value=formatted_sym,
                        key=f"s_{title}_{i}",
                        help="Exact symbol from Definedge master file"
                    )
                else:
                    custom_sym = None
            
            col_target, col_sl = st.columns(2)
            with col_target:
                target_price = st.number_input(
                    "Target Price",
                    key=f"tgt_{title}_{i}",
                    value=float(signal['target']),
                    step=0.05,
                    format="%.2f"
                )
            
            with col_sl:
                sl_price = st.number_input(
                    "Stop Loss",
                    key=f"sl_{title}_{i}",
                    value=float(signal['initial_stop_loss']),
                    step=0.05,
                    format="%.2f"
                )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                disabled = not signal['can_trade']
                
                if st.button(f"📤 Place Entry Order", key=f"b_{title}_{i}", type="primary", disabled=disabled, use_container_width=True):
                    sym_to_use = custom_sym if manual and custom_sym else signal['symbol']
                    
                    with st.spinner(f"Placing {signal['type']} order..."):
                        data, msg = api.place_order(
                            symbol=sym_to_use,
                            side=signal['type'],
                            quantity=int(qty),
                            exchange="NSE"
                        )
                        
                        if data and data.get('status') == 'SUCCESS':
                            entry_order_id = data.get('order_id')
                            st.success(f"✅ {msg}")
                            
                            time.sleep(2)
                            is_executed, exec_price = check_order_executed(reports, entry_order_id)
                            
                            if is_executed:
                                trade_data = {
                                    'entry_order_id': entry_order_id,
                                    'symbol': formatted_sym,
                                    'side': signal['type'],
                                    'quantity': qty,
                                    'entry_price': exec_price if exec_price > 0 else signal['ltp'],
                                    'target_price': target_price,
                                    'stop_loss': sl_price,
                                    'entry_time': datetime.now(IST).isoformat(),
                                    'status': 'OPEN'
                                }
                                
                                save_trade_to_db(trade_data)
                                st.info(f"📊 Trade executed and saved! Auto-exit enabled for Target: ₹{target_price:.2f} | SL: ₹{sl_price:.2f}")
                            else:
                                st.warning(f"⏳ Order placed but not yet executed. Check order book.")
                        else:
                            st.error(msg)
            
            with col_btn2:
                sym_to_check = custom_sym if manual and custom_sym else formatted_sym
                has_position = position_dict.get(sym_to_check, 0) > 0
                
                has_db_entry = False
                if not open_trades.empty:
                    has_db_entry = any(open_trades['symbol'] == sym_to_check)
                
                can_exit = has_position and has_db_entry
                
                if st.button(
                    f"🔥 Place Exit Order", 
                    key=f"exit_{title}_{i}", 
                    type="secondary", 
                    disabled=not can_exit, 
                    use_container_width=True,
                    help="Exit enabled only for traded positions in database"
                ):
                    sym_to_use = custom_sym if manual and custom_sym else signal['symbol']
                    exit_side = 'SELL' if signal['type'] == 'BUY' else 'BUY'
                    
                    with st.spinner(f"Placing {exit_side} order..."):
                        data, msg = api.place_order(
                            symbol=sym_to_use,
                            side=exit_side,
                            quantity=int(qty),
                            exchange="NSE"
                        )
                        
                        if data and data.get('status') == 'SUCCESS':
                            st.success(f"✅ Exit {msg}")
                            
                            trade_row = open_trades[open_trades['symbol'] == sym_to_check].iloc[0]
                            exit_price = signal['ltp']
                            pnl = (exit_price - trade_row['entry_price']) * trade_row['quantity'] if trade_row['side'] == 'BUY' else (trade_row['entry_price'] - exit_price) * trade_row['quantity']
                            
                            update_trade_exit(
                                entry_order_id=trade_row['entry_order_id'],
                                exit_order_id=data.get('order_id'),
                                exit_price=exit_price,
                                exit_time=datetime.now(IST).isoformat(),
                                pnl=pnl
                            )
                            st.info(f"💰 P&L: ₹{pnl:.2f}")
                        else:
                            st.error(msg)

# ==================== MAIN DASHBOARD ====================
def show_trading_dashboard():
    api = st.session_state.api_instance
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user = st.session_state.api_credentials.get('user_id', 'User')
        st.success(f"🟢 Logged in as: {user}")
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
    
    st.title("📊 NSE Enhanced Trading Strategy")
    st.markdown("**Breakout Strategy | Auto Entry/Exit | Value-Based | F&O Securities**")
    
    if api is None:
        st.error("API instance not initialized. Please log in.")
        return
    
    if supabase is None:
        st.error("Supabase client not initialized. Check credentials.")
        return
        
    reports = DefinedgeReports(api.access_token, api.api_base_url)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("""
        ### Strategy
        - Entry: LTP crosses day high/low
        - Target: Open price (editable)
        - Stop: Day low/high (editable)
        - Auto Exit: Monitors positions
        """)
        
        st.markdown("---")
        
        st.subheader("💰 Funds & Margin")
        success, limits_data = reports.limits()
        if success:
            cash = limits_data.get('cash', 0)
            margin_used = limits_data.get('margin_used', 0)
            margin_available = limits_data.get('margin_available', 0)
            
            st.metric("Available Cash", f"₹{float(cash):,.2f}")
            st.metric("Margin Used", f"₹{float(margin_used):,.2f}")
            st.metric("Margin Available", f"₹{float(margin_available):,.2f}")
        else:
            st.warning("Unable to fetch limits")
        
        st.markdown("---")
        st.metric("Time", datetime.now(IST).strftime('%H:%M:%S'))
        
        now = datetime.now(IST).time()
        if dt_time(9, 15) <= now <= dt_time(23, 59):
            st.success("✅ Trading Hours")
        else:
            st.error("❌ Outside Hours")
        
        st.markdown("---")
        
        st.subheader("📊 Filters")
        min_traded_value = st.number_input(
            "Min Traded Value (Crore)", 
            min_value=10, 
            max_value=5000, 
            value=st.session_state.min_traded_value_cr,
            step=50
        )
        st.session_state.min_traded_value_cr = min_traded_value
        
        st.markdown("---")
        
        st.subheader("🤖 Auto Entry")
        auto_entry = st.checkbox("Enable Auto Entry", value=st.session_state.auto_entry_enabled)
        st.session_state.auto_entry_enabled = auto_entry
        
        if auto_entry:
            st.success("✅ Auto-entry enabled")
            st.warning("⚠️ Orders will be placed automatically for new signals")
        else:
            st.info("ℹ️ Manual entry mode")
        
        st.markdown("---")
        
        st.subheader("🤖 Auto Exit")
        auto_exit = st.checkbox("Enable Auto Exit", value=st.session_state.auto_exit_enabled)
        st.session_state.auto_exit_enabled = auto_exit
        
        if auto_exit:
            st.success("✅ Auto-exit monitoring enabled")
        else:
            st.warning("⚠️ Auto-exit disabled")
        
        st.markdown("---")
        max_signals = st.slider("Max Signals", 10, 100, 30)
        total_risk = st.number_input("Risk/Trade (₹)", 10, 10000, 100, 10)
        
        st.markdown("---")
        auto_refresh = st.checkbox("Auto Refresh (30s)")
        scan = st.button("🔍 Fetch Live", use_container_width=True, type="primary")
    
    if st.session_state.auto_exit_enabled:
        monitor_and_exit_positions(api, reports)
    
    df_all = st.session_state.last_data

    if scan or auto_refresh:
        with st.spinner("Fetching from NSE..."):
            fetcher = NSEDataFetcher()
            data = fetcher.fetch_all_securities()
            if data is None:
                st.error("❌ Failed to fetch from NSE.")
                new_df_all = pd.DataFrame()
            else:
                new_df_all = fetcher.parse_nse_data(data)
        
        if not new_df_all.empty:
            st.session_state.last_data = new_df_all
            df_all = new_df_all
        else:
            st.warning("New fetch returned no data.")

    if df_all.empty:
        if not (scan or auto_refresh):
            st.info("👆 Click 'Fetch Live' to start the analysis.")
        return

    strategy = EnhancedTradingStrategy()
    signals, gainers, losers = strategy.generate_signals(df_all, api, st.session_state.min_traded_value_cr)
    
    buy_signals = [s for s in signals if s['type'] == 'BUY'][:max_signals]
    sell_signals = [s for s in signals if s['type'] == 'SELL'][:max_signals]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Total Stocks", len(df_all))
    col2.metric("Value Filtered", len(strategy.filter_by_traded_value(df_all, st.session_state.min_traded_value_cr)))
    col3.metric("Gainers", len(gainers))
    col4.metric("Losers", len(losers))
    col5.metric("Buy Signals", len(buy_signals))
    col6.metric("Sell Signals", len(sell_signals))
        
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🟢 Buy", 
        "🔴 Sell", 
        "📊 Overview", 
        "📜 Order Book", 
        "📈 Trade Book", 
        "💼 Positions",
        "🗄️ Trade Database"
    ])
    
    with tab1:
        display_signals_table(buy_signals, "🟢 Long (Day High Breakout)", api, reports, total_risk, st.session_state.auto_entry_enabled)
    
    with tab2:
        display_signals_table(sell_signals, "🔴 Short (Day Low Breakdown)", api, reports, total_risk, st.session_state.auto_entry_enabled)
    
    with tab3:
        st.subheader("📊 Market Overview (NSE Data)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Top Gainers")
            if not gainers.empty:
                display_df = gainers[['symbol', 'ltp', 'pChange', 'tradedValue']].head(20).copy()
                display_df['ltp'] = display_df['ltp'].apply(lambda x: f"₹{x:.2f}")
                display_df['pChange'] = display_df['pChange'].apply(lambda x: f"{x:.2f}%")
                display_df['tradedValue'] = display_df['tradedValue'].apply(lambda x: f"₹{x/1e7:.2f} Cr")
                display_df.columns = ['Symbol', 'LTP', 'Change %', 'Value']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No gainers to show.")
        
        with col2:
            st.markdown("### 🔴 Top Losers")
            if not losers.empty:
                display_df = losers[['symbol', 'ltp', 'pChange', 'tradedValue']].head(20).copy()
                display_df['ltp'] = display_df['ltp'].apply(lambda x: f"₹{x:.2f}")
                display_df['pChange'] = display_df['pChange'].apply(lambda x: f"{x:.2f}%")
                display_df['tradedValue'] = display_df['tradedValue'].apply(lambda x: f"₹{x/1e7:.2f} Cr")
                display_df.columns = ['Symbol', 'LTP', 'Change %', 'Value']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No losers to show.")

    with tab4:
        st.header("📜 Order Book")
        st.caption("All orders placed today")
        if st.button("Refresh Order Book", key="refresh_orders"):
            st.rerun()
            
        success, data = reports.orders()
        if success:
            if data:
                df = pd.DataFrame(data)
                
                def style_status(s):
                    if s == 'COMPLETE':
                        return 'background-color: #28a745; color: white;'
                    elif s == 'OPEN':
                        return 'background-color: #ffc107; color: black;'
                    elif s == 'REJECTED':
                        return 'background-color: #dc3545; color: white;'
                    elif s == 'CANCELLED':
                        return 'background-color: #6c757d; color: white;'
                    return ''
                
                if 'order_status' in df.columns:
                    st.dataframe(
                        df.style.applymap(style_status, subset=['order_status']),
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Order book is empty.")
        else:
            st.error(f"Failed to load order book: {data}")

    with tab5:
        st.header("📈 Trade Book")
        st.caption("All executed trades for today")
        if st.button("Refresh Trade Book", key="refresh_trades"):
            st.rerun()

        success, data = reports.trades()
        if success:
            if data:
                df = pd.DataFrame(data)
                
                def style_side(s):
                    return 'color: green' if s == 'BUY' else 'color: red'
                
                if 'order_type' in df.columns:
                    st.dataframe(
                        df.style.applymap(style_side, subset=['order_type']),
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Trade book is empty.")
        else:
            st.error(f"Failed to load trade book: {data}")

    with tab6:
        st.header("💼 Positions")
        st.caption("Your current open positions")
        if st.button("Refresh Positions", key="refresh_positions"):
            st.rerun()

        success, data = reports.positions()
        if success:
            if data:
                df = pd.DataFrame(data)
                
                try:
                    mtm_col = None
                    for col in ['mtm', 'unrealized_pnl', 'pnl']:
                        if col in df.columns:
                            mtm_col = col
                            break
                    
                    if mtm_col:
                        df[mtm_col] = pd.to_numeric(df[mtm_col], errors='coerce').fillna(0)
                        total_mtm = df[mtm_col].sum()
                        
                        st.metric("Total MTM", f"₹{total_mtm:,.2f}")
                        
                        def style_mtm(v):
                            try:
                                val = float(v)
                                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                return f'color: {color}'
                            except:
                                return ''
                        
                        st.dataframe(
                            df.style.applymap(style_mtm, subset=[mtm_col]),
                            use_container_width=True, 
                            hide_index=True
                        )
                    else:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error processing position data: {e}")
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions.")
        else:
            st.error(f"Failed to load positions: {data}")
    
    with tab7:
        st.header("🗄️ Trade Database (Supabase)")
        st.caption("All trades with entry/exit tracking from cloud DB")
        
        st.markdown("### DB Management")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh DB", key="refresh_db", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button(f"⬇️ Download to CSV", key="download_db", use_container_width=True):
                with st.spinner("Fetching data..."):
                    all_trades_df = get_all_trades()
                    if not all_trades_df.empty:
                        all_trades_df.to_csv(LOCAL_CSV_FILE, index=False)
                        st.success(f"Data saved to {LOCAL_CSV_FILE}")
                    else:
                        st.warning("No data in Supabase to download.")

        with col3:
            if st.button(f"⬆️ Upload CSV to DB", key="upload_db", use_container_width=True):
                if not os.path.exists(LOCAL_CSV_FILE):
                    st.error(f"{LOCAL_CSV_FILE} not found. Download first or create one.")
                else:
                    with st.spinner(f"Uploading {LOCAL_CSV_FILE}..."):
                        try:
                            df = pd.read_csv(LOCAL_CSV_FILE)
                            
                            # Clean data for Supabase
                            df['entry_time'] = pd.to_datetime(df['entry_time']).apply(lambda x: x.isoformat() if pd.notnull(x) else None)
                            df['exit_time'] = pd.to_datetime(df['exit_time']).apply(lambda x: x.isoformat() if pd.notnull(x) else None)
                            
                            # Convert Pandas NaT/NaN to None
                            df = df.where(pd.notnull(df), None)
                            
                            records = df.to_dict('records')
                            
                            if records:
                                # Upsert records based on primary key
                                response = supabase.table('trades').upsert(
                                    records, 
                                    on_conflict='entry_order_id'
                                ).execute()
                                
                                st.success(f"Successfully uploaded/updated {len(response.data)} records.")
                            else:
                                st.warning("CSV is empty.")
                        except Exception as e:
                            st.error(f"Upload failed: {e}")

        with col4:
            st.warning("Reset DB (Careful!)")
            confirm_reset = st.checkbox("Confirm Delete All", key="confirm_reset")
            
            if st.button("🔥 RESET DB", key="reset_db", use_container_width=True, disabled=not confirm_reset):
                with st.spinner("Deleting all records..."):
                    try:
                        # Delete all rows. `gt('quantity', -999999)` is a trick to delete all rows.
                        response = supabase.table("trades").delete().gt("quantity", -9999999).execute()
                        st.success(f"Successfully deleted {len(response.data)} records.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to reset DB: {e}")
        
        st.markdown("---")
        
        st.markdown("### Trade History")
        all_trades = get_all_trades()
        
        if not all_trades.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            open_trades = all_trades[all_trades['status'] == 'OPEN']
            closed_trades = all_trades[all_trades['status'] == 'CLOSED']
            
            col1.metric("Total Trades", len(all_trades))
            col2.metric("Open Trades", len(open_trades))
            col3.metric("Closed Trades", len(closed_trades))
            
            if not closed_trades.empty:
                try:
                    total_pnl = closed_trades['pnl'].astype(float).sum()
                    col4.metric("Total P&L", f"₹{total_pnl:,.2f}")
                except:
                    col4.metric("Total P&L", "N/A")
            else:
                col4.metric("Total P&L", "₹0.00")
            
            st.markdown("---")
            
            filter_option = st.selectbox("Filter", ["All", "Open", "Closed"])
            
            if filter_option == "Open":
                display_trades = open_trades
            elif filter_option == "Closed":
                display_trades = closed_trades
            else:
                display_trades = all_trades
            
            if not display_trades.empty:
                def style_pnl(v):
                    try:
                        if pd.isna(v):
                            return ''
                        val = float(v)
                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                        return f'color: {color}; font-weight: bold'
                    except:
                        return ''
                
                def style_status(s):
                    if s == 'OPEN':
                        return 'background-color: #ffc107; color: black; font-weight: bold'
                    elif s == 'CLOSED':
                        return 'background-color: #28a745; color: white; font-weight: bold'
                    return ''
                
                styled_df = display_trades.style.applymap(style_status, subset=['status'])
                
                if 'pnl' in display_trades.columns:
                    styled_df = styled_df.applymap(style_pnl, subset=['pnl'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "entry_price": st.column_config.NumberColumn(format="₹%.2f"),
                        "target_price": st.column_config.NumberColumn(format="₹%.2f"),
                        "stop_loss": st.column_config.NumberColumn(format="₹%.2f"),
                        "exit_price": st.column_config.NumberColumn(format="₹%.2f"),
                        "pnl": st.column_config.NumberColumn(format="₹%.2f"),
                        "entry_time": st.column_config.DatetimeColumn(format="DD-MM-YYYY HH:mm:ss"),
                        "exit_time": st.column_config.DatetimeColumn(format="DD-MM-YYYY HH:mm:ss"),
                    }
                )
                
                if filter_option == "Open" and not open_trades.empty:
                    st.markdown("---")
                    st.subheader("🔥 Manual Exit")
                    
                    trade_to_exit = st.selectbox(
                        "Select trade to exit",
                        open_trades['entry_order_id'].tolist(),
                        format_func=lambda x: f"{x} - {open_trades[open_trades['entry_order_id']==x]['symbol'].values[0]}"
                    )
                    
                    if st.button("Exit Selected Trade", type="primary"):
                        trade_row = open_trades[open_trades['entry_order_id'] == trade_to_exit].iloc[0]
                        
                        exit_side = 'SELL' if trade_row['side'] == 'BUY' else 'BUY'
                        
                        data, msg = api.place_order(
                            symbol=trade_row['symbol'],
                            side=exit_side,
                            quantity=int(trade_row['quantity']),
                            exchange="NSE"
                        )
                        
                        if data and data.get('status') == 'SUCCESS':
                            success, positions = reports.positions()
                            exit_price = trade_row['entry_price']
                            
                            if success and positions:
                                for pos in positions:
                                    if pos.get('tradingsymbol') == trade_row['symbol']:
                                        exit_price = float(pos.get('ltp', exit_price))
                                        break
                            
                            pnl = (exit_price - trade_row['entry_price']) * trade_row['quantity'] if trade_row['side'] == 'BUY' else (trade_row['entry_price'] - exit_price) * trade_row['quantity']
                            
                            update_trade_exit(
                                entry_order_id=trade_to_exit,
                                exit_order_id=data.get('order_id'),
                                exit_price=exit_price,
                                exit_time=datetime.now(IST).isoformat(),
                                pnl=pnl
                            )
                            
                            st.success(f"✅ {msg} | P&L: ₹{pnl:.2f}")
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info(f"No {filter_option.lower()} trades found.")
        else:
            st.info("🔭 No trades in database yet. Place your first trade to start tracking!")
            st.markdown("""
            **How it works:**
            1. Go to Buy/Sell tabs
            2. Enable Auto Entry or click "Place Entry Order"
            3. Trade is saved to database only when executed
            4. Enable "Auto Exit" in sidebar to automatically exit at target/SL
            5. View all trades history here
            """)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} | NSE India | Definedge Securities")
    st.caption("⚠️ Educational purposes only. Not financial advice.")

# ==================== MAIN ====================
def main():
    if not st.session_state.get('authenticated', False):
        show_authentication_page()
    else:
        show_trading_dashboard()

if __name__ == "__main__":
    main()
