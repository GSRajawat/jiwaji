import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time
import time
import json
import uuid

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NSE Automated Trading Strategy",
    page_icon="🤖",
    layout="wide"
)

# ==================== UTILITY FUNCTIONS ====================
def generate_order_id():
    """Generates a unique, readable order ID for simulation."""
    return f"ORD{uuid.uuid4().hex[:8].upper()}"

def generate_position_id():
    """Generates a unique position ID."""
    return f"POS{uuid.uuid4().hex[:6].upper()}"

def generate_trade_id():
    """Generates a unique trade ID for closed trades."""
    return f"TRD{uuid.uuid4().hex[:6].upper()}"

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
# New state variables for logging and tracking
if 'signals_log' not in st.session_state:
    st.session_state.signals_log = pd.DataFrame() # Record of all generated signals
if 'trade_book' not in st.session_state:
    st.session_state.trade_book = pd.DataFrame() # Executed trades (closed positions)
if 'order_book' not in st.session_state:
    st.session_state.order_book = pd.DataFrame() # All orders (filled, rejected, etc.)
if 'open_positions' not in st.session_state:
    st.session_state.open_positions = pd.DataFrame() # Active positions

# ==================== DEMO DATA GENERATOR ====================
def generate_demo_data():
    """Generate realistic demo data when NSE API is unavailable"""
    stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", 
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT",
        "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN", "BAJFINANCE",
        "NESTLEIND", "WIPRO", "TECHM", "HCLTECH", "TATAMOTORS", "ONGC",
        "NTPC", "POWERGRID", "ADANIPORTS", "JSWSTEEL", "TATASTEEL", "HINDALCO"
    ]
    
    all_stocks = []
    np.random.seed(42)
    
    for symbol in stocks:
        base_price = np.random.uniform(100, 2000)
        open_price = base_price * np.random.uniform(0.98, 1.02)
        pct_change = np.random.uniform(-8.0, 8.0)
        ltp = open_price * (1 + pct_change/100)
        
        # Ensure Day High/Low are properly set relative to LTP and Open
        if pct_change > 0:
            day_high = max(ltp, open_price) * 1.005
            day_low = min(ltp, open_price) * 0.995
        else:
            day_high = max(ltp, open_price) * 1.005
            day_low = min(ltp, open_price) * 0.995
        
        # Ensure LTP is at the high/low for breakout signals
        if np.random.rand() < 0.2: # 20% chance of breakout signal
            if np.random.rand() > 0.5: # 50% chance for buy
                ltp = day_high
            else: # 50% chance for sell
                ltp = day_low
                
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
    # (NSEDataFetcher methods remain largely unchanged)
    
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
                st.success("✅ Fetched F&O securities")
                return response.json()
            
            st.error("❌ API failed. Using demo mode.")
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
            
            # Column mapping
            column_mapping = {
                'lastPrice': 'ltp',
                'previousClose': 'prev_close',
                'dayHigh': 'high',
                'dayLow': 'low',
                'totalTradedValue': 'tradedValue',
                'perChange': 'pChange'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns
            required = ['symbol', 'ltp', 'open', 'high', 'low', 'tradedValue']
            for col in required:
                if col not in df.columns:
                    if col == 'open' and 'previousClose' in df.columns:
                        df['open'] = df['previousClose']
                    elif col not in df.columns:
                        #st.error(f"❌ Missing column: {col}")
                        return pd.DataFrame()
            
            # Convert to numeric
            numeric_cols = ['ltp', 'open', 'high', 'low', 'tradedValue']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate pChange
            if 'pChange' not in df.columns and 'prev_close' in df.columns:
                df['prev_close'] = pd.to_numeric(df['prev_close'], errors='coerce')
                df['pChange'] = ((df['ltp'] - df['prev_close']) / df['prev_close']) * 100
            elif 'pChange' not in df.columns:
                df['pChange'] = ((df['ltp'] - df['open']) / df['open']) * 100
            else:
                df['pChange'] = pd.to_numeric(df['pChange'], errors='coerce')
            
            # Simulate previous traded value
            np.random.seed(42)
            df['prevTradedValue'] = df['tradedValue'] * np.random.uniform(0.8, 1.2, len(df))
            
            # Clean data
            df = df.dropna(subset=['ltp', 'open', 'high', 'low', 'tradedValue'])
            df = df[df['ltp'] > 0]
            df = df[df['tradedValue'] > 0]
            df['pChange'] = df['pChange'].fillna(0)
            
            #st.success(f"✅ Parsed {len(df)} stocks")
            return df
            
        except Exception as e:
            st.error(f"❌ Parsing error: {str(e)}")
            return pd.DataFrame()


# ==================== DEFINEDGE API ====================
class DefinedgeAPI:
    """Definedge Securities API Wrapper (Includes local logging)"""
    
    # (API methods like set_credentials, generate_otp, verify_otp_and_authenticate remain unchanged)
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

    def place_order(self, symbol, side, quantity, exchange="NSE", product="INTRADAY", order_type="MARKET", price=0, is_exit=False, position_id=None):
        """Place order with proper symbol formatting and log it locally."""
        if not self.access_token:
            return None, "Not authenticated"
        
        # Format symbol
        formatted_symbol = self.format_symbol(symbol, exchange)
        
        # --- API Call Simulation & Local Logging ---
        order_id = generate_order_id()

        # Simulate API request/response
        # In a real app, this would be the actual request and we'd check the status code
        status = 'SUCCESS' # Assume success for simulation
        message = f"Order {order_id} placed successfully."

        # Log to Order Book
        order_log = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'order_id': order_id,
            'symbol': symbol,
            'formatted_symbol': formatted_symbol,
            'side': side,
            'quantity': quantity,
            'exchange': exchange,
            'product': product,
            'order_type': order_type,
            'price': price,
            'status': 'FILLED', # Assume market order is instantly filled for simplicity
            'fill_price': -1, # To be updated by Strategy/Position Manager using LTP
            'is_exit': is_exit,
            'position_id': position_id if is_exit else None,
            'message': message,
        }
        
        st.session_state.order_book = pd.concat([st.session_state.order_book, pd.DataFrame([order_log])], ignore_index=True)

        # Simulate API response
        data = {'status': status, 'message': message, 'order_id': order_id}

        return data, message
    
    def format_symbol(self, symbol, exchange):
        """Format trading symbol for API"""
        clean = symbol.replace("-EQ", "").strip()
        
        if exchange == "NSE":
            # Check if derivative (has numbers or FUT/CE/PE)
            is_derivative = any(x in clean.upper() for x in ["FUT", "CE", "PE"]) or any(c.isdigit() for c in clean)
            
            if not is_derivative:
                return f"{clean}-EQ"
        
        return clean

# ==================== TRADING STRATEGY AND MANAGEMENT ====================
class EnhancedTradingStrategy:
    """Manages signals, positions, and automatically places entry/exit orders."""
    
    def is_trading_time(self):
        now = datetime.now().time()
        return dt_time(9, 25) <= now <= dt_time(14, 30)

    def filter_by_traded_value(self, df):
        if df.empty:
            return df
        return df[df['tradedValue'] > df['prevTradedValue']].copy()
    
    def is_already_in_position(self, symbol):
        """Check if an open position exists for the given symbol."""
        if st.session_state.open_positions.empty:
            return False
        return symbol in st.session_state.open_positions['symbol'].tolist()
        
    def record_signal(self, signal):
        """Record the signal generation event."""
        log_entry = signal.copy()
        log_entry['signal_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.signals_log = pd.concat([st.session_state.signals_log, pd.DataFrame([log_entry])], ignore_index=True)

    def generate_signals(self, df_all):
        signals = []
        
        if df_all.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        df_filtered = self.filter_by_traded_value(df_all)
        
        df_gainers = df_filtered[df_filtered['pChange'] > 0].copy().sort_values('tradedValue', ascending=False)
        df_losers = df_filtered[df_filtered['pChange'] < 0].copy().sort_values('tradedValue', ascending=False)
        
        is_trading_time = self.is_trading_time()
        
        # BUY signals - day high breakout
        for _, stock in df_gainers.iterrows():
            if stock['ltp'] >= stock['high']:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'BUY',
                    'ltp': float(stock['ltp']),
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['low']),
                    'sl_distance': float(stock['ltp'] - stock['low']),
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day High Breakout'
                })
        
        # SELL signals - day low breakdown
        for _, stock in df_losers.iterrows():
            if stock['ltp'] <= stock['low']:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'SELL',
                    'ltp': float(stock['ltp']),
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['high']),
                    'sl_distance': float(stock['high'] - stock['ltp']),
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'can_trade': is_trading_time,
                    'breakout_type': 'Day Low Breakdown'
                })
        
        return signals, df_gainers, df_losers

    def check_and_manage_positions(self, df_ltp, api):
        """Check all open positions for SL/Target hit and place exit orders."""
        if st.session_state.open_positions.empty or df_ltp.empty:
            return

        exit_orders = []
        df_positions = st.session_state.open_positions.copy()
        
        # Merge current LTP into positions dataframe
        df_positions = df_positions.merge(df_ltp[['symbol', 'ltp']], on='symbol', how='left')
        df_positions.rename(columns={'ltp': 'current_ltp'}, inplace=True)
        
        for index, pos in df_positions.iterrows():
            symbol = pos['symbol']
            current_ltp = pos['current_ltp']

            if pd.isna(current_ltp):
                continue

            exit_side = 'SELL' if pos['entry_side'] == 'BUY' else 'BUY'
            
            # Check for Exit Condition (SL or Target)
            hit_sl = False
            hit_target = False

            if pos['entry_side'] == 'BUY':
                if current_ltp <= pos['stop_loss']:
                    hit_sl = True
                if current_ltp >= pos['target']:
                    hit_target = True
            
            if pos['entry_side'] == 'SELL':
                if current_ltp >= pos['stop_loss']:
                    hit_sl = True
                if current_ltp <= pos['target']:
                    hit_target = True

            exit_reason = None
            if hit_sl:
                exit_reason = "SL Hit"
            elif hit_target:
                exit_reason = "Target Hit"
            
            if exit_reason:
                # Place Exit Order (always MARKET for SL/Target hit)
                st.info(f"🚨 Exiting {symbol} ({exit_reason}) at {current_ltp:.2f}")
                
                if api and hasattr(api, 'place_order'):
                    data, msg = api.place_order(
                        symbol=symbol,
                        side=exit_side,
                        quantity=int(pos['quantity']),
                        is_exit=True,
                        position_id=pos['position_id']
                    )
                    
                    if data and data.get('status') == 'SUCCESS':
                        pos_data = pos.to_dict()
                        pos_data['status'] = 'CLOSED'
                        pos_data['exit_reason'] = exit_reason
                        pos_data['exit_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        pos_data['exit_order_id'] = data['order_id']
                        exit_orders.append(pos_data)
                    else:
                        st.error(f"Failed to place exit order for {symbol}: {msg}")
        
        # Now update the position and trade books based on successful exit order placement
        self.update_books_after_exit(exit_orders, df_ltp)

    def update_books_after_exit(self, exited_positions_data, df_ltp):
        """Finalize positions and update the trade book after an exit order is executed."""
        
        positions_to_remove = []
        new_trade_entries = []

        for pos_data in exited_positions_data:
            symbol = pos_data['symbol']
            
            current_ltp = df_ltp.loc[df_ltp['symbol'] == symbol, 'ltp'].iloc[0] if symbol in df_ltp['symbol'].tolist() else None
            
            if current_ltp is None:
                continue

            # Update Order Book (simulate filled order with exit price)
            st.session_state.order_book.loc[
                st.session_state.order_book['order_id'] == pos_data['exit_order_id'], 
                ['status', 'fill_price']
            ] = ['FILLED', current_ltp]

            # Calculate PNL
            entry_price = pos_data['entry_price']
            exit_price = current_ltp
            quantity = pos_data['quantity']
            
            if pos_data['entry_side'] == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            # Record final trade
            trade_entry = {
                'trade_id': generate_trade_id(),
                'symbol': symbol,
                'entry_timestamp': pos_data['entry_timestamp'],
                'entry_side': pos_data['entry_side'],
                'entry_price': entry_price,
                'exit_timestamp': pos_data['exit_timestamp'],
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl,
                'exit_reason': pos_data['exit_reason'],
            }
            new_trade_entries.append(trade_entry)
            positions_to_remove.append(pos_data['position_id'])

        # Update Trade Book
        if new_trade_entries:
            st.session_state.trade_book = pd.concat([st.session_state.trade_book, pd.DataFrame(new_trade_entries)], ignore_index=True)
        
        # Remove positions from Open Positions
        if positions_to_remove:
            st.session_state.open_positions = st.session_state.open_positions[
                ~st.session_state.open_positions['position_id'].isin(positions_to_remove)
            ]

    def calculate_current_pnl(self, df_ltp):
        """Calculate current PNL for all open positions."""
        if st.session_state.open_positions.empty or df_ltp.empty:
            return st.session_state.open_positions.copy()
        
        df_pos = st.session_state.open_positions.copy()
        df_pos = df_pos.merge(df_ltp[['symbol', 'ltp']], on='symbol', how='left')
        
        df_pos['current_ltp'] = df_pos['ltp'].fillna(df_pos['entry_price'])
        
        def calculate_pnl(row):
            if row['entry_side'] == 'BUY':
                return (row['current_ltp'] - row['entry_price']) * row['quantity']
            else: # SELL
                return (row['entry_price'] - row['current_ltp']) * row['quantity']

        df_pos['current_pnl'] = df_pos.apply(calculate_pnl, axis=1)
        df_pos['pnl_pct'] = (df_pos['current_pnl'] / (df_pos['entry_price'] * df_pos['quantity'])).replace([np.inf, -np.inf], 0) * 100
        
        return df_pos.drop(columns=['ltp'])

# ==================== SIGNAL DISPLAY AND EXECUTION ====================
def display_signals_table(signals, title, api, total_risk, auto_trade_enabled):
    if not signals:
        st.info(f"No {title.lower()} found")
        return
    
    st.markdown(f"### {title}")
    
    now = datetime.now().time()
    is_trading_time = dt_time(9, 25) <= now <= dt_time(14, 30)
    is_api_connected = (api is not None)
    
    if is_trading_time:
        st.success(f"✅ Trading Active ({datetime.now().strftime('%H:%M:%S')})")
    else:
        st.warning(f"⚠️ Outside Hours | Current: {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    strategy = EnhancedTradingStrategy()
    
    # --- AUTO TRADING EXECUTION BLOCK ---
    if auto_trade_enabled and is_api_connected and is_trading_time:
        
        st.subheader("🤖 Automatic Execution Status")
        
        for signal in signals:
            symbol = signal['symbol']
            
            if not strategy.is_already_in_position(symbol):
                # 1. Calculate Quantity
                risk_per_share = signal['sl_distance']
                qty = int(total_risk / risk_per_share) if risk_per_share > 0 else 1
                qty = max(1, qty)
                
                # 2. Place Order (Entry)
                formatted_sym = api.format_symbol(symbol, "NSE")
                
                with st.spinner(f"🤖 Placing {signal['type']} order for {symbol} (Qty: {qty})..."):
                    data, msg = api.place_order(
                        symbol=symbol,
                        side=signal['type'],
                        quantity=int(qty),
                        exchange="NSE",
                        is_exit=False
                    )
                
                if data and data.get('status') == 'SUCCESS':
                    st.success(f"✅ Auto Order for {symbol} placed: {msg}")
                    
                    # 3. Add to Open Positions (Simulate fill immediately at LTP)
                    position_id = generate_position_id()
                    
                    new_pos = {
                        'position_id': position_id,
                        'symbol': symbol,
                        'entry_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_side': signal['type'],
                        'entry_price': signal['ltp'], # Entry at signal LTP
                        'quantity': qty,
                        'stop_loss': signal['initial_stop_loss'],
                        'target': signal['target'],
                        'initial_risk': signal['sl_distance'] * qty,
                        'status': 'OPEN',
                        'entry_order_id': data['order_id']
                    }
                    st.session_state.open_positions = pd.concat([st.session_state.open_positions, pd.DataFrame([new_pos])], ignore_index=True)
                    
                    # 4. Record Signal
                    strategy.record_signal(signal)

                else:
                    st.error(f"❌ Auto Order Failed for {symbol}: {msg}")
            else:
                st.info(f"⏭️ Skipping {symbol}. Position already open.")
                
        st.markdown("---")
        st.success(f"🤖 Auto Trading Cycle Complete for {title}. See Positions tab.")
        
    # --- MANUAL TRADING EXECUTION BLOCK ---
    else:
        for i, signal in enumerate(signals[:30]):
            symbol = signal['symbol']
            is_open = strategy.is_already_in_position(symbol)
            status = "🟢" if signal['can_trade'] else "🔴"
            
            with st.expander(
                f"{status} {symbol} | ₹{signal['ltp']:.2f} | {signal['pct_change']:.2f}% | {signal['breakout_type']}",
                expanded=(i < 3 and not is_open)
            ):
                col1, col2, col3, col4 = st.columns(4)
                
                # Metrics
                with col1: st.metric("LTP", f"₹{signal['ltp']:.2f}")
                with col2: st.metric("Target (Open)", f"₹{signal['target']:.2f}")
                with col3: st.metric("Stop Loss", f"₹{signal['initial_stop_loss']:.2f}")
                with col4: st.metric("% Change", f"{signal['pct_change']:.2f}%")
                
                st.markdown("---")
                
                # Calculate quantity
                risk_per_share = signal['sl_distance']
                calc_qty = int(total_risk / risk_per_share) if risk_per_share > 0 else 1
                qty_default = max(1, calc_qty)
                
                st.subheader(f"🚀 Execute Trade (Manual)")
                
                if api and hasattr(api, 'format_symbol'):
                    formatted_sym = api.format_symbol(symbol, "NSE")
                else:
                    formatted_sym = f"{symbol}-EQ"
                
                st.info(f"📋 **API Format:** `{formatted_sym}`")
                
                col_qty, col_btn = st.columns([1, 1])
                
                with col_qty:
                    qty = st.number_input(
                        f"Quantity (Risk: ₹{total_risk})",
                        key=f"q{i}",
                        min_value=1,
                        value=qty_default,
                        help=f"Auto: ₹{total_risk} ÷ ₹{risk_per_share:.2f}"
                    )
                
                with col_btn:
                    st.write("") # Spacing
                    st.write("")
                    disabled = not signal['can_trade'] or is_open or not is_api_connected
                    
                    button_text = f"Already OPEN" if is_open else f"Execute {signal['type']}"
                    
                    if st.button(button_text, key=f"b{i}", type="primary", disabled=disabled):
                        
                        if is_api_connected:
                            with st.spinner(f"Placing {signal['type']} manually..."):
                                data, msg = api.place_order(
                                    symbol=formatted_sym,
                                    side=signal['type'],
                                    quantity=int(qty),
                                    exchange="NSE",
                                    is_exit=False
                                )
                                
                                if data and data.get('status') == 'SUCCESS':
                                    st.success(f"✅ Manual Order for {symbol} placed: {msg}")
                                    
                                    # Add to Open Positions (Simulate fill immediately at LTP)
                                    position_id = generate_position_id()
                                    new_pos = {
                                        'position_id': position_id,
                                        'symbol': symbol,
                                        'entry_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'entry_side': signal['type'],
                                        'entry_price': signal['ltp'],
                                        'quantity': qty,
                                        'stop_loss': signal['initial_stop_loss'],
                                        'target': signal['target'],
                                        'initial_risk': signal['sl_distance'] * qty,
                                        'status': 'OPEN',
                                        'entry_order_id': data['order_id']
                                    }
                                    st.session_state.open_positions = pd.concat([st.session_state.open_positions, pd.DataFrame([new_pos])], ignore_index=True)
                                    strategy.record_signal(signal)
                                    st.rerun() # Refresh to update status
                                else:
                                    st.error(f"❌ Manual Order Failed: {msg}")
                        else:
                            st.error("Cannot place orders in Demo Mode.")


# ==================== MAIN DASHBOARD ====================
def show_trading_dashboard():
    api = st.session_state.api_instance
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user = st.session_state.api_credentials.get('user_id', 'DEMO')
        st.success(f"🟢 {user}")
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            for key in ['authenticated', 'otp_sent', 'api_credentials', 'api_instance', 'access_token']:
                st.session_state[key] = False if 'authenticated' in key or 'otp_sent' in key else {} if 'credentials' in key else None
            st.rerun()
    
    st.title("📊 NSE Automated Trading Strategy")
    st.markdown("**Breakout Strategy | Value-Based | F&O Securities**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        auto_trade_enabled = st.checkbox("🤖 Enable Auto Trading", value=False, help="Automatically place entry orders for new signals and exit orders when SL/Target is hit.")
        
        st.markdown("---")
        st.metric("Time", datetime.now().strftime('%H:%M:%S'))
        
        now = datetime.now().time()
        if dt_time(9, 25) <= now <= dt_time(14, 30):
            st.success("✅ Trading Hours")
        else:
            st.error("❌ Outside Hours")
        
        st.markdown("---")
        max_signals = st.slider("Max Signals", 10, 100, 30)
        total_risk = st.number_input("Risk/Trade (₹)", 10, 10000, 100, 10)
        
        st.markdown("---")
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        scan = st.button("🔍 Fetch Live / Run Cycle", use_container_width=True, type="primary")
        demo = st.button("🧪 Demo Data", use_container_width=True)
    
    # Fetch data
    if scan or auto_refresh or demo:
        with st.spinner("Fetching..."):
            if demo:
                data = generate_demo_data()
                fetcher = NSEDataFetcher()
                df_all = fetcher.parse_nse_data(data)
            else:
                fetcher = NSEDataFetcher()
                data = fetcher.fetch_all_securities()
                if data is None:
                    st.error("❌ Failed. Try demo mode.")
                    st.stop()
                df_all = fetcher.parse_nse_data(data)
    else:
        st.info("👆 Click 'Fetch Live / Run Cycle' or 'Demo Data' to start")
        st.stop()
    
    # Generate signals
    strategy = EnhancedTradingStrategy()
    signals, gainers, losers = strategy.generate_signals(df_all)
    
    # --- POSITION MANAGEMENT AND EXIT LOGIC ---
    if not df_all.empty and auto_trade_enabled and api is not None:
        strategy.check_and_manage_positions(df_all, api)
        
    # --- PNL Calculation for Display ---
    current_positions_df = strategy.calculate_current_pnl(df_all)
    
    buy_signals = [s for s in signals if s['type'] == 'BUY'][:max_signals]
    sell_signals = [s for s in signals if s['type'] == 'SELL'][:max_signals]
    
    # Metrics
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7) 
    col1.metric("Total Stocks", len(df_all))
    col2.metric("Filtered", len(df_all[df_all['tradedValue'] > df_all['prevTradedValue']]))
    col3.metric("Gainers", len(gainers))
    col4.metric("Losers", len(losers))
    col5.metric("Buy Signals", len(buy_signals))
    col6.metric("Open Positions", len(current_positions_df))

    total_pnl = current_positions_df['current_pnl'].sum() if not current_positions_df.empty else 0
    pnl_color = 'green' if total_pnl >= 0 else 'red'
    col7.metric("Total Open PNL", f"₹{total_pnl:.2f}", delta_color=pnl_color)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🟢 Buy Signals", "🔴 Sell Signals", "📋 Open Positions", "📜 Trade Book", "📒 Order Book"]) 
    
    with tab1:
        display_signals_table(buy_signals, "🟢 Long (Day High Breakout)", api, total_risk, auto_trade_enabled)
    
    with tab2:
        display_signals_table(sell_signals, "🔴 Short (Day Low Breakdown)", api, total_risk, auto_trade_enabled)

    # --- NEW TABS FOR TRADING DATA ---
    with tab3:
        st.subheader("📋 Open Positions (Real-time PNL)")
        if current_positions_df.empty:
            st.info("No open positions.")
        else:
            display_df = current_positions_df[[
                'symbol', 'entry_side', 'quantity', 'entry_price', 'current_ltp', 
                'stop_loss', 'target', 'current_pnl', 'pnl_pct', 'entry_timestamp', 'position_id'
            ]].copy()
            
            # Formatting
            for col in ['entry_price', 'current_ltp', 'stop_loss', 'target', 'current_pnl']:
                display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📜 Trade Book (Closed Positions)")
        if st.session_state.trade_book.empty:
            st.info("No closed trades recorded.")
        else:
            display_df = st.session_state.trade_book[[
                'trade_id', 'symbol', 'entry_side', 'quantity', 'entry_price', 
                'exit_price', 'pnl', 'exit_reason', 'entry_timestamp', 'exit_timestamp'
            ]].copy()
            
            # Formatting
            for col in ['entry_price', 'exit_price', 'pnl']:
                display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("📒 Order Book (All Orders)")
        if st.session_state.order_book.empty:
            st.info("No orders recorded.")
        else:
            st.dataframe(st.session_state.order_book[[
                'timestamp', 'order_id', 'symbol', 'side', 'quantity', 
                'order_type', 'status', 'fill_price', 'is_exit'
            ]], use_container_width=True, hide_index=True)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NSE India | Definedge Securities")
    st.caption("⚠️ Educational purposes only. Not financial advice. Trade simulation relies on local data updates.")

# ==================== MAIN ====================
def main():
    if not st.session_state.get('authenticated', False):
        show_authentication_page()
    else:
        show_trading_dashboard()

def show_authentication_page():
    # (Authentication page remains unchanged)
    st.title("🔐 Definedge Securities Login")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Quick Start
        1. Enter API credentials
        2. Generate OTP
        3. Enter OTP and login
        
        Or use Demo Mode (no trading)
        """)
        
        if st.button("🧪 Demo Mode", type="secondary", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.api_instance = None
            st.session_state.api_credentials = {'user_id': 'DEMO'}
            st.rerun()
    
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
            st.markdown("### 🔑 Enter OTP")
            otp = st.text_input("6-digit OTP")
            
            if st.form_submit_button("✅ Verify & Login", use_container_width=True):
                if otp and len(otp) == 6:
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
                    st.error("Enter 6-digit OTP")

if __name__ == "__main__":
    main()
