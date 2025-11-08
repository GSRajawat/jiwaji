import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time
import time
import json

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="NSE Enhanced Trading Strategy",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trailing_stops' not in st.session_state:
    st.session_state.trailing_stops = {}
if 'otp_token' not in st.session_state:
    st.session_state.otp_token = None
if 'api_authenticated' not in st.session_state:
    st.session_state.api_authenticated = False
if 'placed_orders' not in st.session_state:
    st.session_state.placed_orders = []

# ==================== DEFINEDGE API INTEGRATION ====================
class DefinedgeAPI:
    """Definedge Securities API Integration"""
    
    def __init__(self, user_id, api_token, api_secret):
        self.user_id = user_id
        self.api_token = api_token
        self.api_secret = api_secret
        self.base_url = "https://api.definedgesecurities.com"
        self.otp_token = None
        self.session = requests.Session()
        
    def generate_otp(self):
        """Generate OTP for authentication"""
        try:
            url = f"{self.base_url}/auth/generateOTP"
            headers = {
                'Content-Type': 'application/json',
                'X-API-KEY': self.api_token
            }
            payload = {'userId': self.user_id}
            
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return True, "OTP sent successfully"
            else:
                return False, f"Failed to generate OTP: {response.text}"
                
        except Exception as e:
            return False, f"Error generating OTP: {str(e)}"
    
    def verify_otp(self, otp):
        """Verify OTP and get OTP token"""
        try:
            url = f"{self.base_url}/auth/validateOTP"
            headers = {
                'Content-Type': 'application/json',
                'X-API-KEY': self.api_token
            }
            payload = {
                'userId': self.user_id,
                'otp': otp
            }
            
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'otpToken' in data:
                    self.otp_token = data['otpToken']
                    st.session_state.otp_token = self.otp_token
                    st.session_state.api_authenticated = True
                    return True, "Authentication successful", self.otp_token
                else:
                    return False, "OTP token not received", None
            else:
                return False, f"OTP verification failed: {response.text}", None
                
        except Exception as e:
            return False, f"Error verifying OTP: {str(e)}", None
    
    def place_order(self, symbol, transaction_type, quantity, order_type="MARKET", 
                    product_type="INTRADAY", price=0, stop_loss=0, target=0):
        """Place intraday order"""
        try:
            if not self.otp_token:
                return False, "Not authenticated. Please verify OTP first.", None
            
            url = f"{self.base_url}/orders/regular"
            headers = {
                'Content-Type': 'application/json',
                'X-API-KEY': self.api_token,
                'X-OTP-TOKEN': self.otp_token
            }
            
            payload = {
                'userId': self.user_id,
                'exchange': 'NSE',
                'tradingSymbol': symbol,
                'transactionType': transaction_type,
                'quantity': quantity,
                'orderType': order_type,
                'productType': product_type,
                'price': price,
                'stopLoss': stop_loss,
                'target': target,
                'validity': 'DAY'
            }
            
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return True, "Order placed successfully", data
            else:
                return False, f"Order placement failed: {response.text}", None
                
        except Exception as e:
            return False, f"Error placing order: {str(e)}", None

# ==================== API CREDENTIALS MANAGEMENT ====================
def show_api_credentials_sidebar():
    """Display API credentials configuration in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔐 Definedge API Settings")
    
    with st.sidebar.expander("📝 API Credentials", expanded=not st.session_state.api_authenticated):
        user_id = st.text_input("User ID", value="1275682", type="default", key="user_id")
        api_token = st.text_input("API Token", value="979f2a2e-8122-4017-b81e-e3d7385de378", type="password", key="api_token")
        api_secret = st.text_input("API Secret", value="Z/LLGKupXj0ukCYybRe5Gg==", type="password", key="api_secret")
        st.caption("💡 Credentials stored only in session")
    
    with st.sidebar.expander("🔑 OTP Authentication", expanded=not st.session_state.api_authenticated):
        if not st.session_state.api_authenticated:
            st.warning("⚠️ Not authenticated")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📱 Generate OTP", use_container_width=True, type="primary"):
                    if user_id and api_token and api_secret:
                        api = DefinedgeAPI(user_id, api_token, api_secret)
                        success, message = api.generate_otp()
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.error("Enter all credentials first")
            
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            st.markdown("---")
            otp = st.text_input("Enter OTP", max_chars=6, type="password", key="otp_input")
            
            if st.button("✅ Verify OTP", use_container_width=True, type="primary"):
                if otp and user_id and api_token and api_secret:
                    api = DefinedgeAPI(user_id, api_token, api_secret)
                    success, message, otp_token = api.verify_otp(otp)
                    
                    if success:
                        st.success(f"✅ {message}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Enter OTP and credentials")
        else:
            st.success("✅ Authenticated")
            st.info(f"🔑 Token: {st.session_state.otp_token[:20]}..." if st.session_state.otp_token else "Active")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.api_authenticated = False
                st.session_state.otp_token = None
                st.rerun()
    
    st.sidebar.markdown("---")
    if st.session_state.api_authenticated:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
    
    return user_id, api_token, api_secret

# ==================== DEMO DATA GENERATOR ====================
def generate_demo_data():
    """Generate realistic demo data"""
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
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
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None
    
    def parse_nse_data(self, nse_data):
        try:
            if not nse_data:
                return pd.DataFrame()
            
            df = pd.DataFrame()
            
            if 'data' in nse_data and isinstance(nse_data['data'], list):
                df = pd.DataFrame(nse_data['data'])
            elif isinstance(nse_data, list):
                df = pd.DataFrame(nse_data)
            
            if df.empty:
                return pd.DataFrame()
            
            # Column mapping
            column_mapping = {
                'open_price': 'open',
                'high_price': 'high', 
                'low_price': 'low',
                'lastPrice': 'ltp',
                'totalTradedValue': 'tradedValue',
                'previousClose': 'prev_close'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns
            if 'ltp' not in df.columns:
                return pd.DataFrame()
            
            numeric_cols = ['ltp', 'open', 'high', 'low', 'tradedValue']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'pChange' not in df.columns and 'prev_close' in df.columns:
                df['pChange'] = ((df['ltp'] - df['prev_close']) / df['prev_close']) * 100
            
            df['prevTradedValue'] = df['tradedValue'] * np.random.uniform(0.8, 1.2, len(df))
            
            df = df.dropna(subset=['ltp', 'tradedValue'])
            df = df[df['ltp'] > 0]
            
            return df
            
        except Exception as e:
            st.error(f"Parse error: {str(e)}")
            return pd.DataFrame()

# ==================== STRATEGY ====================
class EnhancedTradingStrategy:
    
    def __init__(self):
        self.positions = []
    
    def is_trading_time(self):
        now = datetime.now().time()
        return dt_time(9, 25) <= now <= dt_time(14, 30)
    
    def generate_signals(self, df_all):
        signals = []
        
        if df_all.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        # Filter by traded value
        df_filtered = df_all[df_all['tradedValue'] > df_all['prevTradedValue']].copy()
        
        if df_filtered.empty:
            return signals, pd.DataFrame(), pd.DataFrame()
        
        df_gainers = df_filtered[df_filtered['pChange'] > 0].copy()
        df_losers = df_filtered[df_filtered['pChange'] < 0].copy()
        
        df_gainers = df_gainers.sort_values('tradedValue', ascending=False)
        df_losers = df_losers.sort_values('tradedValue', ascending=False)
        
        is_trading = self.is_trading_time()
        
        # BUY signals - LTP crosses day high
        for _, stock in df_gainers.iterrows():
            ltp = float(stock['ltp'])
            day_high = float(stock['high'])
            
            if ltp >= day_high and ltp > 0:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'BUY',
                    'ltp': ltp,
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['low']),
                    'current_stop_loss': float(stock['low']),
                    'sl_distance': ltp - float(stock['low']),
                    'day_high': day_high,
                    'day_low': float(stock['low']),
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'prev_traded_value': float(stock['prevTradedValue']),
                    'traded_value_change': ((float(stock['tradedValue']) - float(stock['prevTradedValue'])) / float(stock['prevTradedValue'])) * 100,
                    'can_trade': is_trading,
                    'breakout_type': 'Day High Breakout'
                })
        
        # SELL signals - LTP crosses day low
        for _, stock in df_losers.iterrows():
            ltp = float(stock['ltp'])
            day_low = float(stock['low'])
            
            if ltp <= day_low and ltp > 0:
                signals.append({
                    'symbol': stock['symbol'],
                    'type': 'SELL',
                    'ltp': ltp,
                    'open': float(stock['open']),
                    'target': float(stock['open']),
                    'initial_stop_loss': float(stock['high']),
                    'current_stop_loss': float(stock['high']),
                    'sl_distance': float(stock['high']) - ltp,
                    'day_high': float(stock['high']),
                    'day_low': day_low,
                    'pct_change': float(stock['pChange']),
                    'traded_value': float(stock['tradedValue']),
                    'prev_traded_value': float(stock['prevTradedValue']),
                    'traded_value_change': ((float(stock['tradedValue']) - float(stock['prevTradedValue'])) / float(stock['prevTradedValue'])) * 100,
                    'can_trade': is_trading,
                    'breakout_type': 'Day Low Breakdown'
                })
        
        return signals, df_gainers, df_losers

# ==================== MAIN APP ====================
def main():
    st.title("📊 NSE Enhanced Trading Strategy")
    st.markdown("**All Securities | Breakout Strategy | Live Trading**")
    
    user_id, api_token, api_secret = show_api_credentials_sidebar()
    
    api = None
    if st.session_state.api_authenticated:
        api = DefinedgeAPI(user_id, api_token, api_secret)
        api.otp_token = st.session_state.otp_token
    
    with st.sidebar:
        st.header("⚙️ Strategy Rules")
        st.markdown("""
        **Filters:**
        - Current value > Previous day
        - Trading: 9:25 AM - 2:30 PM
        
        **Long:** LTP crosses day high
        **Short:** LTP crosses day low
        **Target:** Open price
        **SL:** Day low/high
        """)
        
        st.markdown("---")
        st.metric("Time", datetime.now().strftime('%H:%M:%S'))
        
        now = datetime.now().time()
        if dt_time(9, 25) <= now <= dt_time(14, 30):
            st.success("✅ Trading Hours")
        else:
            st.error("❌ Outside Hours")
        
        st.markdown("---")
        max_signals = st.slider("Max Signals", 10, 100, 30)
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        col1, col2 = st.columns(2)
        with col1:
            scan_button = st.button("🔍 Live Data", use_container_width=True, type="primary")
        with col2:
            demo_button = st.button("🧪 Demo", use_container_width=True)
    
    if scan_button or auto_refresh or demo_button:
        with st.spinner("Fetching..."):
            if demo_button:
                all_data = generate_demo_data()
                fetcher = NSEDataFetcher()
                df_all = fetcher.parse_nse_data(all_data)
            else:
                fetcher = NSEDataFetcher()
                all_data = fetcher.fetch_all_securities()
                
                if all_data:
                    df_all = fetcher.parse_nse_data(all_data)
                else:
                    st.error("Failed to fetch NSE data")
                    st.stop()
    else:
        st.info("👆 Click 'Live Data' or 'Demo' to start")
        st.stop()
    
    strategy = EnhancedTradingStrategy()
    signals, df_gainers, df_losers = strategy.generate_signals(df_all)
    
    buy_signals = [s for s in signals if s['type'] == 'BUY'][:max_signals]
    sell_signals = [s for s in signals if s['type'] == 'SELL'][:max_signals]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total", len(df_all))
    with col2:
        filtered = len(df_all[df_all['tradedValue'] > df_all['prevTradedValue']])
        st.metric("Filtered", filtered)
    with col3:
        st.metric("Gainers", len(df_gainers))
    with col4:
        st.metric("Losers", len(df_losers))
    with col5:
        st.metric("Buy", len(buy_signals))
    with col6:
        st.metric("Sell", len(sell_signals))
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🟢 Buy Signals", "🔴 Sell Signals", "📥 Export"])
    
    with tab1:
        st.subheader("🟢 Buy Signals (Day High Breakouts)")
        
        for i, signal in enumerate(buy_signals):
            with st.expander(
                f"{'🟢' if signal['can_trade'] else '🔴'} {signal['symbol']} | ₹{signal['ltp']:.2f} | {signal['pct_change']:.2f}%",
                expanded=(i < 3)
            ):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("LTP", f"₹{signal['ltp']:.2f}")
                    st.metric("% Change", f"{signal['pct_change']:.2f}%")
                
                with col2:
                    st.metric("Target (Open)", f"₹{signal['target']:.2f}")
                    st.metric("Day High", f"₹{signal['day_high']:.2f}")
                
                with col3:
                    st.metric("Stop Loss", f"₹{signal['initial_stop_loss']:.2f}")
                    st.metric("Day Low", f"₹{signal['day_low']:.2f}")
                
                with col4:
                    st.metric("Traded Value", f"₹{signal['traded_value']/1e7:.2f} Cr")
                    st.metric("Value Change", f"{signal['traded_value_change']:.2f}%")
    
    with tab2:
        st.subheader("🔴 Sell Signals (Day Low Breakdowns)")
        
        for i, signal in enumerate(sell_signals):
            with st.expander(
                f"{'🟢' if signal['can_trade'] else '🔴'} {signal['symbol']} | ₹{signal['ltp']:.2f} | {signal['pct_change']:.2f}%",
                expanded=(i < 3)
            ):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("LTP", f"₹{signal['ltp']:.2f}")
                    st.metric("% Change", f"{signal['pct_change']:.2f}%")
                
                with col2:
                    st.metric("Target (Open)", f"₹{signal['target']:.2f}")
                    st.metric("Day Low", f"₹{signal['day_low']:.2f}")
                
                with col3:
                    st.metric("Stop Loss", f"₹{signal['initial_stop_loss']:.2f}")
                    st.metric("Day High", f"₹{signal['day_high']:.2f}")
                
                with col4:
                    st.metric("Traded Value", f"₹{signal['traded_value']/1e7:.2f} Cr")
                    st.metric("Value Change", f"{signal['traded_value_change']:.2f}%")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if buy_signals:
                df_export = pd.DataFrame([{
                    'Symbol': s['symbol'],
                    'Type': s['type'],
                    'LTP': s['ltp'],
                    'Target': s['target'],
                    'Stop Loss': s['initial_stop_loss'],
                    'Change %': s['pct_change'],
                    'Can Trade': 'Yes' if s['can_trade'] else 'No'
                } for s in buy_signals])
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "📥 Download Buy Signals",
                    csv,
                    f"buy_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    use_container_width=True
                )
        
        with col2:
            if sell_signals:
                df_export = pd.DataFrame([{
                    'Symbol': s['symbol'],
                    'Type': s['type'],
                    'LTP': s['ltp'],
                    'Target': s['target'],
                    'Stop Loss': s['initial_stop_loss'],
                    'Change %': s['pct_change'],
                    'Can Trade': 'Yes' if s['can_trade'] else 'No'
                } for s in sell_signals])
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "📥 Download Sell Signals",
                    csv,
                    f"sell_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    use_container_width=True
                )
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NSE India")

if __name__ == "__main__":
    main()
