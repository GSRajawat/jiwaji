import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import requests
import json


# --- API Credentials ---
# NOTE: In a real application, never hardcode passwords. Use environment variables.
USER_SESSION = "f68b270591263a92f1d4182a6a5397142b0c254bdf885738c55d854445b3ac9c"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2" # Note: Only used if session ID expires/is invalid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flattrade API Helper (simplified version based on the documentation)
class FlattradeAPI:
    def __init__(self):
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.user_id = None
        self.session_token = None
        
    def set_session(self, user_id: str, session_token: str):
        """Set user session details"""
        self.user_id = user_id
        self.session_token = session_token
        return {"status": "Session set successfully"}
    
    def searchscrip(self, exchange: str, searchtext: str):
        """Search for scrips"""
        # Simplified mock implementation for demo
        # In real implementation, make API call to Flattrade
        url = f"{self.base_url}/SearchScrip"
        payload = {
            'uid': self.user_id,
            'exch': exchange,
            'stext': searchtext
        }
        # Mock response for demo
        return {
            "stat": "Ok",
            "values": [{"exch": exchange, "token": "123", "tsym": f"{searchtext}-EQ"}]
        }
    
    def get_time_price_series(self, exchange: str, token: str, starttime: float, interval: int = 5):
        """Get historical price data"""
        # Mock implementation - in real app, make actual API call
        url = f"{self.base_url}/TPSeries"
        payload = {
            'uid': self.user_id,
            'exch': exchange,
            'token': token,
            'st': str(int(starttime)),
            'intrv': str(interval)
        }
        
        # Mock candlestick data for demo
        mock_data = []
        base_price = 100
        for i in range(10):
            # Generate random candles with some green/red patterns
            direction = np.random.choice([-1, 1])
            open_price = base_price + np.random.uniform(-2, 2)
            close_price = open_price + direction * np.random.uniform(0.5, 3)
            high_price = max(open_price, close_price) + np.random.uniform(0, 1)
            low_price = min(open_price, close_price) - np.random.uniform(0, 1)
            
            mock_data.append({
                "stat": "Ok",
                "time": (datetime.now() - timedelta(minutes=5*i)).strftime("%d/%m/%Y %H:%M:%S"),
                "into": f"{open_price:.2f}",
                "inth": f"{high_price:.2f}",
                "intl": f"{low_price:.2f}",
                "intc": f"{close_price:.2f}",
                "intv": str(np.random.randint(1000, 5000))
            })
            base_price = close_price
            
        return mock_data[::-1]  # Reverse to get chronological order
    
    def place_order(self, buy_or_sell: str, product_type: str, exchange: str, 
                   tradingsymbol: str, quantity: int, discloseqty: int, 
                   price_type: str, price: float, trigger_price: Optional[float] = None,
                   retention: str = 'DAY', remarks: str = ''):
        """Place order"""
        url = f"{self.base_url}/PlaceOrder"
        payload = {
            'uid': self.user_id,
            'actid': self.user_id,
            'exch': exchange,
            'tsym': tradingsymbol,
            'qty': str(quantity),
            'dscqty': str(discloseqty),
            'prc': str(price),
            'prd': product_type,
            'trantype': buy_or_sell,
            'prctyp': price_type,
            'ret': retention,
            'remarks': remarks
        }
        
        if trigger_price:
            payload['trgprc'] = str(trigger_price)
            
        # Mock response for demo
        return {
            "stat": "Ok",
            "norenordno": f"ORD{int(time.time())}",
            "request_time": datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        }

# Initialize API
@st.cache_resource
def init_api():
    return FlattradeAPI()

def load_stock_list():
    """Load stock list from CSV"""
    # Mock data for demo - replace with actual CSV loading
    mock_data = {
        'Exchange': ['NSE', 'NSE', 'NSE', 'NSE', 'NSE'],
        'Token': ['22', '2885', '11536', '3045', '1594'],
        'Lotsize': [1, 1, 1, 1, 1],
        'Symbol': ['ACC', 'RELIANCE', 'TCS', 'HDFCBANK', 'INFY'],
        'Tradingsymbol': ['ACC-EQ', 'RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ', 'INFY-EQ'],
        'Instrument': ['EQ', 'EQ', 'EQ', 'EQ', 'EQ']
    }
    return pd.DataFrame(mock_data)

def analyze_candles(price_data: List[Dict]) -> Dict:
    """Analyze candlestick patterns for consecutive green/red candles"""
    if len(price_data) < 2:
        return {"pattern": "insufficient_data", "signal": None}
    
    # Get last 3 candles for pattern analysis
    recent_candles = price_data[-3:] if len(price_data) >= 3 else price_data
    
    candle_colors = []
    for candle in recent_candles:
        open_price = float(candle['into'])
        close_price = float(candle['intc'])
        if close_price > open_price:
            candle_colors.append('green')
        elif close_price < open_price:
            candle_colors.append('red')
        else:
            candle_colors.append('doji')
    
    # Check for 2 consecutive patterns
    if len(candle_colors) >= 2:
        last_two = candle_colors[-2:]
        if last_two == ['green', 'green']:
            return {
                "pattern": "two_green", 
                "signal": "BUY",
                "last_price": float(recent_candles[-1]['intc'])
            }
        elif last_two == ['red', 'red']:
            return {
                "pattern": "two_red", 
                "signal": "SELL",
                "last_price": float(recent_candles[-1]['intc'])
            }
    
    return {"pattern": "no_signal", "signal": None}

def screen_stocks(api, stock_df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """Screen stocks for trading signals"""
    results = []
    
    # Calculate start time for historical data
    start_time = (datetime.now() - timedelta(days=lookback_days)).timestamp()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in stock_df.iterrows():
        progress = (idx + 1) / len(stock_df)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {row['Symbol']} ({idx+1}/{len(stock_df)})")
        
        try:
            # Get historical data
            price_data = api.get_time_price_series(
                exchange=row['Exchange'],
                token=row['Token'],
                starttime=start_time,
                interval=5  # 5-minute candles
            )
            
            # Analyze patterns
            analysis = analyze_candles(price_data)
            
            result = {
                'Symbol': row['Symbol'],
                'Tradingsymbol': row['Tradingsymbol'],
                'Exchange': row['Exchange'],
                'Token': row['Token'],
                'Pattern': analysis['pattern'],
                'Signal': analysis.get('signal', 'NONE'),
                'Last_Price': analysis.get('last_price', 0),
                'Candles_Count': len(price_data)
            }
            results.append(result)
            
        except Exception as e:
            st.error(f"Error analyzing {row['Symbol']}: {str(e)}")
            continue
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def place_orders_batch(api, signals_df: pd.DataFrame, order_params: Dict):
    """Place orders for stocks with signals"""
    order_results = []
    
    buy_signals = signals_df[signals_df['Signal'] == 'BUY']
    sell_signals = signals_df[signals_df['Signal'] == 'SELL']
    
    st.subheader("🟢 Buy Orders")
    if not buy_signals.empty:
        for _, row in buy_signals.iterrows():
            try:
                order_price = row['Last_Price'] * (1 + order_params['price_buffer'] / 100)
                
                result = api.place_order(
                    buy_or_sell='B',
                    product_type=order_params['product_type'],
                    exchange=row['Exchange'],
                    tradingsymbol=row['Tradingsymbol'],
                    quantity=order_params['quantity'],
                    discloseqty=0,
                    price_type='LMT',
                    price=round(order_price, 2),
                    retention='DAY',
                    remarks=f"Auto_Buy_{row['Pattern']}"
                )
                
                order_results.append({
                    'Symbol': row['Symbol'],
                    'Action': 'BUY',
                    'Price': round(order_price, 2),
                    'Status': result.get('stat', 'Unknown'),
                    'Order_ID': result.get('norenordno', 'N/A')
                })
                
                st.success(f"✅ BUY order placed for {row['Symbol']} at ₹{order_price:.2f}")
                
            except Exception as e:
                st.error(f"❌ Failed to place BUY order for {row['Symbol']}: {str(e)}")
                order_results.append({
                    'Symbol': row['Symbol'],
                    'Action': 'BUY',
                    'Price': 0,
                    'Status': 'Failed',
                    'Order_ID': 'Error'
                })
    
    st.subheader("🔴 Sell Orders")
    if not sell_signals.empty:
        for _, row in sell_signals.iterrows():
            try:
                order_price = row['Last_Price'] * (1 - order_params['price_buffer'] / 100)
                
                result = api.place_order(
                    buy_or_sell='S',
                    product_type=order_params['product_type'],
                    exchange=row['Exchange'],
                    tradingsymbol=row['Tradingsymbol'],
                    quantity=order_params['quantity'],
                    discloseqty=0,
                    price_type='LMT',
                    price=round(order_price, 2),
                    retention='DAY',
                    remarks=f"Auto_Sell_{row['Pattern']}"
                )
                
                order_results.append({
                    'Symbol': row['Symbol'],
                    'Action': 'SELL',
                    'Price': round(order_price, 2),
                    'Status': result.get('stat', 'Unknown'),
                    'Order_ID': result.get('norenordno', 'N/A')
                })
                
                st.success(f"✅ SELL order placed for {row['Symbol']} at ₹{order_price:.2f}")
                
            except Exception as e:
                st.error(f"❌ Failed to place SELL order for {row['Symbol']}: {str(e)}")
                order_results.append({
                    'Symbol': row['Symbol'],
                    'Action': 'SELL',
                    'Price': 0,
                    'Status': 'Failed',
                    'Order_ID': 'Error'
                })
    
    return pd.DataFrame(order_results)

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Flattrade Stock Screener & Auto Trader",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Flattrade Stock Screener & Auto Trader")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration
    st.sidebar.subheader("🔐 API Settings")
    user_id = st.sidebar.text_input("User ID", value=st.secrets.get("FLATTRADE_USER_ID", "FZ03508"))
    user_session = st.sidebar.text_input("User Session", value=st.secrets.get("FLATTRADE_USER_SESSION", ""), type="password")
    
    # Trading Parameters
    st.sidebar.subheader("📊 Trading Parameters")
    quantity = st.sidebar.number_input("Order Quantity", min_value=1, max_value=1000, value=1)
    price_buffer = st.sidebar.slider("Price Buffer (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    product_type = st.sidebar.selectbox("Product Type", ["C", "M", "I"], index=0, 
                                       help="C=Cash, M=Margin, I=Intraday")
    lookback_days = st.sidebar.slider("Analysis Period (days)", min_value=1, max_value=30, value=5)
    
    # Risk Management
    st.sidebar.subheader("⚠️ Risk Management")
    max_orders = st.sidebar.number_input("Max Orders per Session", min_value=1, max_value=50, value=10)
    enable_auto_trade = st.sidebar.checkbox("Enable Auto Trading", value=False, 
                                           help="⚠️ Use with caution! Test thoroughly first.")
    
    # Initialize API
    api = init_api()
    
    if user_id and user_session:
        api.set_session(user_id, user_session)
        st.sidebar.success("✅ API Session Active")
    else:
        st.sidebar.error("❌ Please provide API credentials")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Stock List")
        
        # Load stock data
        if st.button("🔄 Load Stock List", type="primary"):
            stock_df = load_stock_list()
            st.session_state.stock_df = stock_df
            st.success(f"Loaded {len(stock_df)} stocks")
        
        if 'stock_df' in st.session_state:
            st.dataframe(st.session_state.stock_df, use_container_width=True)
    
    with col2:
        st.header("🎯 Quick Actions")
        
        if st.button("🔍 Screen Stocks", type="primary"):
            if 'stock_df' not in st.session_state:
                st.error("Please load stock list first")
                return
                
            with st.spinner("Analyzing stocks..."):
                results_df = screen_stocks(api, st.session_state.stock_df, lookback_days)
                st.session_state.screening_results = results_df
            
            st.success("Screening completed!")
        
        if st.button("📈 View Signals"):
            if 'screening_results' not in st.session_state:
                st.error("Please run screening first")
                return
            
            st.session_state.show_signals = True
    
    # Display screening results
    if 'screening_results' in st.session_state:
        st.header("📊 Screening Results")
        
        results_df = st.session_state.screening_results
        
        # Filter for signals only
        signals_df = results_df[results_df['Signal'].isin(['BUY', 'SELL'])]
        
        if not signals_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = len(signals_df[signals_df['Signal'] == 'BUY'])
                st.metric("🟢 Buy Signals", buy_count)
            
            with col2:
                sell_count = len(signals_df[signals_df['Signal'] == 'SELL'])
                st.metric("🔴 Sell Signals", sell_count)
            
            with col3:
                total_signals = len(signals_df)
                st.metric("📊 Total Signals", total_signals)
            
            st.subheader("🎯 Trading Signals")
            st.dataframe(signals_df, use_container_width=True)
            
            # Auto trading section
            if enable_auto_trade and total_signals > 0:
                st.subheader("🤖 Auto Trading")
                
                if total_signals > max_orders:
                    st.warning(f"Found {total_signals} signals but limited to {max_orders} orders")
                    signals_df = signals_df.head(max_orders)
                
                if st.button("🚀 Execute Orders", type="primary"):
                    order_params = {
                        'quantity': quantity,
                        'price_buffer': price_buffer,
                        'product_type': product_type
                    }
                    
                    with st.spinner("Placing orders..."):
                        order_results = place_orders_batch(api, signals_df, order_params)
                        st.session_state.order_results = order_results
                    
                    st.success("Order execution completed!")
        else:
            st.info("No trading signals found in current screening")
    
    # Display order results
    if 'order_results' in st.session_state:
        st.header("📋 Order Results")
        st.dataframe(st.session_state.order_results, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This is a demo application for educational purposes. 
    Always test thoroughly and understand the risks before using automated trading.
    
    **Risk Warning:** Trading carries significant financial risk. Never trade with money you cannot afford to lose.
    """)

if __name__ == "__main__":
    main()
