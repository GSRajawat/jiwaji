
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta
import time as time_module
from supabase import create_client, Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Add the parent directory to the path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
# Set logging level to DEBUG to see all detailed messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "f4ef64a7f58a248c6611b3bfc027cbbb8ae2fcf86ae6f7b57d1fabbf74878f1a")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://zybakxpyibubzjhzdcwl.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU")
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, time, timedelta
import time as time_module
from supabase import create_client, Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import yfinance as yf
from typing import Dict, Optional, Tuple

# Add the parent directory to the path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "")
PASSWORD = st.secrets.get("FLATTRADE_PASSWORD", "")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Supabase credentials not found in secrets")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flattrade API
@st.cache_resource
def init_flattrade_api():
    if not USER_ID or not USER_SESSION or not PASSWORD:
        st.error("Flattrade credentials not found in secrets")
        return None
    
    try:
        api = NorenApiPy()
        login_result = api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=PASSWORD)
        
        if login_result and login_result.get('stat') == 'Ok':
            st.success("✅ Connected to Flattrade API")
            return api
        else:
            st.error(f"Failed to login to Flattrade: {login_result}")
            return None
    except Exception as e:
        st.error(f"Error initializing Flattrade API: {e}")
        return None

class LiveDataProvider:
    def __init__(self, api: NorenApiPy):
        self.api = api
        self.cache = {}
        self.cache_expiry = {}
    
    def get_live_price(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get live price data for a symbol"""
        try:
            # Use Flattrade API to get live quotes
            result = self.api.get_quotes(exchange=exchange, token=symbol)
            
            if result and result.get('stat') == 'Ok':
                return {
                    'symbol': result.get('tsym'),
                    'ltp': float(result.get('lp', 0)),
                    'open': float(result.get('o', 0)),
                    'high': float(result.get('h', 0)),
                    'low': float(result.get('l', 0)),
                    'close': float(result.get('c', 0)),
                    'volume': int(result.get('v', 0)),
                    'timestamp': datetime.now()
                }
            else:
                logging.error(f"Failed to get quotes: {result}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting live price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, exchange: str = "NSE", 
                          interval: str = "1", days: int = 1) -> Optional[pd.DataFrame]:
        """Get historical OHLC data"""
        try:
            # Calculate from and to dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Use Flattrade API to get historical data
            result = self.api.get_time_price_series(
                exchange=exchange,
                token=symbol,
                starttime=from_date.timestamp(),
                endtime=to_date.timestamp(),
                interval=interval
            )
            
            if result and isinstance(result, list) and len(result) > 0:
                df_data = []
                for candle in result:
                    df_data.append({
                        'DateTime': pd.to_datetime(candle.get('time'), unit='s'),
                        'Open': float(candle.get('into', 0)),
                        'High': float(candle.get('inth', 0)),
                        'Low': float(candle.get('intl', 0)),
                        'Close': float(candle.get('intc', 0)),
                        'Volume': int(candle.get('intv', 0))
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('DateTime', inplace=True)
                df = df.sort_index()
                
                # Add VWAP calculations
                df = self.calculate_vwap_bands(df)
                
                return df
            else:
                logging.error(f"No historical data received: {result}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return self.get_fallback_data(symbol)
    
    def get_fallback_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fallback to Yahoo Finance for data"""
        try:
            # Convert NSE symbol to Yahoo Finance format
            yf_symbol = f"{symbol.split('-')[0]}.NS"
            
            # Get 1-minute data for current day
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="1d", interval="1m")
            
            if df.empty:
                return None
            
            # Rename columns to match our format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Add VWAP calculations
            df = self.calculate_vwap_bands(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Fallback data fetch failed: {e}")
            return None
    
    def calculate_vwap_bands(self, df: pd.DataFrame, std_dev: float = 1.0) -> pd.DataFrame:
        """Calculate VWAP and standard deviation bands"""
        try:
            # Calculate VWAP
            df['PV'] = df['Close'] * df['Volume']
            df['Cumulative_PV'] = df['PV'].cumsum()
            df['Cumulative_Volume'] = df['Volume'].cumsum()
            df['VWAP'] = df['Cumulative_PV'] / df['Cumulative_Volume']
            
            # Calculate standard deviation of price from VWAP
            df['Price_Diff_Sq'] = ((df['Close'] - df['VWAP']) ** 2) * df['Volume']
            df['Cumulative_Price_Diff_Sq'] = df['Price_Diff_Sq'].cumsum()
            df['VWAP_StdDev'] = np.sqrt(df['Cumulative_Price_Diff_Sq'] / df['Cumulative_Volume'])
            
            # Calculate VWAP bands
            df['SDVWAP1_plus'] = df['VWAP'] + (std_dev * df['VWAP_StdDev'])
            df['SDVWAP1_minus'] = df['VWAP'] - (std_dev * df['VWAP_StdDev'])
            
            # Clean up temporary columns
            df = df.drop(['PV', 'Cumulative_PV', 'Cumulative_Volume', 
                         'Price_Diff_Sq', 'Cumulative_Price_Diff_Sq'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP bands: {e}")
            return df

class OLAELECStrategy:
    def __init__(self):
        self.profit_target = 0.04  # 4% profit target
        self.entry_price = None
        self.current_position = None
        self.position_size = 0
        self.last_exit_date = None
        self.last_exit_direction = None
        self.target_hit_today = False
        self.current_date = None
        self.entry_time = None
        
    def reset_daily_flags(self, current_date):
        """Reset daily flags for new trading day"""
        if self.current_date != current_date:
            self.current_date = current_date
            self.target_hit_today = False
            self.last_exit_date = None
            self.last_exit_direction = None
    
    def check_time_constraints(self, current_time):
        """Check if current time is within trading hours"""
        market_open = time(9, 30)
        market_close = time(15, 20)
        return market_open <= current_time <= market_close
    
    def check_exit_conditions(self, current_price):
        """Check if position should be exited based on profit target"""
        if not self.current_position or not self.entry_price:
            return False
            
        if self.current_position == 'long':
            profit_pct = (current_price - self.entry_price) / self.entry_price
            return profit_pct >= self.profit_target
        elif self.current_position == 'short':
            profit_pct = (self.entry_price - current_price) / self.entry_price
            return profit_pct >= self.profit_target
        return False
    
    def get_signals(self, df):
        """Generate buy/sell signals based on VWAP strategy"""
        if len(df) < 2:
            return False, False
            
        # Get last two candles
        candle_1 = df.iloc[-1]  # Current candle
        candle_2 = df.iloc[-2]  # Previous candle
        
        # Check sell conditions (short entry)
        sell_condition_1 = (candle_2['Close'] < candle_2['Open'] and 
                           candle_2['Close'] < candle_2['SDVWAP1_minus'])
        sell_condition_2 = (candle_1['Close'] < candle_1['Open'] and 
                           candle_1['Close'] < candle_1['SDVWAP1_minus'])
        
        # Check buy conditions (long entry)
        buy_condition_1 = (candle_2['Close'] > candle_2['Open'] and 
                          candle_2['Close'] > candle_2['SDVWAP1_plus'])
        buy_condition_2 = (candle_1['Close'] > candle_1['Open'] and 
                          candle_1['Close'] > candle_1['SDVWAP1_plus'])
        
        sell_signal = sell_condition_1 and sell_condition_2
        buy_signal = buy_condition_1 and buy_condition_2
        
        return buy_signal, sell_signal

def get_symbol_token(api, symbol: str, exchange: str = "NSE") -> Optional[str]:
    """Get token for a symbol"""
    try:
        # Search for symbol to get token
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        
        if result and isinstance(result, list) and len(result) > 0:
            for item in result:
                if item.get('tsym') == symbol:
                    return item.get('token')
        
        return None
    except Exception as e:
        logging.error(f"Error getting token for {symbol}: {e}")
        return None

def create_live_chart(df, current_price=None, signals_df=None):
    """Create candlestick chart with live data"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=('Price & VWAP Bands', 'Volume', 'VWAP'),
                        row_heights=[0.6, 0.2, 0.2])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OLAELEC',
                                increasing_line_color='green',
                                decreasing_line_color='red'), row=1, col=1)
    
    # VWAP and bands
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'],
                                mode='lines', name='VWAP',
                                line=dict(color='blue', width=2)), row=1, col=1)
    
    if 'SDVWAP1_plus' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SDVWAP1_plus'],
                                mode='lines', name='VWAP+',
                                line=dict(color='green', dash='dash')), row=1, col=1)
    
    if 'SDVWAP1_minus' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SDVWAP1_minus'],
                                mode='lines', name='VWAP-',
                                line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Volume
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        name='Volume', marker_color=colors), row=2, col=1)
    
    # VWAP in separate subplot
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'],
                                mode='lines', name='VWAP Detail',
                                line=dict(color='blue', width=1)), row=3, col=1)
    
    # Add current price line if provided
    if current_price:
        fig.add_hline(y=current_price, line_dash="dot", line_color="orange",
                     annotation_text=f"Live: ₹{current_price:.2f}", row=1, col=1)
    
    # Add buy/sell signals
    if signals_df is not None and len(signals_df) > 0:
        buy_signals = signals_df[signals_df['Signal'] == 'BUY']
        sell_signals = signals_df[signals_df['Signal'] == 'SELL']
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Price'],
                                   mode='markers', name='Buy Signal',
                                   marker=dict(color='green', size=12, symbol='triangle-up')),
                         row=1, col=1)
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Price'],
                                   mode='markers', name='Sell Signal',
                                   marker=dict(color='red', size=12, symbol='triangle-down')),
                         row=1, col=1)
    
    fig.update_layout(title='OLAELEC Live Trading Strategy',
                     xaxis_rangeslider_visible=False,
                     height=800,
                     showlegend=True)
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="VWAP (₹)", row=3, col=1)
    
    return fig

def execute_trade(api, signal_type, quantity, symbol="OLAELEC-EQ"):
    """Execute trade through Flattrade API"""
    try:
        buy_or_sell = 'B' if signal_type == 'BUY' else 'S'
        
        result = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type='C',  # Cash
            exchange='NSE',
            tradingsymbol=symbol,
            quantity=str(quantity),
            discloseqty=0,
            price_type='MKT',  # Market order
            retention='DAY'
        )
        
        logging.info(f"Trade execution result: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

def save_trade_to_db(supabase, trade_data):
    """Save trade data to Supabase"""
    if not supabase:
        return None
        
    try:
        result = supabase.table('trades').insert(trade_data).execute()
        logging.info(f"Trade saved to DB: {result}")
        return result
    except Exception as e:
        logging.error(f"Error saving trade to database: {e}")
        return None

def main():
    st.set_page_config(page_title="OLAELEC Live Trading", layout="wide", 
                      page_icon="📈", initial_sidebar_state="expanded")
    
    st.title("📈 OLAELEC Live Trading Strategy")
    st.caption("Real-time VWAP-based trading system")
    
    # Initialize APIs
    api = init_flattrade_api()
    supabase = init_supabase()
    
    if not api:
        st.error("❌ Failed to initialize Flattrade API. Please check your credentials.")
        st.stop()
    
    # Initialize strategy and data provider
    strategy = OLAELECStrategy()
    data_provider = LiveDataProvider(api)
    
    # Initialize session state
    if 'signals_history' not in st.session_state:
        st.session_state.signals_history = pd.DataFrame(columns=['Time', 'Signal', 'Price'])
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Trading Controls")
        
        # Trading parameters
        st.subheader("Strategy Parameters")
        profit_target = st.slider("Profit Target (%)", 1.0, 10.0, 4.0, 0.1) / 100
        strategy.profit_target = profit_target
        
        quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=1)
        symbol = st.selectbox("Trading Symbol", 
                             ["OLAELEC-EQ", "RELIANCE-EQ", "TCS-EQ", "INFY-EQ"],
                             index=0)
        
        # Data refresh settings
        st.subheader("⚙️ Settings")
        refresh_interval = st.selectbox("Refresh Interval", 
                                      [30, 60, 120, 300], 
                                      index=1, 
                                      format_func=lambda x: f"{x} seconds")
        
        # Auto-trading toggle
        auto_trading = st.toggle("🤖 Enable Auto Trading", False)
        if auto_trading:
            st.warning("⚠️ Auto trading is enabled!")
        
        # Manual trading buttons
        st.subheader("🎯 Manual Trading")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 SELL", type="secondary", use_container_width=True):
                if api:
                    with st.spinner("Executing sell order..."):
                        result = execute_trade(api, 'SELL', quantity, symbol)
                        if result and result.get('stat') == 'Ok':
                            st.success(f"✅ Sell order: {result.get('norenordno')}")
                        else:
                            st.error(f"❌ Sell failed: {result}")
        
        with col2:
            if st.button("🟢 BUY", type="primary", use_container_width=True):
                if api:
                    with st.spinner("Executing buy order..."):
                        result = execute_trade(api, 'BUY', quantity, symbol)
                        if result and result.get('stat') == 'Ok':
                            st.success(f"✅ Buy order: {result.get('norenordno')}")
                        else:
                            st.error(f"❌ Buy failed: {result}")
        
        # Emergency controls
        st.subheader("🚨 Emergency")
        if st.button("🛑 Close All Positions", type="secondary"):
            if strategy.current_position:
                exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                result = execute_trade(api, exit_signal, strategy.position_size, symbol)
                if result and result.get('stat') == 'Ok':
                    strategy.current_position = None
                    strategy.position_size = 0
                    strategy.entry_price = None
                    st.success("✅ All positions closed!")
    
    # Main content area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.subheader("📊 Live Market Data & Signals")
        
        # Get symbol token
        token = get_symbol_token(api, symbol.replace("-EQ", ""))
        
        if token:
            # Get live price data
            live_price_data = data_provider.get_live_price(token)
            
            # Display current status
            status_cols = st.columns(4)
            
            with status_cols[0]:
                if live_price_data:
                    current_price = live_price_data['ltp']
                    prev_close = live_price_data['close']
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    if change >= 0:
                        st.metric("Current Price", f"₹{current_price:.2f}", 
                                f"+₹{change:.2f} ({change_pct:+.2f}%)")
                    else:
                        st.metric("Current Price", f"₹{current_price:.2f}", 
                                f"₹{change:.2f} ({change_pct:.2f}%)")
                else:
                    st.metric("Current Price", "Loading...")
            
            # Get historical data for chart
            df = data_provider.get_historical_data(token, interval="1")
            
            if df is not None and len(df) > 0:
                current_time = datetime.now().time()
                current_date = datetime.now().date()
                
                strategy.reset_daily_flags(current_date)
                
                # Check trading hours and generate signals
                if strategy.check_time_constraints(current_time):
                    with status_cols[1]:
                        st.success("🟢 Market Open")
                    
                    buy_signal, sell_signal = strategy.get_signals(df)
                    
                    with status_cols[2]:
                        if buy_signal:
                            st.success("📈 BUY SIGNAL")
                            # Add to signals history
                            if live_price_data:
                                new_signal = pd.DataFrame({
                                    'Time': [datetime.now()],
                                    'Signal': ['BUY'],
                                    'Price': [live_price_data['ltp']]
                                })
                                st.session_state.signals_history = pd.concat([
                                    st.session_state.signals_history, new_signal
                                ], ignore_index=True)
                        elif sell_signal:
                            st.error("📉 SELL SIGNAL")
                            # Add to signals history
                            if live_price_data:
                                new_signal = pd.DataFrame({
                                    'Time': [datetime.now()],
                                    'Signal': ['SELL'],
                                    'Price': [live_price_data['ltp']]
                                })
                                st.session_state.signals_history = pd.concat([
                                    st.session_state.signals_history, new_signal
                                ], ignore_index=True)
                        else:
                            st.info("⏸️ No Signal")
                    
                    with status_cols[3]:
                        if strategy.current_position:
                            if strategy.current_position == 'long':
                                st.info("📈 LONG Position")
                            else:
                                st.info("📉 SHORT Position")
                        else:
                            st.info("🔄 No Position")
                    
                    # Auto-execute trades
                    if auto_trading and live_price_data:
                        current_price = live_price_data['ltp']
                        
                        if buy_signal and not strategy.current_position:
                            with st.spinner("Executing auto buy..."):
                                result = execute_trade(api, 'BUY', quantity, symbol)
                                if result and result.get('stat') == 'Ok':
                                    strategy.current_position = 'long'
                                    strategy.position_size = quantity
                                    strategy.entry_price = current_price
                                    strategy.entry_time = datetime.now()
                                    st.success("✅ Auto Buy Executed!")
                                    
                                    # Save to database
                                    if supabase:
                                        trade_data = {
                                            'symbol': symbol,
                                            'action': 'BUY',
                                            'quantity': quantity,
                                            'price': current_price,
                                            'timestamp': datetime.now().isoformat(),
                                            'order_id': result.get('norenordno')
                                        }
                                        save_trade_to_db(supabase, trade_data)
                        
                        elif sell_signal and not strategy.current_position:
                            with st.spinner("Executing auto sell..."):
                                result = execute_trade(api, 'SELL', quantity, symbol)
                                if result and result.get('stat') == 'Ok':
                                    strategy.current_position = 'short'
                                    strategy.position_size = quantity
                                    strategy.entry_price = current_price
                                    strategy.entry_time = datetime.now()
                                    st.success("✅ Auto Sell Executed!")
                                    
                                    # Save to database
                                    if supabase:
                                        trade_data = {
                                            'symbol': symbol,
                                            'action': 'SELL',
                                            'quantity': quantity,
                                            'price': current_price,
                                            'timestamp': datetime.now().isoformat(),
                                            'order_id': result.get('norenordno')
                                        }
                                        save_trade_to_db(supabase, trade_data)
                        
                        # Check exit conditions
                        if strategy.check_exit_conditions(current_price):
                            exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                            with st.spinner("Closing position at target..."):
                                result = execute_trade(api, exit_signal, strategy.position_size, symbol)
                                if result and result.get('stat') == 'Ok':
                                    # Save exit trade
                                    if supabase:
                                        trade_data = {
                                            'symbol': symbol,
                                            'action': exit_signal,
                                            'quantity': strategy.position_size,
                                            'price': current_price,
                                            'timestamp': datetime.now().isoformat(),
                                            'order_id': result.get('norenordno'),
                                            'exit_reason': 'TARGET_HIT'
                                        }
                                        save_trade_to_db(supabase, trade_data)
                                    
                                    strategy.current_position = None
                                    strategy.position_size = 0
                                    strategy.entry_price = None
                                    strategy.target_hit_today = True
                                    st.success("🎯 Position Closed at Target!")
                else:
                    with status_cols[1]:
                        st.error("🔴 Market Closed")
                    with status_cols[2:]:
                        for col in status_cols[2:]:
                            with col:
                                st.info("⏰ Outside Trading Hours")
                
                # Create and display chart
                chart_current_price = live_price_data['ltp'] if live_price_data else None
                chart = create_live_chart(df, chart_current_price, st.session_state.signals_history)
                st.plotly_chart(chart, use_container_width=True)
                
                # Display recent data table
                with st.expander("📋 Recent Candles Data", expanded=False):
                    st.dataframe(df.tail(10).round(2), use_container_width=True)
            
            else:
                st.error("❌ Failed to load market data")
        else:
            st.error(f"❌ Could not find token for symbol: {symbol}")
    
    with col2:
        st.subheader("💼 Position & Portfolio")
        
        # Current position display
        if strategy.current_position and live_price_data:
            st.info(f"**Position:** {strategy.current_position.upper()}")
            st.info(f"**Size:** {strategy.position_size}")
            st.info(f"**Entry:** ₹{strategy.entry_price:.2f}")
            
            if strategy.entry_time:
                duration = datetime.now() - strategy.entry_time
                st.info(f"**Duration:** {str(duration).split('.')[0]}")
            
            current_price = live_price_data['ltp']
            if strategy.current_position == 'long':
                pnl = (current_price - strategy.entry_price) * strategy.position_size
                pnl_pct = (current_price - strategy.entry_price) / strategy.entry_price * 100
            else:
                pnl = (strategy.entry_price - current_price) * strategy.position_size
                pnl_pct = (strategy.entry_price - current_price) / strategy.entry_price * 100
            
            if pnl >= 0:
                st.success(f"**P&L:** +₹{pnl:.2f} ({pnl_pct:+.2f}%)")
            else:
                st.error(f"**P&L:** ₹{pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Progress bar for profit target
            progress = min(abs(pnl_pct) / (strategy.profit_target * 100), 1.0)
            st.progress(progress, text=f"Target Progress: {progress*100:.1f}%")
        
        else:
            st.info("**No Open Position**")
        
        # Account information
        st.subheader("💰 Account Status")
        try:
            limits = api.get_limits()
            if limits and limits.get('stat') == 'Ok':
                cash_available = float(limits.get('cash', 0))
                margin_used = float(limits.get('marginused', 0))
                
                st.metric("Available Cash", f"₹{cash_available:,.2f}")
                st.metric("Margin Used", f"₹{margin_used:,.2f}")
                
                # Portfolio value calculation
                portfolio_value = cash_available + margin_used
                st.metric("Portfolio Value", f"₹{portfolio_value:,.2f}")
            else:
                st.warning("Could not fetch account details")
                
        except Exception as e:
            st.warning(f"Account data unavailable: {str(e)[:50]}...")
        
        # Trading statistics
        st.subheader("📊 Today's Stats")
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            buy_signals = len(st.session_state.signals_history[
                st.session_state.signals_history['Signal'] == 'BUY'
            ])
            st.metric("Buy Signals", buy_signals)
        
        with col_stats2:
            sell_signals = len(st.session_state.signals_history[
                st.session_state.signals_history['Signal'] == 'SELL'
            ])
            st.metric("Sell Signals", sell_signals)
        
        # Recent orders
        st.subheader("📋 Recent Orders")
        try:
            order_book = api.get_order_book()
            if order_book and len(order_book) > 0:
                orders_df = pd.DataFrame(order_book[:5])
                
                # Select relevant columns
                display_columns = []
                if 'tsym' in orders_df.columns:
                    display_columns.append('tsym')
                if 'trantype' in orders_df.columns:
                    display_columns.append('trantype')
                if 'qty' in orders_df.columns:
                    display_columns.append('qty')
                if 'status' in orders_df.columns:
                    display_columns.append('status')
                if 'avgprc' in orders_df.columns:
                    display_columns.append('avgprc')
                
                if display_columns:
                    st.dataframe(orders_df[display_columns], use_container_width=True, height=200)
                else:
                    st.dataframe(orders_df.head(), use_container_width=True, height=200)
            else:
                st.info("No recent orders")
                
        except Exception as e:
            st.warning(f"Orders unavailable: {str(e)[:50]}...")
        
        # Signal history
        st.subheader("🔔 Signal History")
        if len(st.session_state.signals_history) > 0:
            recent_signals = st.session_state.signals_history.tail(5).copy()
            recent_signals['Time'] = recent_signals['Time'].dt.strftime('%H:%M:%S')
            st.dataframe(recent_signals, use_container_width=True, height=150)
        else:
            st.info("No signals generated yet")
        
        # Market status
        st.subheader("🕐 Market Status")
        current_time = datetime.now().time()
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        if market_open <= current_time <= market_close:
            st.success("🟢 Market is Open")
            time_to_close = datetime.combine(datetime.today(), market_close) - datetime.now()
            st.info(f"⏰ Closes in: {str(time_to_close).split('.')[0]}")
        else:
            st.error("🔴 Market is Closed")
            if current_time < market_open:
                time_to_open = datetime.combine(datetime.today(), market_open) - datetime.now()
                st.info(f"⏰ Opens in: {str(time_to_open).split('.')[0]}")
        
        # System status
        st.subheader("⚙️ System Status")
        st.success("🟢 Flattrade API Connected")
        
        if supabase:
            st.success("🟢 Database Connected")
        else:
            st.warning("🟡 Database Disconnected")
        
        # Refresh controls
        st.subheader("🔄 Data Refresh")
        col_refresh1, col_refresh2 = st.columns(2)
        
        with col_refresh1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        with col_refresh2:
            if st.button("🗑️ Clear Signals", use_container_width=True):
                st.session_state.signals_history = pd.DataFrame(columns=['Time', 'Signal', 'Price'])
                st.success("Signals cleared!")
                st.rerun()
    
    # Footer with auto-refresh
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    with col_footer2:
        st.caption(f"📊 Data Points: {len(df) if 'df' in locals() and df is not None else 0}")
    
    with col_footer3:
        st.caption(f"🔄 Auto-refresh: {refresh_interval}s")
    
    # Auto-refresh mechanism
    if auto_trading or st.button("🔄 Enable Auto Refresh"):
        time_module.sleep(refresh_interval)
        st.rerun()

# Additional utility functions
def format_indian_currency(amount):
    """Format currency in Indian format"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f}L"
    elif amount >= 1000:  # 1 thousand
        return f"₹{amount/1000:.2f}K"
    else:
        return f"₹{amount:.2f}"

def get_market_sentiment(df):
    """Calculate basic market sentiment"""
    if df is None or len(df) < 5:
        return "Neutral"
    
    recent_closes = df['Close'].tail(5)
    if recent_closes.iloc[-1] > recent_closes.iloc[0]:
        return "Bullish"
    elif recent_closes.iloc[-1] < recent_closes.iloc[0]:
        return "Bearish"
    else:
        return "Neutral"

def calculate_volatility(df, periods=20):
    """Calculate price volatility"""
    if df is None or len(df) < periods:
        return 0
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.tail(periods).std() * np.sqrt(252)  # Annualized
    return volatility * 100

# Risk management functions
def check_risk_limits(strategy, current_price, max_loss_pct=2.0):
    """Check if risk limits are breached"""
    if not strategy.current_position or not strategy.entry_price:
        return False, ""
    
    if strategy.current_position == 'long':
        loss_pct = (strategy.entry_price - current_price) / strategy.entry_price * 100
    else:
        loss_pct = (current_price - strategy.entry_price) / strategy.entry_price * 100
    
    if loss_pct > max_loss_pct:
        return True, f"Stop loss triggered: -{loss_pct:.2f}%"
    
    return False, ""

# Enhanced error handling and logging
def safe_api_call(func, *args, **kwargs):
    """Safely execute API calls with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"API call failed: {func.__name__} - {str(e)}")
        return None

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Main application error: {e}")
        st.info("Please refresh the page and try again.")
