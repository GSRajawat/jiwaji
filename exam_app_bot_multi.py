
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
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "f68b270591263a92f1d4182a6a5397142b0c254bdf885738c55d854445b3ac9c")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

# --- Supabase Credentials ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://zybakxpyibubzjhzdcwl.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flattrade API
@st.cache_resource
def init_flattrade_api():
    api = NorenApiPy()
    # Set session with userid, usertoken, and password
    # The password must be provided through a secure method, like Streamlit secrets
    PASSWORD = st.secrets.get("FLATTRADE_PASSWORD") 
    api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=PASSWORD)
    return api

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
        
    def reset_daily_flags(self, current_date):
        """Reset daily flags for new trading day"""
        if self.current_date != current_date:
            self.current_date = current_date
            self.target_hit_today = False
            self.last_exit_date = None
            self.last_exit_direction = None
    
    def check_time_constraints(self, current_time):
        """Check if current time is within trading hours"""
        if current_time <= time(9, 30) or current_time >= time(15, 20):
            return False
        return True
    
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
            return None, None
            
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

def load_market_data():
    """Load market data from CSV or database"""
    try:
        # This would typically load real-time data
        # For demo purposes, we'll create sample data
        dates = pd.date_range(start='2025-09-09 09:15:00', periods=100, freq='1Min')
        
        # Generate sample OHLC data with VWAP bands
        np.random.seed(42)
        base_price = 60.0
        data = []
        
        for i, date in enumerate(dates):
            price_change = np.random.normal(0, 0.5)
            open_price = base_price + price_change
            high_price = open_price + abs(np.random.normal(0, 0.3))
            low_price = open_price - abs(np.random.normal(0, 0.3))
            close_price = open_price + np.random.normal(0, 0.2)
            volume = np.random.randint(1000, 10000)
            
            # Simple VWAP bands (would be calculated properly in real implementation)
            vwap_plus = close_price * 1.02
            vwap_minus = close_price * 0.98
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'SDVWAP1_plus': vwap_plus,
                'SDVWAP1_minus': vwap_minus
            })
            
            base_price = close_price
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
        
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None

def create_chart(df, signals_df=None):
    """Create candlestick chart with VWAP bands and signals"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=('Price & VWAP Bands', 'Volume'),
                        row_heights=[0.8, 0.2])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OLAELEC'), row=1, col=1)
    
    # VWAP bands
    fig.add_trace(go.Scatter(x=df.index, y=df['SDVWAP1_plus'],
                            mode='lines', name='VWAP+',
                            line=dict(color='green', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SDVWAP1_minus'],
                            mode='lines', name='VWAP-',
                            line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        name='Volume'), row=2, col=1)
    
    # Add buy/sell signals if provided
    if signals_df is not None and len(signals_df) > 0:
        buy_signals = signals_df[signals_df['Signal'] == 'BUY']
        sell_signals = signals_df[signals_df['Signal'] == 'SELL']
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(x=buy_signals['Time'], y=buy_signals['Price'],
                                   mode='markers', name='Buy Signal',
                                   marker=dict(color='green', size=10, symbol='triangle-up')),
                         row=1, col=1)
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(x=sell_signals['Time'], y=sell_signals['Price'],
                                   mode='markers', name='Sell Signal',
                                   marker=dict(color='red', size=10, symbol='triangle-down')),
                         row=1, col=1)
    
    fig.update_layout(title='OLAELEC Trading Strategy',
                     xaxis_rangeslider_visible=False,
                     height=800)
    
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
            quantity=quantity,
            discloseqty=0,
            price_type='MKT',  # Market order
            retention='DAY'
        )
        
        return result
    except Exception as e:
        st.error(f"Error executing trade: {e}")
        return None

def save_trade_to_db(supabase, trade_data):
    """Save trade data to Supabase"""
    try:
        result = supabase.table('trades').insert(trade_data).execute()
        return result
    except Exception as e:
        st.error(f"Error saving trade to database: {e}")
        return None

def main():
    st.set_page_config(page_title="OLAELEC Trading App", layout="wide")
    
    st.title("OLAELEC Trading Strategy App")
    
    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    # Initialize connections
    try:
        supabase = init_supabase()
        strategy = OLAELECStrategy()
        
        # Handle API initialization
        if not st.session_state['logged_in']:
            api = init_flattrade_api()
            if api is None:
                # Show login form
                api = login_to_flattrade()
                if api is None:
                    st.stop()
        else:
            api = init_flattrade_api()
            
    except Exception as e:
        st.error(f"Error initializing APIs: {e}")
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Trading Controls")
        
        # Trading parameters
        st.subheader("Strategy Parameters")
        profit_target = st.slider("Profit Target (%)", 1, 10, 4) / 100
        strategy.profit_target = profit_target
        
        quantity = st.number_input("Quantity", min_value=1, value=1)
        symbol = st.text_input("Trading Symbol", value="OLAELEC-EQ")
        
        # Auto-trading toggle
        auto_trading = st.toggle("Enable Auto Trading", False)
        
        # Manual trade buttons
        st.subheader("Manual Trading")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("ðŸ”´ Manual Sell", type="secondary"):
                if auto_trading:
                    result = execute_trade(api, 'SELL', quantity, symbol)
                    if result and result.get('stat') == 'Ok':
                        st.success(f"Sell order placed: {result.get('norenordno')}")
                    else:
                        st.error("Failed to place sell order")
        
        with col2:
            if st.button("ðŸŸ¢ Manual Buy", type="primary"):
                if auto_trading:
                    result = execute_trade(api, 'BUY', quantity, symbol)
                    if result and result.get('stat') == 'Ok':
                        st.success(f"Buy order placed: {result.get('norenordno')}")
                    else:
                        st.error("Failed to place buy order")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("ðŸ“ˆ Market Data & Signals")
        
        # Load and display market data
        df = load_market_data()
        if df is not None:
            # Get current signals
            current_time = datetime.now().time()
            current_date = datetime.now().date()
            
            strategy.reset_daily_flags(current_date)
            
            if strategy.check_time_constraints(current_time):
                buy_signal, sell_signal = strategy.get_signals(df)
                
                # Display current signals
                signal_col1, signal_col2, signal_col3 = st.columns(3)
                
                with signal_col1:
                    if buy_signal:
                        st.success("ðŸŸ¢ BUY SIGNAL ACTIVE")
                    else:
                        st.info("âšª No Buy Signal")
                
                with signal_col2:
                    if sell_signal:
                        st.error("ðŸ”´ SELL SIGNAL ACTIVE")
                    else:
                        st.info("âšª No Sell Signal")
                
                with signal_col3:
                    current_price = df['Close'].iloc[-1]
                    st.metric("Current Price", f"â‚¹{current_price:.2f}")
                
                # Auto-execute trades
                if auto_trading:
                    if buy_signal and not strategy.current_position:
                        result = execute_trade(api, 'BUY', quantity, symbol)
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = 'long'
                            strategy.position_size = quantity
                            strategy.entry_price = current_price
                            st.success("âœ… Auto Buy Order Executed!")
                    
                    elif sell_signal and not strategy.current_position:
                        result = execute_trade(api, 'SELL', quantity, symbol)
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = 'short'
                            strategy.position_size = quantity
                            strategy.entry_price = current_price
                            st.success("âœ… Auto Sell Order Executed!")
                    
                    # Check exit conditions
                    if strategy.check_exit_conditions(current_price):
                        exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                        result = execute_trade(api, exit_signal, strategy.position_size, symbol)
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = None
                            strategy.position_size = 0
                            strategy.entry_price = None
                            strategy.target_hit_today = True
                            st.success("âœ… Position Closed at Target!")
            
            else:
                st.warning("â° Outside Trading Hours (9:30 AM - 3:20 PM)")
            
            # Create and display chart
            chart = create_chart(df)
            st.plotly_chart(chart, use_container_width=True)
        
        else:
            st.error("Failed to load market data")
    
    with col2:
        st.subheader("ðŸ“Š Position & Status")
        
        # Current position status
        if strategy.current_position:
            st.info(f"Position: {strategy.current_position.upper()}")
            st.info(f"Size: {strategy.position_size}")
            st.info(f"Entry: â‚¹{strategy.entry_price:.2f}")
            
            if df is not None:
                current_price = df['Close'].iloc[-1]
                if strategy.current_position == 'long':
                    pnl = (current_price - strategy.entry_price) * strategy.position_size
                    pnl_pct = (current_price - strategy.entry_price) / strategy.entry_price * 100
                else:
                    pnl = (strategy.entry_price - current_price) * strategy.position_size
                    pnl_pct = (strategy.entry_price - current_price) / strategy.entry_price * 100
                
                if pnl >= 0:
                    st.success(f"P&L: +â‚¹{pnl:.2f} ({pnl_pct:.2f}%)")
                else:
                    st.error(f"P&L: â‚¹{pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            st.info("No Open Position")
        
        # Account status
        # Account status
        # Account status
        st.subheader("ðŸ’° Account Status")
        try:
            # Get account details from the API
            account_details = api.get_limits()
            
            # Print the full response to see the available keys
            st.write(account_details)

            # Use the correct keys from the API response
            cash = account_details.get('cash', 'N/A')
            available = account_details.get('margin', 'N/A')

            st.info(f"Cash: â‚¹{cash}")
            st.info(f"Available: â‚¹{available}")

        except Exception as e:
            st.warning(f"Could not fetch account details: {e}")
        
        # Recent trades
        st.subheader("ðŸ“‹ Recent Trades")
        try:
            # Get order book from API
            orders = api.get_order_book()
            if orders and len(orders) > 0:
                orders_df = pd.DataFrame(orders[:5])  # Show last 5 orders
                st.dataframe(orders_df[['tsym', 'trantype', 'qty', 'status']], 
                           use_container_width=True)
            else:
                st.info("No recent trades")
        except Exception as e:
            st.warning(f"Could not fetch orders: {e}")
        
        # Refresh button
        if st.button("ðŸ”„ Refresh Data", type="secondary"):
            st.rerun()

if __name__ == "__main__":
    main()
