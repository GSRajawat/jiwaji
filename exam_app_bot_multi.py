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
USER_SESSION = "f4ef64a7f58a248c6611b3bfc027cbbb8ae2fcf86ae6f7b57d1fabbf74878f1a"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2"  # Replace with actual password

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flattrade API
@st.cache_resource
def init_flattrade_api():
    api = NorenApiPy()
    # Set session with userid, usertoken, and password
    api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=FLATTRADE_PASSWORD)
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

def calculate_vwap_bands(df, period=20, multiplier=1.0):
    """Calculate VWAP and standard deviation bands"""
    # Calculate VWAP
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PV'] = df['TypicalPrice'] * df['Volume']
    df['VWAP'] = df['PV'].rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    
    # Calculate standard deviation
    df['VWAP_STD'] = df['TypicalPrice'].rolling(window=period).std()
    
    # Calculate bands
    df['SDVWAP1_plus'] = df['VWAP'] + (df['VWAP_STD'] * multiplier)
    df['SDVWAP1_minus'] = df['VWAP'] - (df['VWAP_STD'] * multiplier)
    
    return df

def get_token(api, symbol="OLAELEC", exchange="NSE"):
    """Get token for the symbol"""
    try:
        # Search for the symbol
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result and len(result['values']) > 0:
            # Find the exact match
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ":
                    return item.get('token')
        return None
    except Exception as e:
        st.error(f"Error getting token: {e}")
        return None

def load_market_data(api, symbol="OLAELEC", exchange="NSE"):
    """Load real market data from Flattrade API"""
    try:
        # Get token for the symbol
        token = get_token(api, symbol, exchange)
        if not token:
            st.error(f"Could not find token for {symbol}")
            return None
        
        # Get historical data - last 100 minutes
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=200)  # Get more data to ensure we have enough
        
        # Convert dates to required format
        start_time = start_date.strftime("%d-%m-%Y") + " 09:15:00"
        end_time = end_date.strftime("%d-%m-%Y") + " " + end_date.strftime("%H:%M:%S")
        
        # Get historical data
        hist_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_time,
            endtime=end_time,
            interval='1'  # 1-minute interval
        )
        
        if not hist_data:
            st.error("No historical data received")
            return None
        
        # Convert to DataFrame
        data_list = []
        for item in hist_data:
            data_list.append({
                'Date': pd.to_datetime(item['time'], format='%d-%m-%Y %H:%M:%S'),
                'Open': float(item['into']),
                'High': float(item['inth']),
                'Low': float(item['intl']),
                'Close': float(item['intc']),
                'Volume': int(item.get('intv', 0))
            })
        
        if not data_list:
            st.error("No valid data points found")
            return None
        
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Calculate VWAP bands
        df = calculate_vwap_bands(df)
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        return df.tail(100)  # Return last 100 minutes
        
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        logging.error(f"Error details: {str(e)}")
        return None

def get_live_price(api, symbol="OLAELEC", exchange="NSE"):
    """Get current live price"""
    try:
        token = get_token(api, symbol, exchange)
        if not token:
            return None
        
        # Get live feed
        live_data = api.get_quotes(exchange=exchange, token=token)
        if live_data and live_data.get('stat') == 'Ok':
            return float(live_data.get('lp', 0))
        
        return None
    except Exception as e:
        st.error(f"Error getting live price: {e}")
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
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'],
                            mode='lines', name='VWAP',
                            line=dict(color='blue', width=2)), row=1, col=1)
    
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
    
    fig.update_layout(title='OLAELEC Trading Strategy - Live Data',
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
    
    st.title("OLAELEC Trading Strategy App - Live Data")
    
    # Initialize connections
    try:
        supabase = init_supabase()
        api = init_flattrade_api()
        strategy = OLAELECStrategy()
        
        # Test API connection
        try:
            api_status = api.get_limits()
            if api_status and api_status.get('stat') != 'Ok':
                st.error("Failed to connect to Flattrade API. Please check credentials.")
                return
        except Exception as e:
            st.error(f"API connection failed: {e}")
            return
            
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
        symbol = st.text_input("Trading Symbol", value="OLAELEC")
        
        # Auto-trading toggle
        auto_trading = st.toggle("Enable Auto Trading", False)
        
        # Manual trade buttons
        st.subheader("Manual Trading")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Manual Sell", type="secondary"):
                if auto_trading:
                    result = execute_trade(api, 'SELL', quantity, f"{symbol}-EQ")
                    if result and result.get('stat') == 'Ok':
                        st.success(f"Sell order placed: {result.get('norenordno')}")
                    else:
                        st.error("Failed to place sell order")
        
        with col2:
            if st.button("🟢 Manual Buy", type="primary"):
                if auto_trading:
                    result = execute_trade(api, 'BUY', quantity, f"{symbol}-EQ")
                    if result and result.get('stat') == 'Ok':
                        st.success(f"Buy order placed: {result.get('norenordno')}")
                    else:
                        st.error("Failed to place buy order")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Live Market Data & Signals")
        
        # Load and display market data
        with st.spinner("Loading live market data..."):
            df = load_market_data(api, symbol)
        
        if df is not None and len(df) > 0:
            # Get current signals
            current_time = datetime.now().time()
            current_date = datetime.now().date()
            
            strategy.reset_daily_flags(current_date)
            
            # Get live price
            live_price = get_live_price(api, symbol)
            
            if strategy.check_time_constraints(current_time):
                buy_signal, sell_signal = strategy.get_signals(df)
                
                # Display current signals
                signal_col1, signal_col2, signal_col3 = st.columns(3)
                
                with signal_col1:
                    if buy_signal:
                        st.success("🟢 BUY SIGNAL ACTIVE")
                    else:
                        st.info("⚪ No Buy Signal")
                
                with signal_col2:
                    if sell_signal:
                        st.error("🔴 SELL SIGNAL ACTIVE")
                    else:
                        st.info("⚪ No Sell Signal")
                
                with signal_col3:
                    current_price = live_price if live_price else df['Close'].iloc[-1]
                    st.metric("Live Price", f"₹{current_price:.2f}")
                
                # Auto-execute trades
                if auto_trading:
                    if buy_signal and not strategy.current_position:
                        result = execute_trade(api, 'BUY', quantity, f"{symbol}-EQ")
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = 'long'
                            strategy.position_size = quantity
                            strategy.entry_price = current_price
                            st.success("✅ Auto Buy Order Executed!")
                    
                    elif sell_signal and not strategy.current_position:
                        result = execute_trade(api, 'SELL', quantity, f"{symbol}-EQ")
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = 'short'
                            strategy.position_size = quantity
                            strategy.entry_price = current_price
                            st.success("✅ Auto Sell Order Executed!")
                    
                    # Check exit conditions
                    if strategy.check_exit_conditions(current_price):
                        exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                        result = execute_trade(api, exit_signal, strategy.position_size, f"{symbol}-EQ")
                        if result and result.get('stat') == 'Ok':
                            strategy.current_position = None
                            strategy.position_size = 0
                            strategy.entry_price = None
                            strategy.target_hit_today = True
                            st.success("✅ Position Closed at Target!")
            
            else:
                st.warning("⏰ Outside Trading Hours (9:30 AM - 3:20 PM)")
            
            # Create and display chart
            chart = create_chart(df)
            st.plotly_chart(chart, use_container_width=True)
            
            # Display latest data
            st.subheader("📊 Latest OHLC Data")
            st.dataframe(df.tail(5), use_container_width=True)
        
        else:
            st.error("Failed to load market data or no data available")
    
    with col2:
        st.subheader("📊 Position & Status")
        
        # Current position status
        if strategy.current_position:
            st.info(f"Position: {strategy.current_position.upper()}")
            st.info(f"Size: {strategy.position_size}")
            st.info(f"Entry: ₹{strategy.entry_price:.2f}")
            
            if df is not None:
                live_price = get_live_price(api, symbol)
                current_price = live_price if live_price else df['Close'].iloc[-1]
                
                if strategy.current_position == 'long':
                    pnl = (current_price - strategy.entry_price) * strategy.position_size
                    pnl_pct = (current_price - strategy.entry_price) / strategy.entry_price * 100
                else:
                    pnl = (strategy.entry_price - current_price) * strategy.position_size
                    pnl_pct = (strategy.entry_price - current_price) / strategy.entry_price * 100
                
                if pnl >= 0:
                    st.success(f"P&L: +₹{pnl:.2f} ({pnl_pct:.2f}%)")
                else:
                    st.error(f"P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            st.info("No Open Position")
        
        # Account status
        st.subheader("💰 Account Status")
        try:
            account_details = api.get_limits()
            if account_details and account_details.get('stat') == 'Ok':
                cash = account_details.get('cash', 'N/A')
                margin_used = account_details.get('marginused', 'N/A')
                
                st.info(f"Cash: ₹{cash}")
                st.info(f"Margin Used: ₹{margin_used}")
            else:
                st.warning("Could not fetch account details")
        except Exception as e:
            st.warning(f"Could not fetch account details: {e}")
        
        # Recent trades
        st.subheader("📋 Recent Orders")
        try:
            orders = api.get_order_book()
            if orders and len(orders) > 0:
                # Convert to DataFrame and show relevant columns
                orders_df = pd.DataFrame(orders)
                display_cols = ['tsym', 'trantype', 'qty', 'prc', 'status']
                available_cols = [col for col in display_cols if col in orders_df.columns]
                
                if available_cols:
                    st.dataframe(orders_df[available_cols].head(5), use_container_width=True)
                else:
                    st.dataframe(orders_df.head(5), use_container_width=True)
            else:
                st.info("No recent orders")
        except Exception as e:
            st.warning(f"Could not fetch orders: {e}")
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="secondary"):
            st.rerun()

if __name__ == "__main__":
    main()
