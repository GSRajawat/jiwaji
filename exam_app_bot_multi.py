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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flattrade API Credentials ---
USER_SESSION = "368b75b27845ccb54bfab8732261c2cb25e9dfeaffc4f03e44441cfc532f2c28"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@3"

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
    api.set_session(userid=USER_ID, usertoken=USER_SESSION, password=FLATTRADE_PASSWORD)
    return api

class FnOBreakoutStrategy:
    def __init__(self):
        self.stop_loss_pct = 0.01  # 1% stop loss
        self.target_pct = 0.03  # 3% target
        self.trailing_stop_pct = 0.01  # 1% trailing stop
        self.position_symbol = None
        self.product_type = 'C'  # Default to C
        self.screened_stocks = []
        self.entry_price = None
        self.current_position = None  # 'long' or 'short'
        self.position_size = 0
        self.stop_loss = None
        self.target = None
        self.day_high = None
        self.day_low = None
        self.trailing_stop = None
        
        self.screened_stocks = []
        self.current_date = None
        self.avg_vol_opening = None
        
    def reset_daily_data(self, current_date):
        """Reset daily data for new trading day"""
        if self.current_date != current_date:
            self.current_date = current_date
            self.screened_stocks = []
            self.day_high = None
            self.day_low = None
    
    def calculate_rsi(self, series, period=14):
        """Calculate RSI for momentum check"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_opening_candle_metrics(self, df):
        """
        Calculate metrics for 9:15 to 9:29 period
        Returns: avg_volume_opening, trade_value, is_big_body, opening_candle_data
        """
        # Filter data for 9:15 to 9:29
        opening_data = df[(df.index.time >= time(9, 15)) & (df.index.time <= time(9, 29))]
        
        if len(opening_data) == 0:
            return None, None, None, None
        
        # Calculate average volume for opening period
        avg_volume_opening = opening_data['Volume'].mean()
        
        # Calculate trade value (price * volume)
        opening_data_copy = opening_data.copy()
        opening_data_copy['TradeValue'] = opening_data_copy['Close'] * opening_data_copy['Volume']
        total_trade_value = opening_data_copy['TradeValue'].sum()
        trade_value_crores = total_trade_value / 10000000  # Convert to crores
        
        # Create 15-min candle (9:15 to 9:29)
        opening_candle = {
            'Open': opening_data['Open'].iloc[0],
            'High': opening_data['High'].max(),
            'Low': opening_data['Low'].min(),
            'Close': opening_data['Close'].iloc[-1],
            'Volume': opening_data['Volume'].sum()
        }
        
        # Check if body is big (Modified to 0.3% from 0.5%)
        body_size = abs(opening_candle['Close'] - opening_candle['Open'])
        body_pct = (body_size / opening_candle['Open']) * 100
        is_big_body = body_pct >= 0.3 # RELAXED CONDITION
        
        return avg_volume_opening, trade_value_crores, is_big_body, opening_candle
    
    def check_gap(self, df):
        """
        Check if stock has gap up or gap down
        Compare today's open with previous day's close
        """
        if len(df) < 2:
            return False
        
        # Get previous day's last candle
        today = df.index[-1].date()
        yesterday_data = df[df.index.date < today]
        
        if len(yesterday_data) == 0:
            return True  # No previous day data, assume no gap
        
        prev_close = yesterday_data['Close'].iloc[-1]
        today_open = df[df.index.date == today]['Open'].iloc[0]
        
        gap_pct = abs((today_open - prev_close) / prev_close) * 100
        
        # RELAXED CONDITION: Allow gaps up to 1.5% (was 0.5%)
        return gap_pct < 1.5
    
    def screen_stock(self, df):
        """
        Screen stock based on Stage A criteria
        Returns: (passed, reason, metrics)
        """
        # Need enough data for prev 60 candles + current opening 15 candles
        if len(df) < 75:
            return False, "Insufficient data", None
        
        # Check for gap
        no_gap = self.check_gap(df)
        if not no_gap:
            return False, "Gap > 1.5%", None
        
        # Get opening candle metrics (Current Day 9:15-9:29)
        avg_vol_opening, trade_value, big_body, opening_candle = self.calculate_opening_candle_metrics(df)
        
        if avg_vol_opening is None:
            return False, "No opening period data", None
        
        # Calculate average volume of the PREVIOUS 60 candles (before today's open)
        today_date = df.index[-1].date()
        prev_data = df[df.index.date < today_date]
        
        if len(prev_data) < 60:
            return False, "Insufficient previous day data", None
            
        avg_vol_prev_60 = prev_data['Volume'].tail(60).mean()
        
        # RELAXED CONDITION: Opening Volume > 2x Previous 60 Avg (was 5x)
        volume_check = avg_vol_opening > (2 * avg_vol_prev_60)
        
        # RELAXED CONDITION: Trade Value > 10 Cr (was 20 Cr)
        trade_value_check = trade_value > 10
        
        metrics = {
            'avg_vol_opening': avg_vol_opening,
            'avg_vol_prev_60': avg_vol_prev_60,
            'trade_value': trade_value,
            'body_pct': (abs(opening_candle['Close'] - opening_candle['Open']) / opening_candle['Open']) * 100
        }
        
        reasons = []
        if not volume_check:
            reasons.append(f"Volume: {avg_vol_opening:.0f} vs {2*avg_vol_prev_60:.0f} (Req > 2x Prev 60)")
        if not trade_value_check:
            reasons.append(f"Trade Value: ₹{trade_value:.2f}Cr vs ₹10Cr")
        if not big_body:
            reasons.append("Body < 0.3%")
        
        passed = volume_check and trade_value_check and big_body
        
        reason = "Passed all criteria" if passed else ", ".join(reasons)
        
        return passed, reason, metrics
    
    def check_breakout_entry(self, df, current_candle, day_high, day_low):
        """
        Check for breakout entry signals (Stage B)
        Returns: signal_type ('BUY', 'SELL', or None)
        """
        current_volume = current_candle['Volume']
        current_high = current_candle['High']
        current_low = current_candle['Low']
        
        # --- NEW ENTRY LOGIC: SMA VOLUME + RSI ---
        if len(df) < 20:
            return None
            
        recent_avg_volume = df['Volume'].tail(20).mean()
        
        # Volume must be 1.5x the recent average (SMA 20)
        volume_spike = current_volume > (recent_avg_volume * 1.5)
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['Close']).iloc[-1]

        if not volume_spike:
            return None
        
        # Buy signal: Breakout + RSI Momentum
        if current_high > day_high:
            # RSI Confirmation: RSI should be rising (e.g., > 50) but not extremely overbought yet (> 80)
            if 50 < rsi < 80: 
                return 'BUY'
        
        # Sell signal: Breakdown + RSI Momentum
        if current_low < day_low:
            # RSI Confirmation: RSI should be falling (e.g., < 50) but not extremely oversold yet (< 20)
            if 20 < rsi < 50:
                return 'SELL'
        
        return None
    
    def update_trailing_stop(self, current_price, current_day_high, current_day_low):
        """
        Update trailing stop loss based on day high/low
        """
        if self.current_position == 'long':
            # Trailing stop: 1% down from day high
            new_trailing = current_day_high * (1 - self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing
        
        elif self.current_position == 'short':
            # Trailing stop: 1% up from day low
            new_trailing = current_day_low * (1 + self.trailing_stop_pct)
            if self.trailing_stop is None or new_trailing < self.trailing_stop:
                self.trailing_stop = new_trailing
    
    def check_exit_conditions(self, current_price, current_day_high, current_day_low):
        """
        Check exit conditions (Stage C)
        Returns: (should_exit, reason)
        """
        if not self.current_position:
            return False, None
        
        # Update trailing stop
        self.update_trailing_stop(current_price, current_day_high, current_day_low)
        
        if self.current_position == 'long':
            # Check target
            if current_price >= self.target:
                return True, "Target hit"
            
            # Check stop loss
            if current_price <= self.stop_loss:
                return True, "Stop loss hit"
            
            # Check trailing stop
            if self.trailing_stop and current_price <= self.trailing_stop:
                return True, "Trailing stop hit"
        
        elif self.current_position == 'short':
            # Check target
            if current_price <= self.target:
                return True, "Target hit"
            
            # Check stop loss
            if current_price >= self.stop_loss:
                return True, "Stop loss hit"
            
            # Check trailing stop
            if self.trailing_stop and current_price >= self.trailing_stop:
                return True, "Trailing stop hit"
        
        return False, None
    
    def enter_position(self, signal_type, entry_price, day_high, day_low, quantity, symbol):
        """
        Enter a position
        """
        self.current_position = 'long' if signal_type == 'BUY' else 'short'
        self.entry_price = entry_price
        self.position_size = quantity
        self.day_high = day_high
        self.day_low = day_low
        self.trailing_stop = None
        self.position_symbol = symbol
        
        if self.current_position == 'long':
            # Stop loss: max of 1% or day low
            stop_loss_pct = entry_price * (1 - self.stop_loss_pct)
            self.stop_loss = max(stop_loss_pct, day_low)
            
            # Target: 3%
            self.target = entry_price * (1 + self.target_pct)
        
        else:  # short
            # Stop loss: min of 1% or day high
            stop_loss_pct = entry_price * (1 + self.stop_loss_pct)
            self.stop_loss = min(stop_loss_pct, day_high)
            
            # Target: 3%
            self.target = entry_price * (1 - self.target_pct)
    
    def exit_position(self):
        """
        Exit current position
        """
        self.current_position = None
        self.entry_price = None
        self.position_size = 0
        self.stop_loss = None
        self.target = None
        self.trailing_stop = None
        self.position_symbol = None

def get_fno_stocks_list():
    """
    Get list of FnO stocks from CSV
    """
    try:
        # Load from CSV
        equity_df = load_nse_equity_data()
        if equity_df is not None:
            # Try to identify symbol column
            possible_cols = ['Symbol', 'Tradingsymbol', 'Trading Symbol', 'symbol', 'tradingsymbol']
            symbol_col = None
            
            for col in possible_cols:
                if col in equity_df.columns:
                    symbol_col = col
                    break
            
            if symbol_col:
                # Get unique symbols, limit to first 50 for performance
                fno_stocks = equity_df[symbol_col].dropna().unique().tolist()[:50]
                return fno_stocks
        
        # Fallback to sample FnO stocks
        fno_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'HINDUNILVR', 'LT',
            'AXISBANK', 'KOTAKBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI'
        ]
        return fno_stocks
    except Exception as e:
        logging.error(f"Error loading FnO stocks: {e}")
        # Return sample list as fallback
        return ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

def load_nse_equity_data():
    """Load NSE Equity CSV file with ISIN data"""
    try:
        equity_df = pd.read_csv('NSE_Equity.csv')
        return equity_df
    except Exception as e:
        logging.error(f"Error loading NSE_Equity.csv: {e}")
        return None

def get_token_from_isin(api, symbol, exchange="NSE"):
    """Get token for the symbol using ISIN from CSV"""
    try:
        # First try to load from CSV
        equity_df = load_nse_equity_data()
        
        if equity_df is not None:
            # Search for symbol in CSV
            possible_cols = ['Symbol', 'Tradingsymbol', 'Trading Symbol', 'symbol', 'tradingsymbol']
            symbol_col = None
            
            for col in possible_cols:
                if col in equity_df.columns:
                    symbol_col = col
                    break
            
            if symbol_col:
                # Find the row with matching symbol
                mask = equity_df[symbol_col].str.upper() == symbol.upper()
                if mask.any():
                    row = equity_df[mask].iloc[0]
                    
                    # Check for ISIN column
                    isin_cols = ['ISIN', 'Isin', 'isin']
                    isin = None
                    for col in isin_cols:
                        if col in equity_df.columns:
                            isin = row[col]
                            break
                    
                    # Check for Token column (if available)
                    token_cols = ['Token', 'token', 'ScripCode', 'token_number']
                    for col in token_cols:
                        if col in equity_df.columns and pd.notna(row[col]):
                            return str(row[col])
                    
                    # If ISIN is available, search using it
                    if isin and pd.notna(isin):
                        result = api.searchscrip(exchange=exchange, searchtext=isin)
                        if result and 'values' in result and len(result['values']) > 0:
                            return result['values'][0].get('token')
        
        # Fallback: Search directly using symbol
        result = api.searchscrip(exchange=exchange, searchtext=symbol)
        if result and 'values' in result and len(result['values']) > 0:
            for item in result['values']:
                if item.get('tsym') == f"{symbol}-EQ":
                    return item.get('token')
            return result['values'][0].get('token')
        
        return None
    except Exception as e:
        logging.error(f"Error getting token for {symbol}: {e}")
        return None

def load_market_data(api, symbol, exchange="NSE"):
    """Load real market data from Flattrade API"""
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if not token:
            logging.error(f"Could not find token for {symbol}")
            return None
        
        # Fetch last 5 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        start_time = start_date.strftime("%d-%m-%Y") + " 09:15:00"
        end_time = end_date.strftime("%d-%m-%Y") + " " + end_date.strftime("%H:%M:%S")
        
        hist_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_time,
            endtime=end_time,
            interval='1'  # 1-minute interval
        )
        
        if not hist_data:
            return None
        
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
            return None
        
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data for {symbol}: {e}")
        return None

def get_live_price(api, symbol, exchange="NSE"):
    """Get current live price"""
    try:
        token = get_token_from_isin(api, symbol, exchange)
        if not token:
            return None
        
        live_data = api.get_quotes(exchange=exchange, token=token)
        if live_data and live_data.get('stat') == 'Ok':
            return float(live_data.get('lp', 0))
        
        return None
    except Exception as e:
        logging.error(f"Error getting live price for {symbol}: {e}")
        return None

def execute_trade(api, signal_type, quantity, symbol, product_type='C'):
    """Execute trade through Flattrade API"""
    try:
        buy_or_sell = 'B' if signal_type == 'BUY' else 'S'
        
        result = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type=product_type,
            exchange='NSE',
            tradingsymbol=f"{symbol}-EQ",
            quantity=quantity,
            discloseqty=0,
            price_type='MKT',
            retention='DAY'
        )
        
        return result
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None
    
def check_and_sync_positions(api, strategy):
    try:
        positions = api.get_positions()
        if positions and len(positions) > 0:
            for pos in positions:
                netqty = int(pos.get('netqty', 0))
                if netqty != 0:
                    symbol = pos.get('tsym', '').replace('-EQ', '')
                    avg_price = float(pos.get('netavgprc', 0))
                    product_type = pos.get('prd', 'C') 
                    
                    position_type = 'long' if netqty > 0 else 'short'
                    
                    if strategy.position_symbol != symbol:
                        st.warning(f"⚠️ Detected existing {position_type.upper()} position in {symbol} ({product_type})")                        
                        # Load data to get day high/low
                        df = load_market_data(api, symbol)
                        if df is not None and len(df) > 0:
                            today_data = df[df.index.date == datetime.now().date()]
                            if len(today_data) > 0:
                                day_high = today_data['High'].max()
                                day_low = today_data['Low'].min()
                                
                                # Set position in strategy
                                strategy.position_symbol = symbol
                                strategy.product_type = product_type
                                strategy.current_position = position_type
                                strategy.entry_price = avg_price
                                strategy.position_size = abs(netqty)
                                        
                                # Set stop loss and target
                                if position_type == 'long':
                                    stop_loss_pct = avg_price * (1 - strategy.stop_loss_pct)
                                    strategy.stop_loss = max(stop_loss_pct, day_low)
                                    strategy.target = avg_price * (1 + strategy.target_pct)
                                else:
                                    stop_loss_pct = avg_price * (1 + strategy.stop_loss_pct)
                                    strategy.stop_loss = min(stop_loss_pct, day_high)
                                    strategy.target = avg_price * (1 - strategy.target_pct)
                                
                                st.success(f"✅ Position synced: {symbol} - {position_type.upper()} - Type: {product_type}")
                    
                    return True, {
                        'symbol': symbol,
                        'type': position_type,
                        'qty': abs(netqty),
                        'avg_price': avg_price
                    }
        return False, None
    except Exception as e:
        logging.error(f"Error checking positions: {e}")
        return False, None

def get_top_losers(api, limit=5):
    """
    Get Top 5 Losers from the FNO list
    """
    try:
        # Reuse the existing list logic
        stock_list = get_fno_stocks_list()[:30] 
        
        losers_data = []
        
        for symbol in stock_list:
            token = get_token_from_isin(api, symbol)
            if token:
                quote = api.get_quotes(exchange='NSE', token=token)
                if quote and quote.get('stat') == 'Ok':
                    lp = float(quote.get('lp', 0))
                    prev_close = float(quote.get('c', 0))
                    
                    if prev_close > 0:
                        p_change = ((lp - prev_close) / prev_close) * 100
                        
                        # Only add negative changes (Losers)
                        if p_change < 0:
                            losers_data.append({
                                'Symbol': symbol,
                                'LTP': f"₹{lp:.2f}",
                                'Change': p_change
                            })
        
        # Sort by Change ascending
        losers_data.sort(key=lambda x: x['Change'])
        
        # Return top N and format Change for display
        top_losers = losers_data[:limit]
        for item in top_losers:
            item['Change'] = f"{item['Change']:.2f}%"
            
        return top_losers
        
    except Exception as e:
        logging.error(f"Error fetching top losers: {e}")
        return []

def get_top_gainers(api, limit=10):
    """
    Get Top 5 Gainers from the FNO list
    """
    try:
        stock_list = get_fno_stocks_list()[:30] 
        gainers_data = []
        
        for symbol in stock_list:
            token = get_token_from_isin(api, symbol)
            if token:
                quote = api.get_quotes(exchange='NSE', token=token)
                if quote and quote.get('stat') == 'Ok':
                    lp = float(quote.get('lp', 0)) # Last Price
                    prev_close = float(quote.get('c', 0)) # Previous Close
                    
                    if prev_close > 0:
                        p_change = ((lp - prev_close) / prev_close) * 100
                        
                        # Only add positive gainers
                        if p_change > 0:
                            gainers_data.append({
                                'Symbol': symbol,
                                'LTP': f"₹{lp:.2f}",
                                'Change': p_change
                            })
        
        # Sort by Change % descending
        gainers_data.sort(key=lambda x: x['Change'], reverse=True)
        
        # Return top N and format Change for display
        top_gainers = gainers_data[:limit]
        for item in top_gainers:
            item['Change'] = f"+{item['Change']:.2f}%"
            
        return top_gainers
        
    except Exception as e:
        logging.error(f"Error fetching top gainers: {e}")
        return []
    
def create_chart(df, entry_point=None, stop_loss=None, target=None, trailing_stop=None, signals_df=None):
    """Create candlestick chart with levels and signals"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Price & Trading Levels', 'Volume'),
                        row_heights=[0.8, 0.2])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'), row=1, col=1)
    
    # Volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        name='Volume',
                        marker_color=colors), row=2, col=1)
    
    # Add entry/exit levels if position exists
    if entry_point:
        fig.add_hline(y=entry_point, line_dash="dash", line_color="blue",
                     annotation_text="Entry", row=1, col=1)
    if stop_loss:
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red",
                     annotation_text="Stop Loss", row=1, col=1)
    if target:
        fig.add_hline(y=target, line_dash="dash", line_color="green",
                     annotation_text="Target", row=1, col=1)
    if trailing_stop:
        fig.add_hline(y=trailing_stop, line_dash="dot", line_color="orange",
                     annotation_text="Trailing Stop", row=1, col=1)
    
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
    
    fig.update_layout(title='FnO Breakout Trading Strategy - Live Data',
                     xaxis_rangeslider_visible=False,
                     height=800)
    
    return fig

def main():
    st.set_page_config(page_title="FnO Breakout Trading App (Enhanced)", layout="wide")
    
    st.title("⚡ FnO Breakout Trading Strategy App (Enhanced Logic)")
    
    # Initialize connections
    try:
        supabase = init_supabase()
        api = init_flattrade_api()
        
        # Initialize strategy in session state
        if 'strategy' not in st.session_state:
            st.session_state.strategy = FnOBreakoutStrategy()
        
        strategy = st.session_state.strategy
        
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
        st.title("🚀 Market Movers")
        
        # Create tabs for cleaner UI
        tab1, tab2 = st.tabs(["📈 Gainers", "📉 Losers"])
        
        # Button to refresh both
        if st.button("🔄 Refresh Market Data"):
            st.session_state.gainers = get_top_gainers(api)
            st.session_state.losers = get_top_losers(api)
            
        # Initialize session state if empty
        if 'gainers' not in st.session_state:
            st.session_state.gainers = get_top_gainers(api)
        if 'losers' not in st.session_state:
            st.session_state.losers = get_top_losers(api)
            
        # --- Gainers Tab ---
        with tab1:
            if st.session_state.gainers:
                st.dataframe(
                    pd.DataFrame(st.session_state.gainers), 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Symbol": "Stock",
                        "LTP": "Price",
                        "Change": "Change %"
                    }
                )
            else:
                st.info("No gainers found")

        # --- Losers Tab ---
        with tab2:
            if st.session_state.losers:
                st.dataframe(
                    pd.DataFrame(st.session_state.losers), 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Symbol": "Stock",
                        "LTP": "Price",
                        "Change": "Change %"
                    }
                )
            else:
                st.info("No losers found")
            
        st.markdown("---")
        
        st.header("Trading Controls")
        
        # Trading parameters
        st.subheader("Strategy Parameters")
        stop_loss_pct = st.slider("Stop Loss (%)", 0.5, 2.0, 1.0, 0.1)
        target_pct = st.slider("Target (%)", 1.0, 5.0, 3.0, 0.5)
        trailing_stop_pct = st.slider("Trailing Stop (%)", 0.5, 2.0, 1.0, 0.1)
        
        strategy.stop_loss_pct = stop_loss_pct / 100
        strategy.target_pct = target_pct / 100
        strategy.trailing_stop_pct = trailing_stop_pct / 100
        
        # Position sizing
        st.subheader("Position Sizing")
        sizing_mode = st.radio("Quantity Mode", ["Fixed Quantity", "Fixed Amount"])
        
        if sizing_mode == "Fixed Quantity":
            quantity = st.number_input("Quantity", min_value=1, value=1)
            trade_amount = None
        else:
            trade_amount = st.number_input("Trade Amount (₹)", min_value=1000, value=10000, step=1000)
            quantity = None
        
        # Scanning options
        st.subheader("📊 Scanning Options")
        
        # Number of stocks to scan
        num_stocks_to_scan = st.number_input("Number of Stocks to Scan", min_value=5, max_value=250, value=20, step=5)
        
        # Continuous scanning toggle
        continuous_scan = st.toggle("Continuous Scanning", False)
        
        if continuous_scan:
            scan_interval = st.slider("Scan Interval (seconds)", 30, 300, 60, 30)
        
        # Manual scan button
        if st.button("🔍 Run Enhanced Screening", type="primary"):
            st.session_state.run_screening = True
        
        # Auto-trading toggle
        auto_trading = st.toggle("Enable Auto Trading", False)
        
        # Manual trade buttons section
        st.subheader("Manual Trading")
        
        # Show dropdown only if stocks are screened
        if len(strategy.screened_stocks) > 0:
            manual_trade_symbol = st.selectbox("Select Stock", strategy.screened_stocks, key="manual_trade_stock")
        else:
            manual_trade_symbol = None
            st.info("Run screening first")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Manual Sell", type="secondary", disabled=(strategy.current_position is not None or manual_trade_symbol is None)):
                if auto_trading and manual_trade_symbol:
                    # Calculate quantity if using amount mode
                    if sizing_mode == "Fixed Amount":
                        live_price = get_live_price(api, manual_trade_symbol)
                        if live_price:
                            calc_quantity = int(trade_amount / live_price)
                        else:
                            st.error("Could not get live price")
                            calc_quantity = None
                    else:
                        calc_quantity = quantity
                    
                    if calc_quantity:
                        result = execute_trade(api, 'SELL', calc_quantity, manual_trade_symbol)
                        if result and result.get('stat') == 'Ok':
                            live_price = get_live_price(api, manual_trade_symbol)
                            df = load_market_data(api, manual_trade_symbol)
                            if df is not None and len(df) > 0:
                                today_data = df[df.index.date == datetime.now().date()]
                                day_high = today_data['High'].max()
                                day_low = today_data['Low'].min()
                                current_price = live_price if live_price else df['Close'].iloc[-1]
                                strategy.enter_position('SELL', current_price, day_high, day_low, calc_quantity, manual_trade_symbol)
                            st.success(f"Sell order placed: {result.get('norenordno')}")
                        else:
                            st.error("Failed to place sell order")
                else:
                    st.warning("Enable auto-trading first")
        
        with col2:
            if st.button("🟢 Manual Buy", type="primary", disabled=(strategy.current_position is not None or manual_trade_symbol is None)):
                if auto_trading and manual_trade_symbol:
                    # Calculate quantity if using amount mode
                    if sizing_mode == "Fixed Amount":
                        live_price = get_live_price(api, manual_trade_symbol)
                        if live_price:
                            calc_quantity = int(trade_amount / live_price)
                        else:
                            st.error("Could not get live price")
                            calc_quantity = None
                    else:
                        calc_quantity = quantity
                    
                    if calc_quantity:
                        result = execute_trade(api, 'BUY', calc_quantity, manual_trade_symbol)
                        if result and result.get('stat') == 'Ok':
                            live_price = get_live_price(api, manual_trade_symbol)
                            df = load_market_data(api, manual_trade_symbol)
                            if df is not None and len(df) > 0:
                                today_data = df[df.index.date == datetime.now().date()]
                                day_high = today_data['High'].max()
                                day_low = today_data['Low'].min()
                                current_price = live_price if live_price else df['Close'].iloc[-1]
                                strategy.enter_position('BUY', current_price, day_high, day_low, calc_quantity, manual_trade_symbol)
                            st.success(f"Buy order placed: {result.get('norenordno')}")
                        else:
                            st.error("Failed to place buy order")
                else:
                    st.warning("Enable auto-trading first")
        
        # Close position button
        if strategy.current_position:
            if st.button("❌ Close Position", type="secondary"):
                if auto_trading:
                    exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                    # Use the tracked position symbol
                    close_symbol = strategy.position_symbol
                    if close_symbol:
                        result = execute_trade(api, exit_signal, strategy.position_size, close_symbol, product_type=strategy.product_type)
                        if result and result.get('stat') == 'Ok':
                            strategy.exit_position()
                            st.success("✅ Position closed!")
                            st.rerun()
                        else:
                            st.error("Failed to close position")
                else:
                    st.warning("Enable auto-trading first")
    
    # Main content area
    current_date = datetime.now().date()
    current_time = datetime.now().time()
    strategy.reset_daily_data(current_date)
    
    # Check for existing positions from API and sync
    has_existing_position, position_details = check_and_sync_positions(api, strategy)
    
    if has_existing_position:
        st.info(f"📍 Active Position Detected: {position_details['symbol']} - {position_details['type'].upper()} - Qty: {position_details['qty']}")
    
    # Check trading hours
    if current_time < time(1, 15) or current_time > time(23, 30):
        st.warning("⏰ Outside Trading Hours (9:15 AM - 3:30 PM)")
    
    # Screening Section - Only run if no open position
    if 'run_screening' in st.session_state and st.session_state.run_screening:
        if strategy.current_position:
            st.warning("⚠️ Screening disabled - Close existing position first")
            st.session_state.run_screening = False
        else:
            st.header("📋 Stage A: Stock Screening Results (Enhanced)")
        
        with st.spinner("Screening FnO stocks (Relaxed Criteria)..."):
            stocks_to_screen = get_fno_stocks_list()[:num_stocks_to_scan]
            
            screening_results = []
            
            progress_bar = st.progress(0)
            for idx, stock in enumerate(stocks_to_screen):
                df = load_market_data(api, stock)
                
                if df is not None and len(df) >= 300:
                    passed, reason, metrics = strategy.screen_stock(df)
                    
                    result_data = {
                        'Symbol': stock,
                        'Status': '✅ Passed' if passed else '❌ Failed',
                        'Reason': reason,
                        'Current Price': f"₹{df['Close'].iloc[-1]:.2f}" if len(df) > 0 else 'N/A'
                    }
                    
                    if metrics:
                        result_data['Avg Vol (9:15-9:29)'] = f"{metrics['avg_vol_opening']:.0f}"
                        result_data['Trade Value (Cr)'] = f"₹{metrics['trade_value']:.2f}"
                    
                    screening_results.append(result_data)
                    
                    if passed:
                        strategy.screened_stocks.append(stock)
                
                progress_bar.progress((idx + 1) / len(stocks_to_screen))
            
            # Display results
            if screening_results:
                results_df = pd.DataFrame(screening_results)
                st.dataframe(results_df, use_container_width=True)
            
            if len(strategy.screened_stocks) > 0:
                st.success(f"✅ {len(strategy.screened_stocks)} stocks passed screening: {', '.join(strategy.screened_stocks)}")
            else:
                st.info("No stocks passed the screening criteria")
        
            st.session_state.run_screening = False
    
    # Continuous scanning - Only if no open position
    if continuous_scan and len(strategy.screened_stocks) == 0 and not strategy.current_position:
        st.info(f"🔄 Continuous scanning active - Running every {scan_interval} seconds")
        time_module.sleep(scan_interval)
        st.session_state.run_screening = True
        st.rerun()
    
    # Trading Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Live Market Data & Signals")
        
        # If position exists, monitor that stock
        if strategy.current_position and strategy.position_symbol:
            st.info(f"🔒 Monitoring active position: {strategy.position_symbol}")
            selected_stock = strategy.position_symbol
            
            # Also show screened stocks in sidebar info
            if len(strategy.screened_stocks) > 0:
                st.success(f"✅ {len(strategy.screened_stocks)} other stocks screened and ready: {', '.join(strategy.screened_stocks)}")
        # Otherwise, select from screened stocks
        elif len(strategy.screened_stocks) > 0:
            selected_stock = st.selectbox("Select Screened Stock to Monitor", strategy.screened_stocks, key="monitor_stock")
            st.session_state.selected_stock = selected_stock
        else:
            st.info("👆 Run screening to find trading opportunities")
            selected_stock = None
        
        if selected_stock:
            # Load and display market data
            with st.spinner(f"Loading live market data for {selected_stock}..."):
                df = load_market_data(api, selected_stock)
            
            if df is not None and len(df) > 0:
                # Calculate day high/low
                today_data = df[df.index.date == current_date]
                
                if len(today_data) > 0:
                    day_high = today_data['High'].max()
                    day_low = today_data['Low'].min()
                    
                    # Get opening period metrics
                    avg_vol_opening, trade_value, big_body, opening_candle = strategy.calculate_opening_candle_metrics(df)
                    
                    if avg_vol_opening:
                        strategy.avg_vol_opening = avg_vol_opening
                    
                    # Get current candle
                    current_candle = {
                        'Open': df['Open'].iloc[-1],
                        'High': df['High'].iloc[-1],
                        'Low': df['Low'].iloc[-1],
                        'Close': df['Close'].iloc[-1],
                        'Volume': df['Volume'].iloc[-1]
                    }
                    
                    # Get live price
                    live_price = get_live_price(api, selected_stock)
                    current_price = live_price if live_price else current_candle['Close']
                    
                    # Display current signals
                    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
                    
                    with signal_col1:
                        st.metric("Live Price", f"₹{current_price:.2f}")
                    with signal_col2:
                        st.metric("Day High", f"₹{day_high:.2f}")
                    with signal_col3:
                        st.metric("Day Low", f"₹{day_low:.2f}")
                    with signal_col4:
                        st.metric("Volume", f"{current_candle['Volume']:,}")
                    
                    # Check for entry signals if no position
                    if not strategy.current_position and time(1, 30) <= current_time <= time(23, 20):
                        # UPDATED: Use the new check_breakout_entry with dataframe for SMA calculation
                        signal = strategy.check_breakout_entry(
                            df, current_candle, day_high, day_low
                        )
                        
                        if signal:
                            if signal == 'BUY':
                                st.success("🟢 BUY SIGNAL ACTIVE - Breakout + Vol + RSI!")
                            else:
                                st.error("🔴 SELL SIGNAL ACTIVE - Breakdown + Vol + RSI!")
                            
                            if auto_trading:
                                # Calculate quantity based on mode
                                if sizing_mode == "Fixed Amount":
                                    calc_quantity = int(trade_amount / current_price)
                                else:
                                    calc_quantity = quantity
                                
                                result = execute_trade(api, signal, calc_quantity, selected_stock)
                                if result and result.get('stat') == 'Ok':
                                    strategy.enter_position(signal, current_price, day_high, day_low, calc_quantity, selected_stock)
                                    st.success(f"✅ Auto {signal} Order Executed! Qty: {calc_quantity}")
                                    st.rerun()
                        else:
                            st.info("⚪ No Entry Signal - Waiting for breakout & volume spike (1.5x SMA)")
                    elif strategy.current_position:
                        st.info(f"🔒 Position already open in {strategy.position_symbol} - Monitoring for exit")
                    
                    # Check for exit signals if position exists
                    if strategy.current_position:
                        should_exit, reason = strategy.check_exit_conditions(
                            current_price, day_high, day_low
                        )
                        
                        if should_exit:
                            st.warning(f"⚠️ Exit Signal: {reason}")
                            
                            if auto_trading:
                                exit_signal = 'SELL' if strategy.current_position == 'long' else 'BUY'
                                
                                result = execute_trade(
                                    api, 
                                    exit_signal, 
                                    strategy.position_size, 
                                    selected_stock, 
                                    product_type=strategy.product_type 
                                )
                                
                                if result and result.get('stat') == 'Ok':
                                    strategy.exit_position()
                                    st.success(f"✅ Position Closed: {reason}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to close position. API Response: {result}")
                                        
                    # Create and display chart
                    chart = create_chart(
                        df.tail(100),
                        strategy.entry_price,
                        strategy.stop_loss,
                        strategy.target,
                        strategy.trailing_stop
                    )
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Display latest data
                    st.subheader("📊 Latest OHLC Data")
                    st.dataframe(df.tail(10), use_container_width=True)
                
                else:
                    st.warning("No data available for today")
            
            else:
                st.error(f"Failed to load market data for {selected_stock}")
    
    with col2:
        st.subheader("📊 Position & Status")
        
        # Current position status
        if strategy.current_position:
            position_type = strategy.current_position.upper()
            st.info(f"**Position:** {position_type}")
            st.info(f"**Symbol:** {strategy.position_symbol}")
            st.info(f"**Size:** {strategy.position_size}")
            st.info(f"**Entry:** ₹{strategy.entry_price:.2f}")
            
            # Display stop loss and target
            if strategy.stop_loss:
                st.info(f"**Stop Loss:** ₹{strategy.stop_loss:.2f}")
            if strategy.target:
                st.info(f"**Target:** ₹{strategy.target:.2f}")
            if strategy.trailing_stop:
                st.info(f"**Trailing Stop:** ₹{strategy.trailing_stop:.2f}")
            
            # Calculate current P&L for the position symbol
            if strategy.position_symbol:
                live_price = get_live_price(api, strategy.position_symbol)
                position_df = load_market_data(api, strategy.position_symbol)
                
                if position_df is not None and len(position_df) > 0:
                    current_price = live_price if live_price else position_df['Close'].iloc[-1]
                    
                    if strategy.current_position == 'long':
                        pnl = (current_price - strategy.entry_price) * strategy.position_size
                        pnl_pct = ((current_price - strategy.entry_price) / strategy.entry_price) * 100
                    else:
                        pnl = (strategy.entry_price - current_price) * strategy.position_size
                        pnl_pct = ((strategy.entry_price - current_price) / strategy.entry_price) * 100
                    
                    if pnl >= 0:
                        st.success(f"**Current P&L:** +₹{pnl:.2f} ({pnl_pct:.2f}%)")
                    else:
                        st.error(f"**Current P&L:** ₹{pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            st.info("No Open Position")
        
        # Account status
        st.subheader("💰 Account Status")
        try:
            account_details = api.get_limits()
            if account_details and account_details.get('stat') == 'Ok':
                cash = account_details.get('cash', 'N/A')
                margin_used = account_details.get('marginused', 'N/A')
                
                st.info(f"**Cash Available:** ₹{cash}")
                st.info(f"**Margin Used:** ₹{margin_used}")
            else:
                st.warning("Could not fetch account details")
        except Exception as e:
            st.warning(f"Could not fetch account details")
        
        # Open Positions from API
        st.subheader("📍 Open Positions")
        try:
            positions = api.get_positions()
            if positions and len(positions) > 0:
                positions_df = pd.DataFrame(positions)
                display_cols = ['tsym', 'netqty', 'netavgprc', 'lp', 'rpnl', 'upnl']
                available_cols = [col for col in display_cols if col in positions_df.columns]
                
                if available_cols:
                    st.dataframe(positions_df[available_cols], use_container_width=True)
                else:
                    st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No open positions from API")
        except Exception as e:
            st.info("No open positions")
        
        # Recent orders - Order Book
        st.subheader("📋 Order Book")
        try:
            orders = api.get_order_book()
            if orders and len(orders) > 0:
                orders_df = pd.DataFrame(orders)
                display_cols = ['tsym', 'trantype', 'qty', 'prc', 'avgprc', 'status', 'rejreason']
                available_cols = [col for col in display_cols if col in orders_df.columns]
                
                if available_cols:
                    st.dataframe(orders_df[available_cols].head(10), use_container_width=True)
                else:
                    st.dataframe(orders_df.head(10), use_container_width=True)
            else:
                st.info("No recent orders")
        except Exception as e:
            st.info("No recent orders")
        
        # Trade Book
        st.subheader("📒 Trade Book")
        try:
            trades = api.get_trade_book()
            if trades and len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                display_cols = ['tsym', 'trantype', 'qty', 'prc', 'flqty']
                available_cols = [col for col in display_cols if col in trades_df.columns]
                
                if available_cols:
                    st.dataframe(trades_df[available_cols].head(10), use_container_width=True)
                else:
                    st.dataframe(trades_df.head(10), use_container_width=True)
            else:
                st.info("No trades today")
        except Exception as e:
            st.info("No trades today")
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="secondary"):
            st.rerun()
    
    # Footer with strategy info
    st.markdown("---")
    with st.expander("ℹ️ Enhanced Strategy Information & Rules"):
        st.markdown(f"""
        ### 📋 Stage A - Stock Screening (Enhanced)
        1. **Volume Check:** Opening volume (9:15-9:29) > **2x** Previous Day Last Hour Avg (Reduced from 5x)
        2. **Trade Value:** Total value > **₹10 Crores** (Reduced from 20Cr)
        3. **Gap Check:** Gap Up/Down < **1.5%** (Relaxed from 0.5%)
        4. **Big Body:** Opening Candle Body ≥ **0.3%** (Relaxed from 0.5%)
        
        ### 📈 Stage B - Entry Conditions (New Logic)
        1. **Breakout:** Price breaks Day High (Buy) or Day Low (Sell).
        2. **Volume Confirmation:** Current Volume > **1.5x SMA(20)** (More realistic than Opening Vol).
        3. **RSI Filter:**
           - **BUY:** RSI between 50 and 80 (Momentum active, not overbought).
           - **SELL:** RSI between 20 and 50 (Momentum active, not oversold).
        
        ### 🎯 Stage C - Exit Strategy
        1. **Stop Loss:** Better of 1% or day low/high
        2. **Target:** {target_pct}% profit
        3. **Trailing Stop:** {trailing_stop_pct}% trailing
        
        ### ⚙️ Current Settings
        - Auto Trading: {'✅ Enabled' if auto_trading else '❌ Disabled'}
        """)

if __name__ == "__main__":
    main()
