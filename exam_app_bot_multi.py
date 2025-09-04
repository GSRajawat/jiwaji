import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import logging
import requests
import json
import time
from typing import Optional, Tuple

# Add the parent directory to the sys.path to import api_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from api_helper import NorenApiPy
except ImportError:
    st.error("⚠️ api_helper.py not found. Please ensure it's in the correct directory.")
    st.stop()

# Setting up logging for better debugging in a production environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FlatTrade API Credentials ---
USER_SESSION = st.secrets.get("FLATTRADE_USER_SESSION", "d12ba7290d1cfc2235f19d333e8925c1a587418da3c087ee977828e7277c5d2b")
USER_ID = st.secrets.get("FLATTRADE_USER_ID", "FZ03508")

class FlatTradeAPI:
    """FlatTrade API wrapper for fetching historical data"""
    
    def __init__(self, user_id: str, user_session: str):
        self.api = NorenApiPy()
        self.user_id = user_id
        self.user_session = user_session
        self.authenticated = False
        
    def authenticate(self):
        """Authenticate with FlatTrade using session token"""
        try:
            # Set session token
            ret = self.api.set_session(userid=self.user_id, password="", usertoken=self.user_session)
            if ret:
                self.authenticated = True
                logging.info("FlatTrade authentication successful")
                return True
            else:
                logging.error("FlatTrade authentication failed")
                return False
        except Exception as e:
            logging.error(f"FlatTrade authentication error: {e}")
            return False
    
    def get_historical_data(self, symbol: str, token: str, start_date: datetime.datetime, 
                           end_date: datetime.datetime) -> Optional[pd.DataFrame]:
        """
        Fetch historical 1-minute data from FlatTrade
        """
        if not self.authenticated:
            if not self.authenticate():
                return None
        
        try:
            # FlatTrade expects date in 'dd-mm-yyyy' format
            start_date_str = start_date.strftime('%d-%m-%Y')
            end_date_str = end_date.strftime('%d-%m-%Y')
            
            logging.info(f"Fetching FlatTrade data for {symbol} (Token: {token}) from {start_date_str} to {end_date_str}")
            
            # Get historical data using token
            hist_data = self.api.get_time_price_series(
                exchange='NSE',
                token=token,
                starttime=start_date_str,
                endtime=end_date_str,
                interval='1'  # 1 minute interval
            )
            
            if hist_data and 'stat' in hist_data and hist_data['stat'] == 'Ok':
                df_data = []
                for candle in hist_data.get('data', []):
                    try:
                        df_data.append({
                            'date': pd.to_datetime(candle['time'], format='%d-%m-%Y %H:%M:%S'),
                            'open': float(candle['into']),
                            'high': float(candle['inth']),
                            'low': float(candle['intl']),
                            'close': float(candle['intc']),
                            'volume': int(candle.get('intv', 0))
                        })
                    except (KeyError, ValueError) as e:
                        logging.warning(f"Error parsing candle data: {e}")
                        continue
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Filter for market hours (9:30 AM to 3:30 PM)
                    df = df.between_time('09:30', '15:30')
                    
                    logging.info(f"Successfully fetched {len(df)} candles from FlatTrade for {symbol}")
                    return df
                else:
                    logging.warning(f"No valid candle data found in FlatTrade response for {symbol}")
                    return None
            else:
                error_msg = hist_data.get('emsg', 'Unknown error') if hist_data else 'No response'
                logging.error(f"FlatTrade API error for {symbol}: {error_msg}")
                return None
                
        except Exception as e:
            logging.error(f"Error fetching FlatTrade data for {symbol}: {e}")
            return None

def load_nse_symbols():
    """Load NSE symbols from uploaded CSV file"""
    try:
        # Try to read the NSE_Equity.csv file
        if os.path.exists('NSE_Equity.csv'):
            df = pd.read_csv('NSE_Equity.csv')
            return df
        else:
            # Return None if file doesn't exist
            return None
    except Exception as e:
        logging.error(f"Error loading NSE symbols: {e}")
        return None

def get_tradetron_historical_data(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, 
                                 tradetron_cookie: str, tradetron_user_agent: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical 1-minute data from Tradetron API.
    Returns DataFrame with OHLCV data or None if failed.
    """
    if not tradetron_cookie or not tradetron_user_agent:
        logging.warning(f"Tradetron credentials not provided for {symbol}")
        return None

    clean_symbol = symbol.replace('-EQ', '').strip()
    
    # Convert datetime to milliseconds
    stime_ms = int(start_date.timestamp() * 1000)
    etime_ms = int(end_date.timestamp() * 1000)

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
        "user-agent": tradetron_user_agent,
        "cookie": tradetron_cookie
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data and data.get('success') is True and 'Data' in data and isinstance(data['Data'], list):
            df_data = []
            for candle in data['Data']:
                if isinstance(candle, dict) and all(key in candle for key in ['time', 'open', 'high', 'low', 'close']):
                    df_data.append({
                        'date': pd.to_datetime(candle['time'], unit='ms'),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': int(candle.get('volume', 0))
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter for market hours (9:30 AM to 3:30 PM)
                df = df.between_time('09:30', '15:30')
                
                logging.info(f"Successfully fetched {len(df)} candles from Tradetron for {symbol}")
                return df
            else:
                logging.warning(f"No valid candle data found in Tradetron response for {symbol}")
                return None
        else:
            logging.warning(f"Tradetron API returned no valid data for {symbol}: {data}")
            return None

    except Exception as e:
        logging.error(f"Error fetching Tradetron data for {symbol}: {e}")
        return None

def fetch_real_data(symbol: str, token: str, start_date: datetime.datetime, end_date: datetime.datetime, 
                   data_source: str = "auto", tradetron_cookie: str = "", tradetron_user_agent: str = "",
                   flattrade_api: FlatTradeAPI = None) -> Optional[pd.DataFrame]:
    """
    Fetch real market data from Tradetron or FlatTrade APIs.
    Returns None if no data can be fetched.
    """
    st.info(f"📊 Fetching real data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    data = None
    
    if data_source == "tradetron" or data_source == "auto":
        st.info("🔄 Trying Tradetron API...")
        data = get_tradetron_historical_data(symbol, start_date, end_date, tradetron_cookie, tradetron_user_agent)
        if data is not None and not data.empty:
            st.success("✅ Successfully fetched data from Tradetron!")
            return data
        else:
            st.warning("⚠️ Tradetron data fetch failed or returned empty data")
    
    if data_source == "flattrade" or data_source == "auto":
        if flattrade_api and token:
            st.info("🔄 Trying FlatTrade API...")
            data = flattrade_api.get_historical_data(symbol, token, start_date, end_date)
            if data is not None and not data.empty:
                st.success("✅ Successfully fetched data from FlatTrade!")
                return data
            else:
                st.warning("⚠️ FlatTrade data fetch failed or returned empty data")
        else:
            st.info("ℹ️ FlatTrade API not initialized or token not provided, skipping...")
    
    # No fallback - return None if all real data sources failed
    st.error("❌ All real data sources failed. Cannot proceed without valid market data.")
    return None

def calculate_sdvwap(df, period=20):
    """
    Calculate Standard Deviation Volume Weighted Average Price (SDVWAP)
    Returns SDVWAP+1 and SDVWAP-1 bands
    """
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Ensure volume exists and is valid
    if 'volume' not in df.columns or df['volume'].isna().all() or (df['volume'] == 0).all():
        st.warning("⚠️ Volume data is missing or zero. Cannot calculate SDVWAP accurately.")
        return None, None, None
    
    # Calculate VWAP
    vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    # Calculate price deviations from VWAP
    price_deviation = typical_price - vwap
    
    # Calculate standard deviation of price deviations weighted by volume
    variance = ((price_deviation ** 2) * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    std_dev = np.sqrt(variance)
    
    # SDVWAP bands
    sdvwap_upper = vwap + std_dev  # SDVWAP+1
    sdvwap_lower = vwap - std_dev  # SDVWAP-1
    
    return vwap, sdvwap_upper, sdvwap_lower

def backtest_strategy(df, params):
    """
    SIMPLIFIED TRADE LOGIC STATEMENT:
    
    LONG TRADE CONDITIONS:
    1. Time: After 9:30 AM and before 2:30 PM
    2. Entry: Buy 1 quantity when last 2 green candles close above SDVWAP+1
    3. Exit: Close position if price achieves target of 2% profit
    4. Exit: Close position if last 2 red candles close below SDVWAP-1
    5. Exit: Close position at 2:30 PM
    
    SHORT TRADE CONDITIONS:
    1. Time: After 9:30 AM and before 2:30 PM  
    2. Entry: Sell 1 quantity when last 2 red candles close below SDVWAP-1
    3. Exit: Close position if price achieves target of 2% profit
    4. Exit: Close position if last 2 green candles close above SDVWAP+1
    5. Exit: Close position at 2:30 PM
    
    Note: No position reversals - only entry and exit signals.
    """
    
    # Calculate SDVWAP indicators
    vwap, sdvwap_upper, sdvwap_lower = calculate_sdvwap(df, period=params.get('sdvwap_period', 20))
    
    if vwap is None:
        st.error("❌ Cannot calculate SDVWAP indicators. Please ensure volume data is available.")
        return pd.DataFrame(), df
    
    df['vwap'] = vwap
    df['sdvwap_upper'] = sdvwap_upper
    df['sdvwap_lower'] = sdvwap_lower
    
    trade_log = []
    
    # Initialize position variables
    position_type = None  # 'long' or 'short'
    entry_price = 0
    trade_count = 0
    trade_start_time = None
    quantity = params['base_quantity']
    
    for i in range(2, len(df)):  # Start from index 2 to check last 2 candles
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        prev_prev_candle = df.iloc[i-2]
        
        # Rule 1: Only trade between 9:30 AM and 2:30 PM
        current_time = current_candle.name.time()
        start_trade_time = datetime.time(9, 30)
        end_trade_time = datetime.time(14, 30)

        # Close all positions at 2:30 PM (Rule 5)
        if current_time >= end_trade_time and position_type is not None:
            if position_type == 'long':
                trade_pnl = (current_candle['close'] - entry_price) * quantity
            else:  # short
                trade_pnl = (entry_price - current_candle['close']) * quantity
            
            trade_log.append({
                'Trade #': trade_count,
                'Type': f'{position_type}_eod_close',
                'Entry Date': trade_start_time,
                'Exit Date': current_candle.name,
                'Entry Price': entry_price,
                'Exit Price': current_candle['close'],
                'Quantity': quantity,
                'P&L': trade_pnl,
                'Exit Reason': 'End of Day (2:30 PM)'
            })
            
            position_type = None
            trade_count += 1
            continue

        # Skip if outside trading hours
        if not (start_trade_time <= current_time < end_trade_time):
            continue

        # Check candle colors
        current_green = current_candle['close'] > current_candle['open']
        prev_green = prev_candle['close'] > prev_candle['open']
        prev_prev_green = prev_prev_candle['close'] > prev_prev_candle['open']
        
        current_red = current_candle['close'] < current_candle['open']
        prev_red = prev_candle['close'] < prev_candle['open']
        prev_prev_red = prev_prev_candle['close'] < prev_prev_candle['open']
        
        # Last 2 green candles
        last_2_green = prev_green and prev_prev_green
        # Last 2 red candles  
        last_2_red = prev_red and prev_prev_red
        
        # SDVWAP conditions for last 2 candles
        last_2_close_above_sdvwap_upper = (prev_candle['close'] > prev_candle['sdvwap_upper'] and 
                                          prev_prev_candle['close'] > prev_prev_candle['sdvwap_upper'])
        last_2_close_below_sdvwap_lower = (prev_candle['close'] < prev_candle['sdvwap_lower'] and 
                                          prev_prev_candle['close'] < prev_prev_candle['sdvwap_lower'])
        
        # MANAGE EXISTING POSITIONS
        if position_type == 'long':
            # Rule 3: Close position if price achieves target of 2%
            profit_pct = ((current_candle['close'] / entry_price) - 1) * 100
            if profit_pct >= params['profit_target_perc']:
                trade_pnl = (current_candle['close'] - entry_price) * quantity
                trade_log.append({
                    'Trade #': trade_count,
                    'Type': 'long_profit_target',
                    'Entry Date': trade_start_time,
                    'Exit Date': current_candle.name,
                    'Entry Price': entry_price,
                    'Exit Price': current_candle['close'],
                    'Quantity': quantity,
                    'P&L': trade_pnl,
                    'Exit Reason': f'Profit Target ({profit_pct:.2f}%)'
                })
                position_type = None
                trade_count += 1
                
            # Rule 4: Close position if last 2 red candles close below SDVWAP-1
            elif last_2_red and last_2_close_below_sdvwap_lower:
                trade_pnl = (current_candle['close'] - entry_price) * quantity
                trade_log.append({
                    'Trade #': trade_count,
                    'Type': 'long_exit_signal',
                    'Entry Date': trade_start_time,
                    'Exit Date': current_candle.name,
                    'Entry Price': entry_price,
                    'Exit Price': current_candle['close'],
                    'Quantity': quantity,
                    'P&L': trade_pnl,
                    'Exit Reason': '2 Red Candles Below SDVWAP-1'
                })
                position_type = None
                trade_count += 1
                
        elif position_type == 'short':
            # Rule 3: Close position if price achieves target of 2%
            profit_pct = ((entry_price / current_candle['close']) - 1) * 100
            if profit_pct >= params['profit_target_perc']:
                trade_pnl = (entry_price - current_candle['close']) * quantity
                trade_log.append({
                    'Trade #': trade_count,
                    'Type': 'short_profit_target',
                    'Entry Date': trade_start_time,
                    'Exit Date': current_candle.name,
                    'Entry Price': entry_price,
                    'Exit Price': current_candle['close'],
                    'Quantity': quantity,
                    'P&L': trade_pnl,
                    'Exit Reason': f'Profit Target ({profit_pct:.2f}%)'
                })
                position_type = None
                trade_count += 1
                
            # Rule 4: Close position if last 2 green candles close above SDVWAP+1
            elif last_2_green and last_2_close_above_sdvwap_upper:
                trade_pnl = (entry_price - current_candle['close']) * quantity
                trade_log.append({
                    'Trade #': trade_count,
                    'Type': 'short_exit_signal',
                    'Entry Date': trade_start_time,
                    'Exit Date': current_candle.name,
                    'Entry Price': entry_price,
                    'Exit Price': current_candle['close'],
                    'Quantity': quantity,
                    'P&L': trade_pnl,
                    'Exit Reason': '2 Green Candles Above SDVWAP+1'
                })
                position_type = None
                trade_count += 1
        
        # NEW POSITION ENTRY (when no current position)
        if position_type is None:
            # LONG ENTRY: Last 2 green candles close above SDVWAP+1
            if last_2_green and last_2_close_above_sdvwap_upper:
                position_type = 'long'
                entry_price = current_candle['close']
                trade_start_time = current_candle.name
                
            # SHORT ENTRY: Last 2 red candles close below SDVWAP-1
            elif last_2_red and last_2_close_below_sdvwap_lower:
                position_type = 'short'
                entry_price = current_candle['close']
                trade_start_time = current_candle.name
                
    return pd.DataFrame(trade_log), df

def display_results(df, trade_log):
    """
    Displays the backtest results using Streamlit and Plotly.
    """
    st.subheader('Backtest Results')
    
    # Calculate key metrics
    if not trade_log.empty:
        total_pnl = trade_log['P&L'].sum()
        total_trades = len(trade_log)
        winning_trades = len(trade_log[trade_log['P&L'] > 0])
        losing_trades = len(trade_log[trade_log['P&L'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        max_profit = trade_log['P&L'].max() if total_trades > 0 else 0
        max_loss = trade_log['P&L'].min() if total_trades > 0 else 0
        avg_winning_trade = trade_log[trade_log['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trade_log[trade_log['P&L'] <= 0]['P&L'].mean() if losing_trades > 0 else 0
    else:
        total_pnl = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        max_profit = 0
        max_loss = 0
        avg_winning_trade = 0
        avg_losing_trade = 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total P&L', f'₹{total_pnl:,.2f}')
    with col2:
        st.metric('Total Trades', total_trades)
    with col3:
        st.metric('Win Rate', f'{win_rate:.1f}%')
    with col4:
        st.metric('Avg P&L/Trade', f'₹{total_pnl / total_trades:.2f}' if total_trades > 0 else '₹0')
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric('Winning Trades', winning_trades)
    with col6:
        st.metric('Losing Trades', losing_trades)
    with col7:
        st.metric('Max Profit', f'₹{max_profit:.2f}')
    with col8:
        st.metric('Max Loss', f'₹{max_loss:.2f}')
    
    col9, col10 = st.columns(2)
    with col9:
        st.metric('Avg Win', f'₹{avg_winning_trade:.2f}')
    with col10:
        st.metric('Avg Loss', f'₹{avg_losing_trade:.2f}')
    
    st.markdown('---')
    
    # Plot cumulative P&L
    if not trade_log.empty:
        st.subheader('Cumulative P&L')
        trade_log['Cumulative_PnL'] = trade_log['P&L'].cumsum()
        
        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(x=trade_log['Exit Date'], 
                                    y=trade_log['Cumulative_PnL'],
                                    mode='lines+markers',
                                    name='Cumulative P&L',
                                    line=dict(color='blue', width=2)))
        
        pnl_fig.update_layout(
            title='Cumulative P&L Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative P&L (₹)',
            height=400
        )
        st.plotly_chart(pnl_fig, use_container_width=True)
    
    # Plotting the candlestick chart with SDVWAP bands and trade entries/exits
    st.subheader('Price Chart with SDVWAP Bands and Trades')
    
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'],
                                        name='Price')])

    # Add SDVWAP bands
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'],
                                 mode='lines', name='VWAP',
                                 line=dict(color='blue', width=1)))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['sdvwap_upper'],
                                 mode='lines', name='SDVWAP+1',
                                 line=dict(color='green', width=1, dash='dash')))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['sdvwap_lower'],
                                 mode='lines', name='SDVWAP-1',
                                 line=dict(color='red', width=1, dash='dash')))

    # Add trade markers
    if not trade_log.empty:
        # Entry markers
        long_entries = trade_log[trade_log['Type'].str.contains('long') & ~trade_log['Type'].str.contains('exit') & ~trade_log['Type'].str.contains('eod')]
        short_entries = trade_log[trade_log['Type'].str.contains('short') & ~trade_log['Type'].str.contains('exit') & ~trade_log['Type'].str.contains('eod')]
        
        if not long_entries.empty:
            fig.add_trace(go.Scatter(x=long_entries['Entry Date'],
                                     y=long_entries['Entry Price'],
                                     mode='markers',
                                     marker=dict(color='green', size=12, symbol='triangle-up'),
                                     name='Long Entry',
                                     text=long_entries['Type'],
                                     hovertemplate='Long Entry<br>Price: ₹%{y}<br>Date: %{x}<extra></extra>'))

        if not short_entries.empty:
            fig.add_trace(go.Scatter(x=short_entries['Entry Date'],
                                     y=short_entries['Entry Price'],
                                     mode='markers',
                                     marker=dict(color='red', size=12, symbol='triangle-down'),
                                     name='Short Entry',
                                     text=short_entries['Type'],
                                     hovertemplate='Short Entry<br>Price: ₹%{y}<br>Date: %{x}<extra></extra>'))
        
        # Exit markers
        exits = trade_log[trade_log['Type'].str.contains('exit') | trade_log['Type'].str.contains('target') | trade_log['Type'].str.contains('eod')]
        if not exits.empty:
            colors = ['orange' if 'target' in t else 'purple' if 'eod' in t else 'gray' for t in exits['Type']]
            fig.add_trace(go.Scatter(x=exits['Exit Date'],
                                     y=exits['Exit Price'],
                                     mode='markers',
                                     marker=dict(color=colors, size=10, symbol='x'),
                                     name='Exits',
                                     text=exits['Exit Reason'],
                                     hovertemplate='Exit<br>Price: ₹%{y}<br>Date: %{x}<br>Reason: %{text}<extra></extra>'))

    # Update layout
    fig.update_layout(
        title='Price Chart with SDVWAP Bands and Trade Signals',
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade Log
    st.subheader('Detailed Trade Log')
    if not trade_log.empty:
        # Add duration column
        trade_log['Duration'] = trade_log['Exit Date'] - trade_log['Entry Date']
        
        # Display with formatted columns
        display_columns = ['Trade #', 'Type', 'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 
                          'Quantity', 'P&L', 'Exit Reason', 'Duration']
        st.dataframe(trade_log[display_columns], height=400, use_container_width=True)
        
        # Exit reasons summary
        st.subheader('Exit Reasons Summary')
        exit_summary = trade_log['Exit Reason'].value_counts()
        st.bar_chart(exit_summary)
        
    else:
        st.write("No trades executed during the backtest period.")

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Trading Strategy Backtester - Real Data Only", layout="wide")
    st.title('🚀 Enhanced Trading Strategy Backtester - Real Data Only')
    st.markdown("""
    **📈 Trade Logic:**
    
    **🟢 LONG TRADES:**
    1. ⏰ Time: 9:30 AM - 2:30 PM
    2. 📈 Entry: Buy 1 quantity when last 2 green candles close above SDVWAP+1
    3. 🎯 Exit: Close at 2% profit target
    4. ❌ Exit: Close if last 2 red candles close below SDVWAP-1
    5. 🕕 Exit: Close all positions at 2:30 PM
    
    **🔴 SHORT TRADES:**
    1. ⏰ Time: 9:30 AM - 2:30 PM
    2. 📉 Entry: Sell 1 quantity when last 2 red candles close below SDVWAP-1
    3. 🎯 Exit: Close at 2% profit target  
    4. ❌ Exit: Close if last 2 green candles close above SDVWAP+1
    5. 🕕 Exit: Close all positions at 2:30 PM
    """)

    # Load NSE symbols
    nse_symbols_df = load_nse_symbols()
    
    # File upload for NSE_Equity.csv if not found
    if nse_symbols_df is None:
        st.warning("⚠️ NSE_Equity.csv file not found. Please upload it to proceed.")
        uploaded_file = st.file_uploader("Upload NSE_Equity.csv", type=['csv'])
        if uploaded_file is not None:
            nse_symbols_df = pd.read_csv(uploaded_file)
            st.success("✅ NSE symbols loaded successfully!")
        else:
            st.stop()

    # Sidebar for user inputs
    st.sidebar.header('📊 Backtest Parameters')
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        'Data Source',
        ['auto', 'tradetron', 'flattrade'],
        help="Auto tries Tradetron first, then FlatTrade as fallback"
    )
    
    # Initialize FlatTrade API
    flattrade_api = None
    flattrade_status = "❌ Not Connected"
    
    # API credentials section
    with st.sidebar.expander("🔐 API Credentials", expanded=True):
        st.markdown("**Tradetron Settings:**")
        tradetron_cookie = st.text_area(
            "Tradetron Cookie", 
            placeholder="Paste your Tradetron session cookie here...",
            help="Copy from browser developer tools after logging into Tradetron"
        )
        tradetron_user_agent = st.text_input(
            "User Agent", 
            value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            help="Browser user agent string"
        )
        
        st.markdown("**FlatTrade Settings:**")
        use_default_flattrade = st.checkbox("Use Default FlatTrade Credentials", value=True)
        
        if use_default_flattrade:
            ft_user_id = USER_ID
            ft_user_session = USER_SESSION
            st.info(f"Using default credentials for User ID: {ft_user_id[:6]}...")
        else:
            ft_user_id = st.text_input("FlatTrade User ID", placeholder="Enter your FlatTrade User ID")
            ft_user_session = st.text_input("FlatTrade Session Token", type="password", placeholder="Enter your session token")
        
        # Test FlatTrade connection
        if st.button("🔗 Test FlatTrade Connection"):
            if ft_user_id and ft_user_session:
                with st.spinner("Testing FlatTrade connection..."):
                    test_api = FlatTradeAPI(ft_user_id, ft_user_session)
                    if test_api.authenticate():
                        flattrade_status = "✅ Connected Successfully"
                        st.success(flattrade_status)
                    else:
                        flattrade_status = "❌ Authentication Failed"
                        st.error(flattrade_status)
            else:
                st.error("Please provide both User ID and Session Token")
        
        # Initialize FlatTrade API if credentials provided
        if ft_user_id and ft_user_session:
            flattrade_api = FlatTradeAPI(ft_user_id, ft_user_session)
        
        st.info(f"FlatTrade Status: {flattrade_status}")
    
    # Stock selection from NSE symbols
    st.sidebar.markdown("**Stock Selection:**")
    
    # Create a searchable dropdown for symbols
    symbol_options = nse_symbols_df['Symbol'].tolist() if nse_symbols_df is not None else []
    
    if symbol_options:
        selected_symbol = st.sidebar.selectbox(
            'Select Stock Symbol',
            options=symbol_options,
            index=0,
            help="Select a stock from NSE equity symbols"
        )
        
        # Get the corresponding token and trading symbol
        selected_row = nse_symbols_df[nse_symbols_df['Symbol'] == selected_symbol].iloc[0]
        token = str(selected_row['Token'])
        trading_symbol = selected_row['Tradingsymbol']
        
        st.sidebar.info(f"Selected: {selected_symbol} (Token: {token})")
        st.sidebar.info(f"Trading Symbol: {trading_symbol}")
    else:
        st.error("❌ No stock symbols available. Please upload NSE_Equity.csv file.")
        st.stop()
    
    # Date range with better defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.date(2024, 1, 1))
    with col2:
        end_date = st.date_input('End Date', datetime.date(2024, 1, 5))
    
    st.sidebar.markdown("**Strategy Parameters:**")
    base_quantity = st.sidebar.number_input('Quantity per Trade', value=1, min_value=1, step=1)
    profit_target_perc = st.sidebar.slider('Profit Target (%)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    sdvwap_period = st.sidebar.slider('SDVWAP Period', min_value=10, max_value=50, value=20, step=5)

    # Convert date objects to datetime
    start_dt = datetime.datetime.combine(start_date, datetime.time(9, 30))
    end_dt = datetime.datetime.combine(end_date, datetime.time(15, 30))

    # Data source info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Data Source Information:**")
    if data_source == "tradetron":
        st.sidebar.info("🔹 Tradetron: Requires valid session cookie")
    elif data_source == "flattrade":
        st.sidebar.info("🔹 FlatTrade: Professional API for Indian markets")
    else:
        st.sidebar.info("🔹 Auto: Tries Tradetron first, then FlatTrade")

    # Display selected stock information
    st.subheader(f"📊 Selected Stock: {selected_symbol}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symbol", selected_symbol)
    with col2:
        st.metric("Token", token)
    with col3:
        st.metric("Exchange", selected_row['Exchange'])

    if st.sidebar.button('🚀 Run Backtest', type='primary'):
        if start_dt >= end_dt:
            st.error("❌ End date must be after the start date.")
            return

        with st.spinner('⚡ Running backtest with real data...'):
            try:
                # 1. Fetch real data
                data = fetch_real_data(
                    symbol=trading_symbol,
                    token=token,
                    start_date=start_dt,
                    end_date=end_dt,
                    data_source=data_source,
                    tradetron_cookie=tradetron_cookie,
                    tradetron_user_agent=tradetron_user_agent,
                    flattrade_api=flattrade_api
                )
                
                if data is None or data.empty:
                    st.error("❌ No data available for the selected parameters. Cannot proceed without real market data.")
                    return
                
                # Show data info
                st.info(f"📊 Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")
                
                # Check if volume data is available
                if 'volume' not in data.columns or data['volume'].sum() == 0:
                    st.warning("⚠️ Volume data is missing or zero. SDVWAP calculations may be inaccurate.")
                
                # 2. Run the backtest
                params = {
                    'base_quantity': base_quantity,
                    'profit_target_perc': profit_target_perc,
                    'sdvwap_period': sdvwap_period
                }
                trade_log, enhanced_data = backtest_strategy(data, params)
                
                # 3. Display results
                if trade_log is not None and enhanced_data is not None:
                    display_results(enhanced_data, trade_log)
                else:
                    st.error("❌ Backtest failed. Please check the data quality and try again.")
                    return
                
                # 4. Data quality info
                with st.expander("📈 Data Quality Information"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Candles", len(data))
                    with col2:
                        st.metric("Date Range", f"{(end_date - start_date).days} days")
                    with col3:
                        st.metric("Avg Volume", f"{data['volume'].mean():.0f}" if data['volume'].sum() > 0 else "No Volume Data")
                    
                    st.subheader("Sample Data Preview")
                    st.dataframe(data.head(10))

            except Exception as e:
                st.error(f"❌ Error during backtesting: {str(e)}")
                st.exception(e)

    # Additional info and instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📋 Setup Instructions:**
    
    **For Tradetron:**
    1. Login to tradetron.tech
    2. Open browser developer tools (F12)
    3. Go to Network tab
    4. Copy 'cookie' header from any request
    
    **For FlatTrade:**
    1. Login to FlatTrade platform
    2. Get your User ID and Session Token
    3. Or use the default credentials provided
    
    **NSE_Equity.csv Format:**
    ```
    Exchange,Token,Lotsize,Symbol,Tradingsymbol,Instrument
    NSE,22,1,ACC,ACC-EQ,EQ
    NSE,25780,1,APLAPOLLO,APLAPOLLO-EQ,EQ
    ```
    
    **Notes:**
    - SDVWAP = Standard Deviation Volume Weighted Average Price
    - Real data provides accurate backtesting results
    - Volume data is essential for SDVWAP calculations
    """)
    
    # API Status Display
    st.sidebar.markdown("**🔌 API Status:**")
    if tradetron_cookie:
        st.sidebar.success("✅ Tradetron: Credentials provided")
    else:
        st.sidebar.warning("⚠️ Tradetron: No credentials")
    
    st.sidebar.info(f"FlatTrade: {flattrade_status}")
    
    # Warning about real data requirement
    st.sidebar.error("⚠️ REAL DATA ONLY: This backtester requires valid API credentials and proper stock selection.")
    
    # Display NSE symbols info
    if nse_symbols_df is not None:
        with st.sidebar.expander("📊 NSE Symbols Info"):
            st.info(f"Total symbols loaded: {len(nse_symbols_df)}")
            st.dataframe(nse_symbols_df.head(), height=200)

if __name__ == '__main__':
    main()
