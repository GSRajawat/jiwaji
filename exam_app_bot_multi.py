import pandas as pd
import time
import datetime as dt
import sys
# Assume the FLATTRADE API client is installed (e.g., pip install flattrade-api)
from flattrade_api import Flattrade

# --- API Credentials ---
# NOTE: In a real application, never hardcode passwords. Use environment variables.
USER_SESSION = "f68b270591263a92f1d4182a6a5397142b0c254bdf885738c55d854445b3ac9c"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2" # Note: Only used if session ID expires/is invalid

# --- Strategy Parameters ---
PROFIT_TARGET = 0.04 # 4%
# NOTE: Replacing custom VWAP bands with a 20-period SMA for demonstration
SMA_PERIOD = 20 
EXCHANGE = 'NSE' # Assuming all trades are in NSE Equity

# --- Global Trading State Variables (must be persistent in a real system) ---
entry_price = None
target_hit_today = False
last_exit_date = None
last_exit_direction = None # 'long' or 'short'
position_size = 0          # 0 for no position, positive for long, negative for short

# --- Product Type Mapping ---
# 'M' for MIS (Margin Intraday Square off), 'C' for CNC (Cash & Carry)
PRODUCT_MAP = {"MIS": "M", "CNC": "C"}

def initialize_flattrade() -> Flattrade:
    """Initializes and sets the session ID for the Flattrade API."""
    try:
        ft = Flattrade(USER_ID)
        ft.set_session_id(USER_SESSION)
        
        # Check connection validity by fetching limits
        limits = ft.get_limits()
        if limits.get('stat') == 'Ok':
             print(f"✅ Flattrade API initialized successfully for User: {USER_ID}")
             return ft
        else:
             print(f"❌ Failed to validate Flattrade session or fetch limits. Error: {limits.get('emsg', 'Unknown')}")
             sys.exit(1)
             
    except Exception as e:
        print(f"❌ Error initializing Flattrade API: {e}")
        sys.exit(1)

# ----------------------------------------------------------------------
## Helper Functions
# ----------------------------------------------------------------------

def get_user_inputs(default_capital=500):
    """Gets and validates user inputs for the trade."""
    print("\n--- Trade Setup ---")
    
    stock = input("Enter Stock/Instrument symbol (e.g., 'RELIANCE-EQ', 'NTPC-EQ'): ").upper().strip()
    exchange = input(f"Enter Exchange (default {EXCHANGE}): ").upper().strip() or EXCHANGE
    
    while True:
        try:
            quantity = int(input("Enter Quantity (integer > 0): "))
            if quantity > 0:
                break
            print("Quantity must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    order_type_str = input("Enter Order Type (MIS or CNC, default is MIS): ").upper().strip() or "MIS"
    order_type_api = PRODUCT_MAP.get(order_type_str, "M") # Default to 'M' for MIS
    if order_type_api == "M" and order_type_str != "MIS":
         print("Invalid Order Type. Defaulting to MIS ('M').")
         order_type_str = "MIS"
         
    while True:
        try:
            capital = float(input(f"Enter Max Capital (default {default_capital}.0): ") or default_capital)
            if capital > 0:
                break
            print("Capital must be a positive number.")
        except ValueError:
            print(f"Invalid input. Defaulting to {default_capital}.0.")
            capital = default_capital
            break

    print(f"\nTrade Parameters: Stock={stock} on {exchange}, Qty={quantity}, Type={order_type_str} ({order_type_api}), Capital Limit={capital}")
    return exchange, stock, quantity, order_type_api, capital

def fetch_and_prepare_data(ft: Flattrade, exchange: str, symbol: str) -> pd.DataFrame or None:
    """
    Fetches the last N minutes of data (e.g., 5-minute bars) and calculates SMA.
    """
    try:
        # Fetch last 50 candles of 5-minute interval for SMA calculation
        print(f"Fetching 5-minute data for {symbol}...")
        
        # Fetching data using get_time_price_series
        data_response = ft.get_time_price_series(exchange, symbol, '5', '50') 

        if data_response.get('stat') != 'Ok' or not data_response.get('values'):
            print(f"❌ Failed to fetch data: {data_response.get('emsg', 'No data returned')}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data_response['values'])
        # Rename columns to match backtest logic (Open, High, Low, Close)
        df.rename(columns={'into': 'Open', 'inth': 'High', 'intl': 'Low', 'intc': 'Close', 'v': 'Volume', 'ssm':'Date'}, inplace=True)
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])

        # Convert datetime and set index
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
        df.set_index('Date', inplace=True)
        
        # Calculate the SMA (replacing custom VWAP bands)
        df['SMA_plus'] = df['Close'].rolling(window=SMA_PERIOD).mean() * 1.002 # SMA + 0.2% Band
        df['SMA_minus'] = df['Close'].rolling(window=SMA_PERIOD).mean() * 0.998 # SMA - 0.2% Band

        # We only need the last two candles and valid SMA values
        df.dropna(inplace=True) 
        if len(df) < 2:
            print("❌ Not enough *clean* data points to run strategy.")
            return None
        
        print(f"Data prepared. Last close: {df['Close'].iloc[-1]}, Last SMA_plus: {df['SMA_plus'].iloc[-1]}")
        return df

    except Exception as e:
        print(f"❌ Error fetching or processing data: {e}")
        return None

def execute_trade(ft: Flattrade, exchange: str, symbol: str, quantity: int, product_type: str, transaction_type: str) -> bool:
    """Places a Market order using the Flattrade API."""
    
    print(f"Attempting to place {transaction_type} order ({product_type}) for {quantity} of {symbol} at Market price...")

    # Flattrade API uses 'B' for BUY, 'S' for SELL
    buy_or_sell = 'B' if transaction_type == 'BUY' else 'S'

    try:
        # Using MKT order: price=0.0, price_type='MKT'
        order_response = ft.place_order(
            buy_or_sell=buy_or_sell, 
            product_type=product_type,
            exchange=exchange, 
            tradingsymbol=symbol,  
            quantity=quantity, 
            discloseqty=0,
            price_type='MKT',
            price=0.0,
            retention='DAY', 
            amo='NO'
        )
        
        if order_response.get('stat') == 'Ok':
            print(f"✅ Order placed successfully! ID: {order_response['norenordno']}")
            return True
        else:
            print(f"❌ Order failed: {order_response.get('emsg', 'Unknown Error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error executing trade: {e}")
        return False

# ----------------------------------------------------------------------
## Strategy Logic Implementation
# ----------------------------------------------------------------------

def check_and_trade(ft: Flattrade, exchange: str, symbol: str, base_quantity: int, product_type: str, df: pd.DataFrame):
    """
    Applies the adapted strategy logic and executes trades.
    """
    global entry_price, target_hit_today, last_exit_date, last_exit_direction, position_size
    
    current_datetime = df.index[-1]
    current_time = current_datetime.time()
    current_date = current_datetime.date()
    current_close = df['Close'].iloc[-1]
    
    # Reset daily flags if new day
    if last_exit_date != current_date:
        target_hit_today = False
        last_exit_date = None
        last_exit_direction = None

    # --- Time Check ---
    if current_time < dt.time(9, 30):
        print(f"[{current_time}] Market not open yet (before 9:30 AM). Waiting...")
        return
        
    # --- Intraday Square Off (3:20 PM) ---
    if current_time >= dt.time(15, 20) and product_type == 'M': # Only for MIS
        if position_size != 0:
            print(f"[{current_time}] 🕒 Squaring off position at 3:20 PM...")
            transaction_type = 'BUY' if position_size < 0 else 'SELL'
            success = execute_trade(ft, exchange, symbol, abs(position_size), product_type, transaction_type)
            if success:
                position_size = 0
                entry_price = None
                target_hit_today = True # No re-entry after final square-off
        return
        
    # --- Exit Logic (Profit Target) ---
    if position_size != 0 and entry_price is not None:
        profit_pct = 0
        if position_size > 0: # Long
            profit_pct = (current_close - entry_price) / entry_price
        elif position_size < 0: # Short
            profit_pct = (entry_price - current_close) / entry_price
            
        if profit_pct >= PROFIT_TARGET:
            print(f"[{current_time}] 💰 Profit target ({PROFIT_TARGET*100}%) hit at {current_close}. Exiting position.")
            transaction_type = 'BUY' if position_size < 0 else 'SELL'
            success = execute_trade(ft, exchange, symbol, abs(position_size), product_type, transaction_type)
            
            if success:
                # Update state after successful exit
                last_exit_date = current_date
                last_exit_direction = 'long' if position_size > 0 else 'short'
                target_hit_today = True
                position_size = 0
                entry_price = None
                return # Stop further logic for this bar
    
    # --- Target Hit Restriction ---
    if target_hit_today:
        print(f"[{current_time}] 🚫 Target hit today. No further entries.")
        return

    # --- Strategy Entry Logic (Adapted to Open/Close vs SMA Bands) ---
    candle_minus_1_close = df['Close'].iloc[-1]
    candle_minus_1_open = df['Open'].iloc[-1]
    candle_minus_2_close = df['Close'].iloc[-2]
    candle_minus_2_open = df['Open'].iloc[-2]

    # Using SMA bands as proxy for SDVWAP1_plus/minus
    vwap_minus_val_1 = df['SMA_minus'].iloc[-1]
    vwap_minus_val_2 = df['SMA_minus'].iloc[-2]
    vwap_plus_val_1 = df['SMA_plus'].iloc[-1]
    vwap_plus_val_2 = df['SMA_plus'].iloc[-2]

    # SELL condition (short entry)
    sell_condition_1 = (candle_minus_2_close < candle_minus_2_open and candle_minus_2_close < vwap_minus_val_2)
    sell_condition_2 = (candle_minus_1_close < candle_minus_1_open and candle_minus_1_close < vwap_minus_val_1)
    
    # BUY condition (long entry)
    buy_condition_1 = (candle_minus_2_close > candle_minus_2_open and candle_minus_2_close > vwap_plus_val_2)
    buy_condition_2 = (candle_minus_1_close > candle_minus_1_open and candle_minus_1_close > vwap_plus_val_1)

    if position_size == 0:
        # No position - enter fresh
        if sell_condition_1 and sell_condition_2:
            if last_exit_direction != 'short' or last_exit_date != current_date:
                print(f"[{current_time}] ⬇️ SELL signal detected. Entering fresh SHORT position.")
                success = execute_trade(ft, exchange, symbol, base_quantity, product_type, 'SELL')
                if success:
                    position_size = -base_quantity
                    entry_price = current_close
            
        elif buy_condition_1 and buy_condition_2:
            if last_exit_direction != 'long' or last_exit_date != current_date:
                print(f"[{current_time}] ⬆️ BUY signal detected. Entering fresh LONG position.")
                success = execute_trade(ft, exchange, symbol, base_quantity, product_type, 'BUY')
                if success:
                    position_size = base_quantity
                    entry_price = current_close
    
    else:
        # Have position - check for opposite signal (Reverse and Triple)
        current_abs_size = abs(position_size)
        triple_size = current_abs_size * 3
        
        if position_size > 0 and sell_condition_1 and sell_condition_2: # Long position + Short signal
            print(f"[{current_time}] 🔄 Reverse signal (Long -> Short) detected. Exiting Long and entering {triple_size} Short.")
            # Execute a SELL order for triple_size: This closes current long and opens a new short
            success = execute_trade(ft, exchange, symbol, triple_size, product_type, 'SELL')
            if success:
                position_size = -triple_size
                entry_price = current_close
                
        elif position_size < 0 and buy_condition_1 and buy_condition_2: # Short position + Long signal
            print(f"[{current_time}] 🔄 Reverse signal (Short -> Long) detected. Exiting Short and entering {triple_size} Long.")
            # Execute a BUY order for triple_size: This closes current short and opens a new long
            success = execute_trade(ft, exchange, symbol, triple_size, product_type, 'BUY')
            if success:
                position_size = triple_size
                entry_price = current_close


# ----------------------------------------------------------------------
## Main Execution Loop
# ----------------------------------------------------------------------

def run_live_trader():
    """Main function to run the live trading script."""
    
    # 1. Initialize API and get user inputs
    ft = initialize_flattrade()
    exchange, symbol, quantity, product_type_api, capital = get_user_inputs(default_capital=500)
    
    # For Equity, the tradingsymbol is usually appended with '-EQ'
    if '-EQ' not in symbol:
        symbol = f"{symbol}-EQ"

    # Define the trading hours window for the loop
    market_close_time = dt.time(15, 30) # 3:30 PM
    
    print("\n--- Starting Live Trading Loop (Checking every 5 minutes) ---")
    
    # Loop continuously during market hours
    while True:
        try:
            now = dt.datetime.now().time()
            if now < dt.time(9, 15) or now > market_close_time:
                print(f"Market closed/pre-open ({now}). Sleeping...")
                time.sleep(600) # Sleep for 10 minutes
                continue
            
            # 2. Fetch and prepare data
            df = fetch_and_prepare_data(ft, exchange, symbol)
            
            if df is not None:
                # 3. Apply strategy and trade
                check_and_trade(ft, exchange, symbol, quantity, product_type_api, df)
            
            # Wait for 5 minutes (or the candle interval time)
            print(f"Finished check. Current Position: {position_size}. Sleeping for 5 minutes...")
            time.sleep(300) 

        except KeyboardInterrupt:
            print("\n👋 Trading loop manually stopped.")
            break
        except Exception as e:
            print(f"\n❌ An unhandled error occurred in the trading loop: {e}. Retrying in 1 minute.")
            import traceback
            traceback.print_exc()
            time.sleep(60)

    print("\n--- Trading session finished. ---")
    print(f"Final Position Size: {position_size} (Positive for Long, Negative for Short)")
    
    # Final square-off check outside the main loop for any remaining position
    if position_size != 0:
        print("Attempting final square-off before exit...")
        transaction_type = 'BUY' if position_size < 0 else 'SELL'
        execute_trade(ft, exchange, symbol, abs(position_size), product_type_api, transaction_type)
        
    print("Script terminated.")

if __name__ == "__main__":
    # Ensure the Flattrade SDK is imported before running
    if 'Flattrade' not in locals():
         print("Error: Flattrade API SDK is not installed or imported correctly.")
         sys.exit(1)
         
    run_live_trader()
