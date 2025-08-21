import os
import sys
import logging
import datetime
import time
import pandas as pd

# Add the parent directory to the sys.path to import api_helper
# This assumes api_helper.py is in the parent directory of this script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_helper import NorenApiPy

# --- Configuration ---
# Set up logging for detailed output. Change to logging.DEBUG to see detailed screening checks.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual Flattrade user ID and session token.
# IMPORTANT: You need to generate this token via the Flattrade login flow.
USER_SESSION = '63ed108c037529ecdcab5fd089923e35c0437fbfa2caf5031a87ebaef500d444'  # <<< --- REPLACE THIS ---
USER_ID = 'FZ03508'              # <<< --- REPLACE THIS ---

EXCHANGE = 'NSE'
CANDLE_INTERVAL = '1'  # 1-minute candles
REQUIRED_CANDLES = 2 # IMPORTANT: We need the latest candle + previous 20 for calculations

# Screening criteria constants
VOLUME_MULTIPLIER = .1 # Volume of last candle should be greater than 10 times of last 20 candles
TRADED_VALUE_THRESHOLD = 100 # 1 Crore: traded value of last candle should be greater than 10,000,000
HIGH_LOW_DIFF_MULTIPLIER = .1 # Difference of high low of present candle should be 4 times of average of difference of high low of last 20 candles

# --- Trading Parameters ---
# IMPORTANT: Adjusted CAPITAL to a more realistic value for Nifty500 stocks.
# Adjust this based on your actual trading capital.
CAPITAL = 1000  # Example Capital in INR, e.g., 1 Lakh. Adjust as needed.

# Based on your request "quantity = capital/10", this implies the total value of shares
# should be 1/10th of your capital. So, total_order_value = CAPITAL / 10.
# Quantity = int(total_order_value / current_price)
# QUANTITY_FACTOR will be used as a divisor for CAPITAL before dividing by current_price
# so QUANTITY_FACTOR = 10 means 1/10th of capital is used for the trade value.
QUANTITY_FACTOR = 5

STOPLOSS_FACTOR = 1000 # stoploss amount = CAPITAL / STOPLOSS_FACTOR
TARGET_MULTIPLIER = 4 # target amount = TARGET_MULTIPLIER * stoploss_amount

# --- Initialize Flattrade API ---
api = NorenApiPy()

def initialize_api_session(user_id, user_session):
    """
    Initializes the Flattrade API session.
    Args:
        user_id (str): The user ID for the Flattrade account.
        user_session (str): The session token for authentication.
    Returns:
        bool: True if the session is set successfully, False otherwise.
    """
    logging.info("Attempting to set API session...")
    try:
        ret = api.set_session(userid=user_id, password='', usertoken=user_session)
        # Check if ret is explicitly True (for boolean success cases)
        if ret is True:
            logging.info(f"API session set successfully for user: {user_id}")
            return True
        # Check if ret is a dictionary and has 'stat': 'Ok' (for dict success cases)
        elif isinstance(ret, dict) and ret.get('stat') == 'Ok':
            logging.info(f"API session set successfully for user: {user_id}")
            return True
        # If it's not True and not a successful dict, it's a failure
        else:
            error_msg = ret.get('emsg', 'Unknown error') if isinstance(ret, dict) else str(ret)
            logging.error(f"Failed to set API session: {error_msg}")
            return False
    except Exception as e:
        logging.critical(f"An exception occurred during API session setup: {e}")
        return False

def get_nifty500_symbols():
    """
    Placeholder function to get Nifty500 stock symbols and their tokens.
    In a real scenario, you would fetch this list from a reliable source
    (e.g., a pre-defined CSV, an external API, or by using api.searchscrip
    iteratively if you have a way to identify Nifty500 components).

    For demonstration, we'll use a small dummy list.
    You can use api.searchscrip to find tokens if you know the trading symbol:
    
    # Example for searching a scrip and getting its token
    # search_result = api.searchscrip(exchange='NSE', searchtext='RELIANCE-EQ')
    # if search_result and search_result.get('stat') == 'Ok':
    #     for symbol in search_result['values']:
    #         if symbol['tsym'] == 'RELIANCE-EQ':
    #             print(f"Reliance token: {symbol['token']}")
    #             break
    """
    logging.info("Using dummy Nifty500 symbol list. Please replace with actual data.")
    # Dummy list of Nifty500 stocks (Exchange, Token, Trading Symbol)
    # You would need to populate the 'token' correctly for each symbol
    # by searching via `api.searchscrip` or getting a predefined list.
    nifty500_symbols = [
        {'exchange': 'NSE', 'token': '10794', 'tsym': 'CANBK-EQ'},    # Reliance Industries
        
    ]
    return nifty500_symbols

def screen_stock(stock_info):
    """
    Screens a single stock based on the defined criteria using 1-minute candle data.

    Args:
        stock_info (dict): A dictionary containing 'exchange', 'token', 'tsym'.

    Returns:
        tuple: (stock_symbol, 'BUY'/'SELL'/'NEUTRAL', reason, current_price)
               current_price is returned only if a signal is generated, else None.
    """
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']

    logging.info(f"Screening {tradingsymbol}...")

    # Calculate start time for fetching 21 candles (20 for average + 1 current)
    end_time = datetime.datetime.now()
    # Adding a buffer of 5 minutes to ensure we get enough data, especially around minute boundaries
    start_time = end_time - datetime.timedelta(minutes=REQUIRED_CANDLES + 5) 
    
    try:
        # Fetch 1-minute time price series data
        # 'intrv' is the interval volume
        # 'intc' is the interval close (LTP)
        # 'inth' is the interval high
        # 'intl' is the interval low
        # 'into' is the interval open
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=int(start_time.timestamp()),
            endtime=int(end_time.timestamp()),
            interval=CANDLE_INTERVAL
        )

        if not candle_data or len(candle_data) < REQUIRED_CANDLES:
            logging.warning(f"Not enough 1-minute candle data for {tradingsymbol}. Required: {REQUIRED_CANDLES}, Got: {len(candle_data) if candle_data else 0}")
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None

        # The API returns data in descending order of time, so the first element is the latest.
        current_candle = candle_data[0]
        # Get the previous 20 candles for average calculations
        previous_20_candles = candle_data[1:REQUIRED_CANDLES] 

        # --- 1. Volume Check ---
        current_volume = float(current_candle.get('intv', 0))
        
        # Log values for debugging
        logging.debug(f"{tradingsymbol} - Current Volume: {current_volume}")

        if current_volume == 0:
            logging.debug(f"{tradingsymbol}: Current candle volume is zero.")
            return tradingsymbol, 'NEUTRAL', 'Current candle volume is zero', None
            
        previous_volumes = [float(c.get('intv', 0)) for c in previous_20_candles]
        # Filter out zero volumes to avoid division by zero or skewed averages
        non_zero_previous_volumes = [v for v in previous_volumes if v > 0]

        if not non_zero_previous_volumes:
            logging.debug(f"{tradingsymbol}: No non-zero volumes in previous 20 candles for average calculation.")
            return tradingsymbol, 'NEUTRAL', 'No valid volume data in previous 20 candles', None

        average_volume_last_20 = sum(non_zero_previous_volumes) / len(non_zero_previous_volumes)
        logging.debug(f"{tradingsymbol} - Average 20-candle Volume: {average_volume_last_20}")

        if not (current_volume > VOLUME_MULTIPLIER * average_volume_last_20):
            logging.debug(f"{tradingsymbol}: Volume condition not met. Current: {current_volume}, Avg 20: {average_volume_last_20}. Required > {VOLUME_MULTIPLIER}x.")
            return tradingsymbol, 'NEUTRAL', 'Volume condition not met', None

        # --- 2. Traded Value Check ---
        current_close_price = float(current_candle.get('intc', 0))
        logging.debug(f"{tradingsymbol} - Current Close Price: {current_close_price}")

        if current_close_price == 0:
            logging.debug(f"{tradingsymbol}: Current candle close price is zero.")
            return tradingsymbol, 'NEUTRAL', 'Current candle close price is zero', None
            
        current_traded_value = current_volume * current_close_price
        logging.debug(f"{tradingsymbol} - Current Traded Value: {current_traded_value}")
        
        if not (current_traded_value > TRADED_VALUE_THRESHOLD):
            logging.debug(f"{tradingsymbol}: Traded value condition not met. Current: {current_traded_value}, Required > {TRADED_VALUE_THRESHOLD}.")
            return tradingsymbol, 'NEUTRAL', 'Traded value condition not met', None

        # --- 3. High-Low Difference Check ---
        current_high = float(current_candle.get('inth', 0))
        current_low = float(current_candle.get('intl', 0))
        current_high_low_diff = current_high - current_low
        logging.debug(f"{tradingsymbol} - Current High-Low Diff: {current_high_low_diff}")

        if current_high_low_diff <= 0: # Avoid division by zero or meaningless diff
            logging.debug(f"{tradingsymbol}: Current high-low difference is zero or negative.")
            return tradingsymbol, 'NEUTRAL', 'Current high-low difference invalid', None

        previous_high_low_diffs = []
        for c in previous_20_candles:
            high = float(c.get('inth', 0))
            low = float(c.get('intl', 0))
            if high > 0 and low > 0: # Ensure valid prices
                diff = high - low
                if diff > 0: # Only consider positive differences
                    previous_high_low_diffs.append(diff)

        if not previous_high_low_diffs:
            logging.debug(f"{tradingsymbol}: No valid high-low differences in previous 20 candles for average calculation.")
            return tradingsymbol, 'NEUTRAL', 'No valid high-low diff data in previous 20 candles', None

        average_high_low_diff_last_20 = sum(previous_high_low_diffs) / len(previous_high_low_diffs)
        logging.debug(f"{tradingsymbol} - Average 20-candle High-Low Diff: {average_high_low_diff_last_20}")

        if not (current_high_low_diff > HIGH_LOW_DIFF_MULTIPLIER * average_high_low_diff_last_20):
            logging.debug(f"{tradingsymbol}: High-low diff condition not met. Current: {current_high_low_diff}, Avg 20: {average_high_low_diff_last_20}. Required > {HIGH_LOW_DIFF_MULTIPLIER}x.")
            return tradingsymbol, 'NEUTRAL', 'High-low diff condition not met', None

        # --- 4. Candle Color Check ---
        current_open_price = float(current_candle.get('into', 0))
        logging.debug(f"{tradingsymbol} - Current Open: {current_open_price}, Current Close: {current_close_price}")
        
        if current_close_price > current_open_price:
            return tradingsymbol, 'BUY', 'All conditions met: Green candle', current_close_price
        elif current_close_price < current_open_price:
            return tradingsymbol, 'SELL', 'All conditions met: Red candle', current_close_price
        else:
            return tradingsymbol, 'NEUTRAL', 'Current candle is Doji (Open == Close)', None

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}", exc_info=True)
        return tradingsymbol, 'NEUTRAL', f'Error during screening: {e}', None

def place_intraday_bracket_order(
    buy_or_sell, tradingsymbol, quantity, current_price, capital, api
):
    """
    Places an Intraday Bracket Order (Product Type 'B' for Bracket Order).

    Args:
        buy_or_sell (str): 'B' for Buy, 'S' for Sell.
        tradingsymbol (str): The trading symbol of the instrument (e.g., 'RELIANCE-EQ').
        quantity (int): Number of shares to trade.
        current_price (float): The current market price to use as the entry price.
        capital (float): The total capital available for calculations.
        api (NorenApiPy): The initialized Flattrade API object.

    Returns:
        dict: The API response for the order placement.
    """
    # Calculate stoploss and target amounts based on capital
    # Ensure quantity is not zero before division
    if quantity == 0:
        logging.error(f"Cannot place order for {tradingsymbol}: Quantity is zero.")
        return {'stat': 'Not_Ok', 'emsg': 'Quantity is zero'}

    # Calculate risk per share for stoploss and target
    # This approach calculates the stoploss *value* based on capital,
    # then divides by quantity to get the per-share point value.
    stoploss_amount_total = capital / STOPLOSS_FACTOR
    stoploss_points_per_share = round(stoploss_amount_total / quantity, 2)
    target_points_per_share = round(TARGET_MULTIPLIER * stoploss_points_per_share, 2)

    if stoploss_points_per_share <= 0:
        logging.error(f"Cannot place order for {tradingsymbol}: Calculated stoploss points per share is zero or negative.")
        return {'stat': 'Not_Ok', 'emsg': 'Invalid stoploss calculation'}

    # Determine stoploss and target prices
    if buy_or_sell == 'B':
        bookloss_price = round(current_price - stoploss_points_per_share, 2)
        bookprofit_price = round(current_price + target_points_per_share, 2)
    elif buy_or_sell == 'S':
        bookloss_price = round(current_price + stoploss_points_per_share, 2)
        bookprofit_price = round(current_price - target_points_per_share, 2)
    else:
        logging.error(f"Invalid buy_or_sell type: {buy_or_sell}")
        return {'stat': 'Not_Ok', 'emsg': 'Invalid buy_or_sell type'}

    # Ensure prices are positive and sensible (e.g., target price > 0 for sell)
    if bookloss_price <= 0 or bookprofit_price <= 0:
         logging.error(f"Calculated bookloss_price or bookprofit_price is zero or negative for {tradingsymbol}. SL: {bookloss_price}, Target: {bookprofit_price}")
         return {'stat': 'Not_Ok', 'emsg': 'Invalid calculated prices'}

    logging.info(f"Placing {buy_or_sell} Bracket Order for {tradingsymbol}: "
                 f"Qty={quantity}, Entry={current_price}, SL={bookloss_price}, Target={bookprofit_price}")

    try:
        order_response = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type='B',  # 'B' for Bracket Order (Intraday with SL & Target)
            exchange=EXCHANGE,
            tradingsymbol=tradingsymbol,
            quantity=int(quantity), # Quantity must be an integer
            discloseqty=0,
            price_type='LMT',  # Entry order as Limit order (can be 'MKT' if preferred)
            price=current_price,
            trigger_price=None, # Not directly used for entry in BO
            retention='DAY',   # Bracket orders are typically intraday
            remarks='Automated_Screener_Trade',
            bookloss_price=bookloss_price,
            bookprofit_price=bookprofit_price,
            # 'trail_price' parameter in place_order implies point-based trailing SL.
            # It's usually the points by which the stop-loss trails the market.
            trail_price=stoploss_points_per_share # Trailing stoploss set to SL points
        )
        if order_response and order_response.get('stat') == 'Ok':
            logging.info(f"Order placed successfully for {tradingsymbol}: {order_response}")
        else:
            logging.error(f"Failed to place order for {tradingsymbol}: {order_response.get('emsg', 'Unknown error')}")
        return order_response
    except Exception as e:
        logging.error(f"An error occurred while placing order for {tradingsymbol}: {e}", exc_info=True)
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def main():
    # Pass USER_ID and USER_SESSION explicitly
    if not initialize_api_session(USER_ID, USER_SESSION):
        logging.critical("Exiting program due to API session failure.")
        return

    nifty500_stocks = get_nifty500_symbols()
    buy_list = []
    sell_list = []

    logging.info(f"Starting screening for {len(nifty500_stocks)} Nifty500 stocks...")

    for i, stock in enumerate(nifty500_stocks):
        # Implement a small delay to avoid hitting API rate limits
        time.sleep(0.1) # Adjust as necessary

        symbol_info, signal, reason, current_price = screen_stock(stock)
        
        if signal == 'BUY':
            buy_list.append(symbol_info)
            logging.info(f"🟢 BUY SIGNAL for {symbol_info}: {reason}")
            if current_price: # Ensure we have a valid current price for order placement
                # Calculate quantity based on CAPITAL, current_price and QUANTITY_FACTOR
                # Quantity = (CAPITAL / current_price) / QUANTITY_FACTOR
                # Example: If CAPITAL = 100000, current_price = 1000, QUANTITY_FACTOR = 10
                # Quantity = (100000 / 10) / 1000 = 10000 / 1000 = 10 shares
                total_investment_amount = CAPITAL / QUANTITY_FACTOR
                calculated_quantity = int(total_investment_amount / current_price) if current_price > 0 else 0

                if calculated_quantity > 0:
                    logging.info(f"Attempting to place BUY order for {symbol_info} with quantity {calculated_quantity}...")
                    place_intraday_bracket_order(
                        buy_or_sell='B',
                        tradingsymbol=stock['tsym'],
                        quantity=calculated_quantity,
                        current_price=current_price,
                        capital=CAPITAL,
                        api=api
                    )
                else:
                    logging.warning(f"Calculated quantity for {symbol_info} is zero. Not placing order. Capital: {CAPITAL}, Price: {current_price}, Investment Amount: {total_investment_amount}")

        elif signal == 'SELL':
            sell_list.append(symbol_info)
            logging.info(f"🔴 SELL SIGNAL for {symbol_info}: {reason}")
            if current_price: # Ensure we have a valid current price for order placement
                total_investment_amount = CAPITAL / QUANTITY_FACTOR
                calculated_quantity = int(total_investment_amount / current_price) if current_price > 0 else 0

                if calculated_quantity > 0:
                    logging.info(f"Attempting to place SELL order for {symbol_info} with quantity {calculated_quantity}...")
                    place_intraday_bracket_order(
                        buy_or_sell='S',
                        tradingsymbol=stock['tsym'],
                        quantity=calculated_quantity,
                        current_price=current_price,
                        capital=CAPITAL,
                        api=api
                    )
                else:
                    logging.warning(f"Calculated quantity for {symbol_info} is zero. Not placing order. Capital: {CAPITAL}, Price: {current_price}, Investment Amount: {total_investment_amount}")
        else:
            logging.info(f"⚫ NEUTRAL for {symbol_info}: {reason}")
        
        if (i + 1) % 10 == 0: # Print progress every 10 stocks
            logging.info(f"Processed {i + 1}/{len(nifty500_stocks)} stocks.")

    print("\n" + "="*50)
    print("           NIFTY500 STOCK SCREENING RESULTS          ")
    print("="*50)
    print(f"\n🟢 **BUY LIST ({len(buy_list)} stocks):**")
    if buy_list:
        for stock in buy_list:
            print(f"- {stock}")
    else:
        print("No stocks met BUY criteria.")

    print(f"\n🔴 **SELL LIST ({len(sell_list)} stocks):**")
    if sell_list:
        for stock in sell_list:
            print(f"- {stock}")
    else:
        print("No stocks met SELL criteria.")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
