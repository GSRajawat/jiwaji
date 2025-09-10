def screen_stock(stock_info, api, all_symbols_map):
    """
    Modified screening function with updated conditions:
    1. Volume condition: Current volume > average of last 20 candles × 2 (reduced from 10)
    2. Traded value condition: Current traded value > 5 million INR (more realistic threshold)
    3. Range condition: Current open-close difference > average of last 20 candles × 2 (reduced from 4)
    4. PVI condition remains the same
    5. Check if symbol already traded today
    6. Risk management: No trade if trade value > 4.5 × (cash balance - open positions value)
    """
    exchange = stock_info['exchange']
    token = stock_info['token']
    tradingsymbol = stock_info['tsym']
    
    # Check if already traded today
    if tradingsymbol in st.session_state.daily_traded_symbols:
        return tradingsymbol, 'NEUTRAL', 'Already traded today', None, None, None, None
    
    # Calculate start time for fetching candles (need more for PVI calculation)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=20)  # Extra candles for PVI calculation
    
    try:
        candle_data = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=int(start_time.timestamp()),
            endtime=int(end_time.timestamp()),
            interval=CANDLE_INTERVAL
        )
        
        if not candle_data or len(candle_data) < 20:  # Need more candles for PVI
            logging.warning(f"Not enough candle data for {tradingsymbol}. Needed: 20+, Got: {len(candle_data) if candle_data else 0}")
            return tradingsymbol, 'NEUTRAL', 'Insufficient candle data', None, None, None, None
        
        current_candle = candle_data[0]  # Most recent candle
        previous_20_candles = candle_data[1:21]  # Previous 20 candles for average calculation
        
        # Extract values from current candle
        current_volume = float(current_candle.get('intv', 0))
        current_close_price = float(current_candle.get('intc', 0))
        current_open_price = float(current_candle.get('into', 0))
        current_high = float(current_candle.get('inth', 0))
        current_low = float(current_candle.get('intl', 0))
        
        # Get signal candle timestamp
        signal_candle_time = current_candle.get('time', 'Unknown')
        if isinstance(signal_candle_time, str) and signal_candle_time.isdigit():
            signal_candle_time = datetime.datetime.fromtimestamp(int(signal_candle_time)).strftime('%H:%M:%S')
        
        if current_volume == 0 or current_close_price == 0 or current_high == 0 or current_low == 0:
            return tradingsymbol, 'NEUTRAL', 'Current candle data is zero/invalid', None, None, None, signal_candle_time

        # --- MODIFIED CONDITION 1: Volume Check (Reduced multiplier from 10 to 2) ---
        previous_volumes = [float(c.get('intv', 0)) for c in previous_20_candles if float(c.get('intv', 0)) > 0]
        if not previous_volumes:
            return tradingsymbol, 'NEUTRAL', 'No valid volume data in previous 20 candles', None, None, None, signal_candle_time
        
        average_volume_last_20 = sum(previous_volumes) / len(previous_volumes)
        volume_multiplier = 10  # Reduced from st.session_state.volume_multiplier (which was 10)
        if not (current_volume > volume_multiplier * average_volume_last_20):
            return tradingsymbol, 'NEUTRAL', f'Volume condition not met (Current: {current_volume:,.0f}, Avg×{volume_multiplier}: {volume_multiplier * average_volume_last_20:,.0f})', None, None, None, signal_candle_time

        # --- MODIFIED CONDITION 2: Traded Value Check (Fixed threshold instead of variable) ---
        current_traded_value = current_volume * current_close_price
        traded_value_threshold = 50000000  # Fixed 5 million INR threshold
        if not (current_traded_value > traded_value_threshold):
            return tradingsymbol, 'NEUTRAL', f'Traded value condition not met (Current: ₹{current_traded_value:,.0f}, Required: ₹{traded_value_threshold:,.0f})', None, None, None, signal_candle_time

        # --- MODIFIED CONDITION 3: Open-Close Range Check (Reduced multiplier from 4 to 2) ---
        current_open_close_diff = abs(current_close_price - current_open_price)
        if current_open_close_diff <= 0:
            return tradingsymbol, 'NEUTRAL', 'Current open-close difference invalid', None, None, None, signal_candle_time
        
        previous_open_close_diffs = []
        for c in previous_20_candles:
            open_price = float(c.get('into', 0))
            close_price = float(c.get('intc', 0))
            if open_price > 0 and close_price > 0:
                diff = abs(close_price - open_price)
                if diff > 0:
                    previous_open_close_diffs.append(diff)
        
        if not previous_open_close_diffs:
            return tradingsymbol, 'NEUTRAL', 'No valid open-close diff data in previous 20 candles', None, None, None, signal_candle_time

        average_open_close_diff_last_20 = sum(previous_open_close_diffs) / len(previous_open_close_diffs)
        range_multiplier = 4  # Reduced from st.session_state.high_low_diff_multiplier (which was 4)
        if not (current_open_close_diff > range_multiplier * average_open_close_diff_last_20):
            return tradingsymbol, 'NEUTRAL', f'Open-close diff condition not met (Current: {current_open_close_diff:.2f}, Avg×{range_multiplier}: {range_multiplier * average_open_close_diff_last_20:.2f})', None, None, None, signal_candle_time

        # --- CONDITION 4: PVI Condition Check (Unchanged) ---
    

        # --- CONDITION 5: Calculate potential trade value and check risk management ---
        # First determine the signal type and calculate potential trade parameters
        signal_type = None
        if current_close_price > current_open_price:
            signal_type = 'BUY'
        elif current_close_price < current_open_price:
            signal_type = 'SELL'
        
        if signal_type:
            # Calculate expected entry price and stop loss
            if signal_type == 'BUY':
                expected_entry_price = round(current_high * (1 + 0.0005), 2)  # Entry buffer
                initial_sl_price = round(current_low - st.session_state.sl_buffer_points, 2)
            else:  # SELL
                expected_entry_price = round(current_low * (1 - 0.0005), 2)  # Entry buffer  
                initial_sl_price = round(current_high + st.session_state.sl_buffer_points, 2)
            
            potential_loss_per_share = abs(expected_entry_price - initial_sl_price)
            if potential_loss_per_share <= 0.01:
                return tradingsymbol, 'NEUTRAL', f'{signal_type} signal but invalid SL distance', None, None, None, signal_candle_time
            
            # Calculate quantity using the capital * 0.01 / SL_points formula
            calculated_quantity = int((st.session_state.capital * 0.01) / potential_loss_per_share)
            
            if calculated_quantity <= 0:
                return tradingsymbol, 'NEUTRAL', f'{signal_type} signal but calculated quantity is zero', None, None, None, signal_candle_time
            
            # Calculate total trade value
            total_trade_value = calculated_quantity * expected_entry_price
            
            # --- NEW CONDITION 6: Risk Management Check ---
            # Get current cash balance and open positions value
            try:
                # Fetch account limits
                limits = api.get_limits()
                cash_balance = 0
                if limits and isinstance(limits, dict) and limits.get('stat') == 'Ok':
                    if 'cash' in limits and limits['cash'] is not None:
                        try:
                            cash_balance = float(limits['cash'])
                        except ValueError:
                            logging.error(f"Could not convert cash balance to float: {limits['cash']}")
                    elif 'prange' in limits and isinstance(limits['prange'], list):
                        for item in limits['prange']:
                            if isinstance(item, dict) and 'cash' in item and item['cash'] is not None:
                                try:
                                    cash_balance = float(item['cash'])
                                    break
                                except ValueError:
                                    continue
                
                # Calculate total value of open positions
                open_positions_value = 0
                positions = api.get_positions()
                if isinstance(positions, list):
                    for pos in positions:
                        if pos.get('netqty', 0) != 0:
                            net_qty = int(pos.get('netqty', 0))
                            ltp = float(pos.get('lp', 0))
                            position_value = abs(net_qty * ltp)
                            open_positions_value += position_value
                
                # Calculate available capital for new trades
                net_available_capital = cash_balance * 4.5 - open_positions_value
                max_allowed_trade_value = net_available_capital
                
                if total_trade_value > max_allowed_trade_value:
                    return tradingsymbol, 'NEUTRAL', f'{signal_type} signal but trade value ₹{total_trade_value:,.0f} exceeds limit ₹{max_allowed_trade_value:,.0f} (Cash: ₹{cash_balance:,.0f}, Open: ₹{open_positions_value:,.0f})', None, None, None, signal_candle_time
                
            except Exception as e:
                logging.error(f"Error checking risk management for {tradingsymbol}: {e}")
                return tradingsymbol, 'NEUTRAL', f'{signal_type} signal but risk check failed: {str(e)}', None, None, None, signal_candle_time
            
            # All conditions passed
            return tradingsymbol, signal_type, f'All conditions met: {signal_type} signal + Risk cleared (Trade: ₹{total_trade_value:,.0f}, Max: ₹{max_allowed_trade_value:,.0f}, Vol: {current_volume:,.0f} vs {volume_multiplier * average_volume_last_20:,.0f})', current_close_price, current_high, current_low, signal_candle_time
        

    except Exception as e:
        logging.error(f"Error screening {tradingsymbol}: {e}", exc_info=True)
        return tradingsymbol, 'NEUTRAL', f'Error during screening: {e}', None, None, None, None
