# live_trader_olaelec.py

import pandas as pd
import numpy as np
import requests
import hashlib
import time
from datetime import datetime, time as dt_time
import json
import threading
from supabase import create_client, Client
import warnings
warnings.filterwarnings('ignore')

# --- Flattrade API Credentials ---
USER_SESSION = "f68b270591263a92f1d4182a6a5397142b0c254bdf885738c55d854445b3ac9c"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2"
API_SECRET = "2025.523da4413a454b8e878a11f8e10026205facdbeef612c23a"  # You'll need to get this from Flattrade
# live_trader_olaelec.py

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

class FlattradeAPI:
    def __init__(self, user_id, password, user_session, api_secret):
        self.user_id = user_id
        self.password = password
        self.user_session = user_session
        self.api_secret = api_secret
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.session_token = None
        self.login()
    
    def sha256_hash(self, data):
        """Generate SHA256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def login(self):
        """Login to Flattrade API"""
        try:
            # Create hash for login
            hash_string = f"{self.user_id}|{self.password}"
            app_key = self.sha256_hash(hash_string + self.api_secret)
            
            login_data = {
                "apkversion": "1.0.0",
                "uid": self.user_id,
                "pwd": self.sha256_hash(self.password),
                "factor2": "second_factor",
                "vc": "vendor_code",
                "appkey": app_key,
                "imei": "mac_address",
                "source": "API"
            }
            
            response = requests.post(f"{self.base_url}/QuickAuth", json=login_data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('stat') == 'Ok':
                    self.session_token = result.get('susertoken')
                    print("✅ Flattrade login successful")
                    return True
                else:
                    print(f"❌ Login failed: {result.get('emsg', 'Unknown error')}")
                    return False
            else:
                print(f"❌ Login request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return False
    
    def get_holdings(self):
        """Get current holdings"""
        try:
            data = {
                "uid": self.user_id,
                "actid": self.user_id
            }
            response = requests.post(f"{self.base_url}/Holdings", json=data)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"❌ Error getting holdings: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            data = {
                "uid": self.user_id,
                "actid": self.user_id
            }
            response = requests.post(f"{self.base_url}/PositionBook", json=data)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"❌ Error getting positions: {str(e)}")
            return None
    
    def place_order(self, symbol, quantity, side, order_type="MIS", price_type="MKT", price=0):
        """Place an order"""
        try:
            order_data = {
                "uid": self.user_id,
                "actid": self.user_id,
                "exch": "NSE",  # Default to NSE
                "tsym": symbol,
                "qty": str(quantity),
                "prc": str(price) if price > 0 else "0",
                "prd": order_type,  # MIS or CNC
                "trantype": side,   # B for buy, S for sell
                "prctyp": price_type,  # MKT for market, LMT for limit
                "ret": "DAY"
            }
            
            response = requests.post(f"{self.base_url}/PlaceOrder", json=order_data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('stat') == 'Ok':
                    print(f"✅ Order placed successfully: {side} {quantity} {symbol}")
                    return result
                else:
                    print(f"❌ Order failed: {result.get('emsg', 'Unknown error')}")
                    return None
            return None
            
        except Exception as e:
            print(f"❌ Error placing order: {str(e)}")
            return None
    
    def get_quotes(self, symbol, exchange="NSE"):
        """Get real-time quotes"""
        try:
            data = {
                "uid": self.user_id,
                "exch": exchange,
                "token": symbol
            }
            response = requests.post(f"{self.base_url}/GetQuotes", json=data)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"❌ Error getting quotes: {str(e)}")
            return None

class OLAELECLiveTrader:
    def __init__(self, flattrade_api, supabase_client):
        self.api = flattrade_api
        self.supabase = supabase_client
        
        # Trading parameters (will be set by user)
        self.symbol = None
        self.quantity = 1
        self.order_type = "MIS"
        self.capital = 500
        
        # Strategy parameters
        self.profit_target = 0.04  # 4% as in backtest
        self.entry_price = None
        self.current_position = 0  # 0: no position, +ve: long, -ve: short
        
        # Daily restrictions
        self.last_exit_date = None
        self.last_exit_direction = None
        self.target_hit_today = False
        self.current_date = None
        
        # Data storage
        self.price_data = []
        self.vwap_data = []
        
        # Trading status
        self.is_trading = False
        self.trading_thread = None
        
    def setup_trading(self):
        """Interactive setup for trading parameters"""
        print("\n" + "="*50)
        print("🚀 OLAELEC Live Trading Setup")
        print("="*50)
        
        # Get stock symbol
        self.symbol = input("📊 Enter stock symbol (e.g., RELIANCE, TCS): ").upper().strip()
        
        # Get quantity
        try:
            qty_input = input(f"📈 Enter quantity (default 1): ").strip()
            self.quantity = int(qty_input) if qty_input else 1
        except:
            self.quantity = 1
            
        # Get order type
        order_input = input("🔄 Enter order type - MIS/CNC (default MIS): ").upper().strip()
        self.order_type = order_input if order_input in ['MIS', 'CNC'] else 'MIS'
        
        # Get capital
        try:
            capital_input = input("💰 Enter capital amount (default 500): ").strip()
            self.capital = float(capital_input) if capital_input else 500
        except:
            self.capital = 500
            
        print(f"\n✅ Trading setup complete:")
        print(f"   Symbol: {self.symbol}")
        print(f"   Quantity: {self.quantity}")
        print(f"   Order Type: {self.order_type}")
        print(f"   Capital: ₹{self.capital}")
        print(f"   Profit Target: {self.profit_target*100}%")
        
        # Confirmation
        confirm = input("\n🚨 Start live trading? (y/N): ").lower().strip()
        return confirm == 'y'
    
    def calculate_vwap_bands(self, data_points=20):
        """Calculate VWAP bands (simplified version)"""
        if len(self.price_data) < data_points:
            return None, None
            
        # Get recent data
        recent_data = self.price_data[-data_points:]
        
        # Calculate VWAP
        total_volume = sum([d['volume'] for d in recent_data])
        if total_volume == 0:
            return None, None
            
        vwap = sum([d['price'] * d['volume'] for d in recent_data]) / total_volume
        
        # Calculate standard deviation
        prices = [d['price'] for d in recent_data]
        std_dev = np.std(prices)
        
        # VWAP bands (using 1 standard deviation as in backtest)
        vwap_plus = vwap + std_dev
        vwap_minus = vwap - std_dev
        
        return vwap_plus, vwap_minus
    
    def get_current_price_data(self):
        """Get current price and volume data"""
        try:
            quotes = self.api.get_quotes(self.symbol)
            if quotes and quotes.get('stat') == 'Ok':
                price = float(quotes.get('lp', 0))  # Last price
                volume = float(quotes.get('v', 0))   # Volume
                
                return {
                    'timestamp': datetime.now(),
                    'price': price,
                    'volume': volume,
                    'open': float(quotes.get('o', price)),
                    'high': float(quotes.get('h', price)),
                    'low': float(quotes.get('l', price)),
                    'close': price
                }
            return None
        except Exception as e:
            print(f"❌ Error getting price data: {str(e)}")
            return None
    
    def check_trading_hours(self):
        """Check if current time is within trading hours"""
        current_time = datetime.now().time()
        return dt_time(9, 30) <= current_time <= dt_time(15, 20)
    
    def check_exit_time(self):
        """Check if it's time to exit all positions"""
        current_time = datetime.now().time()
        return current_time >= dt_time(15, 20)
    
    def execute_trade(self, side, quantity):
        """Execute a trade"""
        try:
            result = self.api.place_order(
                symbol=self.symbol,
                quantity=quantity,
                side=side,  # 'B' for buy, 'S' for sell
                order_type=self.order_type,
                price_type="MKT"  # Market order
            )
            
            if result:
                # Log trade to Supabase
                self.log_trade_to_supabase(side, quantity, result)
                return True
            return False
            
        except Exception as e:
            print(f"❌ Trade execution error: {str(e)}")
            return False
    
    def log_trade_to_supabase(self, side, quantity, order_result):
        """Log trade details to Supabase"""
        try:
            trade_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'side': side,
                'quantity': quantity,
                'order_type': self.order_type,
                'order_result': order_result,
                'strategy': 'OLAELEC'
            }
            
            self.supabase.table('trades').insert(trade_data).execute()
            print(f"📝 Trade logged to database")
            
        except Exception as e:
            print(f"❌ Error logging trade: {str(e)}")
    
    def check_profit_target(self, current_price):
        """Check if profit target is reached"""
        if self.entry_price is None or self.current_position == 0:
            return False
            
        if self.current_position > 0:  # Long position
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= self.profit_target:
                return True
                
        elif self.current_position < 0:  # Short position
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct >= self.profit_target:
                return True
                
        return False
    
    def trading_loop(self):
        """Main trading loop"""
        print(f"\n🔄 Starting trading loop for {self.symbol}...")
        
        while self.is_trading:
            try:
                current_datetime = datetime.now()
                current_date = current_datetime.date()
                
                # Reset daily flags if new day
                if self.current_date != current_date:
                    self.current_date = current_date
                    self.target_hit_today = False
                    self.last_exit_date = None
                    self.last_exit_direction = None
                    print(f"📅 New trading day: {current_date}")
                
                # Check trading hours
                if not self.check_trading_hours():
                    time.sleep(60)  # Wait 1 minute if outside trading hours
                    continue
                
                # Get current price data
                price_data = self.get_current_price_data()
                if not price_data:
                    time.sleep(30)  # Wait 30 seconds and retry
                    continue
                
                self.price_data.append(price_data)
                
                # Keep only recent data (last 100 points)
                if len(self.price_data) > 100:
                    self.price_data = self.price_data[-100:]
                
                current_price = price_data['price']
                
                # Check if it's time to close all positions (3:20 PM)
                if self.check_exit_time():
                    if self.current_position != 0:
                        if self.current_position > 0:
                            self.execute_trade('S', abs(self.current_position))
                        else:
                            self.execute_trade('B', abs(self.current_position))
                        
                        self.current_position = 0
                        self.entry_price = None
                        self.target_hit_today = True
                        print("🕒 End of trading day - All positions closed")
                    continue
                
                # Check profit target
                if self.check_profit_target(current_price):
                    if self.current_position > 0:
                        self.execute_trade('S', abs(self.current_position))
                        self.last_exit_direction = 'long'
                    else:
                        self.execute_trade('B', abs(self.current_position))
                        self.last_exit_direction = 'short'
                    
                    self.current_position = 0
                    self.entry_price = None
                    self.target_hit_today = True
                    self.last_exit_date = current_date
                    print(f"🎯 Profit target reached! Position closed")
                    continue
                
                # Skip if target hit today
                if self.target_hit_today:
                    time.sleep(30)
                    continue
                
                # Need at least 2 data points for strategy
                if len(self.price_data) < 2:
                    time.sleep(30)
                    continue
                
                # Calculate VWAP bands
                vwap_plus, vwap_minus = self.calculate_vwap_bands()
                if vwap_plus is None or vwap_minus is None:
                    time.sleep(30)
                    continue
                
                # Get previous candle data
                candle_minus_1 = self.price_data[-1]
                candle_minus_2 = self.price_data[-2]
                
                # Strategy conditions (simplified from backtest)
                sell_condition = (
                    candle_minus_2['close'] < candle_minus_2['open'] and
                    candle_minus_2['close'] < vwap_minus and
                    candle_minus_1['close'] < candle_minus_1['open'] and
                    candle_minus_1['close'] < vwap_minus
                )
                
                buy_condition = (
                    candle_minus_2['close'] > candle_minus_2['open'] and
                    candle_minus_2['close'] > vwap_plus and
                    candle_minus_1['close'] > candle_minus_1['open'] and
                    candle_minus_1['close'] > vwap_plus
                )
                
                # Execute strategy
                if self.current_position == 0:  # No position
                    if sell_condition and self.last_exit_direction != 'short':
                        if self.execute_trade('S', self.quantity):
                            self.current_position = -self.quantity
                            self.entry_price = current_price
                            print(f"📉 Opened SHORT position: {self.quantity} @ ₹{current_price}")
                    
                    elif buy_condition and self.last_exit_direction != 'long':
                        if self.execute_trade('B', self.quantity):
                            self.current_position = self.quantity
                            self.entry_price = current_price
                            print(f"📈 Opened LONG position: {self.quantity} @ ₹{current_price}")
                
                else:  # Have position - check for reversal
                    if self.current_position > 0 and sell_condition:  # Long + Sell signal
                        triple_qty = self.quantity * 3
                        close_qty = self.current_position
                        new_short_qty = triple_qty - close_qty
                        
                        if self.execute_trade('S', triple_qty):
                            self.current_position = -new_short_qty
                            self.entry_price = current_price
                            print(f"🔄 Reversed to SHORT: {new_short_qty} @ ₹{current_price}")
                    
                    elif self.current_position < 0 and buy_condition:  # Short + Buy signal
                        triple_qty = self.quantity * 3
                        close_qty = abs(self.current_position)
                        new_long_qty = triple_qty - close_qty
                        
                        if self.execute_trade('B', triple_qty):
                            self.current_position = new_long_qty
                            self.entry_price = current_price
                            print(f"🔄 Reversed to LONG: {new_long_qty} @ ₹{current_price}")
                
                # Display current status
                if self.current_position != 0:
                    position_type = "LONG" if self.current_position > 0 else "SHORT"
                    pnl_pct = 0
                    if self.entry_price:
                        if self.current_position > 0:
                            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                        else:
                            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
                    
                    print(f"💼 Position: {position_type} {abs(self.current_position)} | "
                          f"Entry: ₹{self.entry_price:.2f} | Current: ₹{current_price:.2f} | "
                          f"P&L: {pnl_pct:.2f}%")
                
                time.sleep(30)  # Wait 30 seconds before next iteration
                
            except Exception as e:
                print(f"❌ Trading loop error: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error
    
    def start_trading(self):
        """Start the trading bot"""
        if not self.setup_trading():
            print("❌ Trading setup cancelled")
            return
        
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self.trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        print(f"\n🚀 Trading bot started for {self.symbol}")
        print("Press Ctrl+C to stop trading...")
        
        try:
            while self.is_trading:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop the trading bot"""
        print("\n🛑 Stopping trading bot...")
        self.is_trading = False
        
        # Close any open positions
        if self.current_position != 0:
            if self.current_position > 0:
                self.execute_trade('S', abs(self.current_position))
            else:
                self.execute_trade('B', abs(self.current_position))
            print("🔒 All positions closed")
        
        print("✅ Trading bot stopped successfully")

def main():
    """Main function"""
    print("🎯 OLAELEC Live Trading Bot")
    print("="*50)
    
    # Initialize Flattrade API
    flattrade_api = FlattradeAPI(
        user_id=USER_ID,
        password=FLATTRADE_PASSWORD,
        user_session=USER_SESSION,
        api_secret=API_SECRET
    )
    
    if not flattrade_api.session_token:
        print("❌ Failed to connect to Flattrade API")
        return
    
    # Initialize Supabase
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        supabase = None
    
    # Create and start trading bot
    trader = OLAELECLiveTrader(flattrade_api, supabase)
    trader.start_trading()

if __name__ == "__main__":
    main()
