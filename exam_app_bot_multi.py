# live_trader_olaelec_fixed.py

import pandas as pd
import numpy as np
import requests
import hashlib
import time
from datetime import datetime, time as dt_time
import json
import threading
import sys
import traceback

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("WARNING: Supabase not installed. Run: pip install supabase")
    SUPABASE_AVAILABLE = False

try:
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
    PYTZ_AVAILABLE = True
except ImportError:
    print("WARNING: pytz not installed. Run: pip install pytz")
    print("Using local time instead of IST")
    PYTZ_AVAILABLE = False

# --- Flattrade API Credentials ---
USER_SESSION = "f68b270591263a92f1d4182a6a5397142b0c254bdf885738c55d854445b3ac9c"
USER_ID = "FZ03508"
FLATTRADE_PASSWORD = "Shubhi@2"
API_SECRET = "your_api_secret_here"  # You need to get this from Flattrade

# --- Supabase Credentials ---
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

def get_current_time():
    """Get current time in IST if available, otherwise local time"""
    if PYTZ_AVAILABLE:
        return datetime.now(IST)
    else:
        return datetime.now()

class FlattradeAPI:
    def __init__(self, user_id, password, user_session, api_secret):
        print("Initializing Flattrade API...")
        self.user_id = user_id
        self.password = password
        self.user_session = user_session
        self.api_secret = api_secret
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.session_token = None
        
        # For demo mode if API secret not provided
        if api_secret == "your_api_secret_here":
            print("WARNING: API_SECRET not configured. Running in DEMO mode.")
            print("No actual trades will be placed.")
            self.demo_mode = True
        else:
            self.demo_mode = False
            self.login()
    
    def sha256_hash(self, data):
        """Generate SHA256 hash"""
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            print(f"ERROR: Hash generation failed: {str(e)}")
            return None
    
    def login(self):
        """Login to Flattrade API"""
        if self.demo_mode:
            print("DEMO MODE: Skipping actual login")
            self.session_token = "demo_token"
            return True
            
        try:
            print("Attempting Flattrade login...")
            
            # Create hash for login
            hash_string = f"{self.user_id}|{self.password}"
            app_key = self.sha256_hash(hash_string + self.api_secret)
            
            if not app_key:
                return False
            
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
            
            print("Sending login request...")
            response = requests.post(f"{self.base_url}/QuickAuth", json=login_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('stat') == 'Ok':
                    self.session_token = result.get('susertoken')
                    print("SUCCESS: Flattrade login successful")
                    return True
                else:
                    print(f"ERROR: Login failed: {result.get('emsg', 'Unknown error')}")
                    return False
            else:
                print(f"ERROR: Login request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"ERROR: Login error: {str(e)}")
            traceback.print_exc()
            return False
    
    def place_order(self, symbol, quantity, side, order_type="MIS", price_type="MKT", price=0):
        """Place an order"""
        if self.demo_mode:
            print(f"DEMO ORDER: {side} {quantity} {symbol} at market price")
            return {"stat": "Ok", "norenordno": "demo_order_123"}
            
        try:
            order_data = {
                "uid": self.user_id,
                "actid": self.user_id,
                "exch": "NSE",
                "tsym": symbol,
                "qty": str(quantity),
                "prc": str(price) if price > 0 else "0",
                "prd": order_type,
                "trantype": side,
                "prctyp": price_type,
                "ret": "DAY"
            }
            
            response = requests.post(f"{self.base_url}/PlaceOrder", json=order_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('stat') == 'Ok':
                    print(f"SUCCESS: Order placed: {side} {quantity} {symbol}")
                    return result
                else:
                    print(f"ERROR: Order failed: {result.get('emsg', 'Unknown error')}")
                    return None
            return None
            
        except Exception as e:
            print(f"ERROR: Error placing order: {str(e)}")
            return None
    
    def get_quotes(self, symbol, exchange="NSE"):
        """Get real-time quotes"""
        if self.demo_mode:
            # Return demo price data
            import random
            base_price = 100
            return {
                "stat": "Ok",
                "lp": str(base_price + random.uniform(-5, 5)),
                "v": str(random.randint(1000, 10000)),
                "o": str(base_price),
                "h": str(base_price + 2),
                "l": str(base_price - 2)
            }
            
        try:
            data = {
                "uid": self.user_id,
                "exch": exchange,
                "token": symbol
            }
            response = requests.post(f"{self.base_url}/GetQuotes", json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"ERROR: Error getting quotes: {str(e)}")
            return None

class OLAELECLiveTrader:
    def __init__(self, flattrade_api, supabase_client=None):
        print("Initializing OLAELEC Live Trader...")
        self.api = flattrade_api
        self.supabase = supabase_client
        
        # Trading parameters
        self.symbol = None
        self.quantity = 1
        self.order_type = "MIS"
        self.capital = 500
        
        # Strategy parameters
        self.profit_target = 0.04  # 4%
        self.entry_price = None
        self.current_position = 0
        
        # Daily restrictions
        self.last_exit_date = None
        self.last_exit_direction = None
        self.target_hit_today = False
        self.current_date = None
        
        # Data storage
        self.price_data = []
        
        # Trading status
        self.is_trading = False
        self.trading_thread = None
        
        print("OLAELEC Live Trader initialized successfully")
    
    def setup_trading(self):
        """Interactive setup for trading parameters"""
        try:
            print("\n" + "="*50)
            print("OLAELEC Live Trading Setup")
            print("="*50)
            
            # Get stock symbol
            symbol_input = input("Enter stock symbol (e.g., RELIANCE, TCS): ").strip()
            if not symbol_input:
                print("ERROR: No symbol entered")
                return False
            self.symbol = symbol_input.upper()
            
            # Get quantity
            try:
                qty_input = input("Enter quantity (default 1): ").strip()
                self.quantity = int(qty_input) if qty_input else 1
                if self.quantity <= 0:
                    print("ERROR: Quantity must be positive")
                    return False
            except ValueError:
                print("ERROR: Invalid quantity")
                return False
                
            # Get order type
            order_input = input("Enter order type - MIS/CNC (default MIS): ").upper().strip()
            self.order_type = order_input if order_input in ['MIS', 'CNC'] else 'MIS'
            
            # Get capital
            try:
                capital_input = input("Enter capital amount (default 500): ").strip()
                self.capital = float(capital_input) if capital_input else 500
                if self.capital <= 0:
                    print("ERROR: Capital must be positive")
                    return False
            except ValueError:
                print("ERROR: Invalid capital amount")
                return False
                
            print(f"\nTRADING SETUP COMPLETE:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Quantity: {self.quantity}")
            print(f"   Order Type: {self.order_type}")
            print(f"   Capital: Rs.{self.capital}")
            print(f"   Profit Target: {self.profit_target*100}%")
            
            # Confirmation
            confirm = input("\nStart live trading? (y/N): ").lower().strip()
            return confirm == 'y'
            
        except KeyboardInterrupt:
            print("\nSetup cancelled by user")
            return False
        except Exception as e:
            print(f"ERROR: Setup failed: {str(e)}")
            return False
    
    def get_current_price_data(self):
        """Get current price and volume data"""
        try:
            quotes = self.api.get_quotes(self.symbol)
            if quotes and quotes.get('stat') == 'Ok':
                price = float(quotes.get('lp', 0))
                volume = float(quotes.get('v', 0))
                
                return {
                    'timestamp': get_current_time(),
                    'price': price,
                    'volume': volume if volume > 0 else 1000,  # Default volume
                    'open': float(quotes.get('o', price)),
                    'high': float(quotes.get('h', price)),
                    'low': float(quotes.get('l', price)),
                    'close': price
                }
            return None
        except Exception as e:
            print(f"ERROR: Error getting price data: {str(e)}")
            return None
    
    def check_trading_hours(self):
        """Check if current time is within trading hours"""
        try:
            current_time = get_current_time().time()
            return dt_time(9, 30) <= current_time <= dt_time(15, 20)
        except Exception as e:
            print(f"ERROR: Error checking trading hours: {str(e)}")
            return False
    
    def check_exit_time(self):
        """Check if it's time to exit all positions"""
        try:
            current_time = get_current_time().time()
            return current_time >= dt_time(15, 20)
        except Exception as e:
            print(f"ERROR: Error checking exit time: {str(e)}")
            return False
    
    def execute_trade(self, side, quantity):
        """Execute a trade"""
        try:
            print(f"EXECUTING: {side} {quantity} {self.symbol}")
            result = self.api.place_order(
                symbol=self.symbol,
                quantity=quantity,
                side=side,
                order_type=self.order_type,
                price_type="MKT"
            )
            
            if result and result.get('stat') == 'Ok':
                # Log trade if Supabase is available
                if self.supabase:
                    self.log_trade_to_supabase(side, quantity, result)
                return True
            return False
            
        except Exception as e:
            print(f"ERROR: Trade execution error: {str(e)}")
            return False
    
    def log_trade_to_supabase(self, side, quantity, order_result):
        """Log trade details to Supabase"""
        try:
            trade_data = {
                'timestamp': get_current_time().isoformat(),
                'symbol': self.symbol,
                'side': side,
                'quantity': quantity,
                'order_type': self.order_type,
                'order_result': order_result,
                'strategy': 'OLAELEC'
            }
            
            self.supabase.table('trades').insert(trade_data).execute()
            print("LOGGED: Trade logged to database")
            
        except Exception as e:
            print(f"ERROR: Error logging trade: {str(e)}")
    
    def calculate_simple_vwap_bands(self):
        """Calculate simple VWAP bands"""
        try:
            if len(self.price_data) < 10:
                return None, None
            
            # Use last 20 data points or available data
            recent_data = self.price_data[-20:]
            
            # Simple VWAP calculation
            total_volume = sum([d['volume'] for d in recent_data])
            if total_volume == 0:
                return None, None
            
            vwap = sum([d['price'] * d['volume'] for d in recent_data]) / total_volume
            
            # Simple standard deviation
            prices = [d['price'] for d in recent_data]
            if len(prices) < 2:
                return None, None
                
            mean_price = sum(prices) / len(prices)
            variance = sum([(p - mean_price) ** 2 for p in prices]) / len(prices)
            std_dev = variance ** 0.5
            
            # VWAP bands
            vwap_plus = vwap + std_dev
            vwap_minus = vwap - std_dev
            
            return vwap_plus, vwap_minus
            
        except Exception as e:
            print(f"ERROR: VWAP calculation error: {str(e)}")
            return None, None
    
    def trading_loop(self):
        """Main trading loop"""
        print(f"Starting trading loop for {self.symbol}...")
        
        while self.is_trading:
            try:
                current_datetime = get_current_time()
                current_date = current_datetime.date()
                
                # Reset daily flags if new day
                if self.current_date != current_date:
                    self.current_date = current_date
                    self.target_hit_today = False
                    self.last_exit_date = None
                    self.last_exit_direction = None
                    print(f"NEW DAY: {current_date}")
                
                # Check trading hours
                if not self.check_trading_hours():
                    print("Outside trading hours, waiting...")
                    time.sleep(60)
                    continue
                
                # Get current price data
                price_data = self.get_current_price_data()
                if not price_data:
                    print("No price data available, retrying...")
                    time.sleep(30)
                    continue
                
                self.price_data.append(price_data)
                current_price = price_data['price']
                
                print(f"Current price: Rs.{current_price:.2f}")
                
                # Keep only recent data
                if len(self.price_data) > 100:
                    self.price_data = self.price_data[-100:]
                
                # Check exit time
                if self.check_exit_time():
                    if self.current_position != 0:
                        print("End of trading day - closing positions")
                        if self.current_position > 0:
                            self.execute_trade('S', abs(self.current_position))
                        else:
                            self.execute_trade('B', abs(self.current_position))
                        
                        self.current_position = 0
                        self.entry_price = None
                        self.target_hit_today = True
                    continue
                
                # Simple strategy implementation (demo)
                if len(self.price_data) >= 2 and not self.target_hit_today:
                    if self.current_position == 0:  # No position
                        # Simple buy condition (price rising)
                        if self.price_data[-1]['price'] > self.price_data[-2]['price']:
                            if self.execute_trade('B', self.quantity):
                                self.current_position = self.quantity
                                self.entry_price = current_price
                                print(f"LONG ENTRY: {self.quantity} @ Rs.{current_price}")
                
                # Display position status
                if self.current_position != 0:
                    position_type = "LONG" if self.current_position > 0 else "SHORT"
                    pnl_pct = 0
                    if self.entry_price:
                        if self.current_position > 0:
                            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                        else:
                            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
                    
                    print(f"POSITION: {position_type} {abs(self.current_position)} | "
                          f"Entry: Rs.{self.entry_price:.2f} | P&L: {pnl_pct:.2f}%")
                
                time.sleep(30)  # Wait 30 seconds
                
            except KeyboardInterrupt:
                print("Trading interrupted by user")
                break
            except Exception as e:
                print(f"ERROR: Trading loop error: {str(e)}")
                traceback.print_exc()
                time.sleep(60)
    
    def start_trading(self):
        """Start the trading bot"""
        try:
            if not self.setup_trading():
                print("Trading setup cancelled")
                return
            
            self.is_trading = True
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            print(f"\nSTARTED: Trading bot for {self.symbol}")
            print("Press Ctrl+C to stop trading...")
            
            # Keep main thread alive
            try:
                while self.is_trading and self.trading_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_trading()
                
        except Exception as e:
            print(f"ERROR: Failed to start trading: {str(e)}")
            traceback.print_exc()
    
    def stop_trading(self):
        """Stop the trading bot"""
        print("\nSTOPPING: Trading bot...")
        self.is_trading = False
        
        # Close any open positions
        if self.current_position != 0:
            print("Closing all positions...")
            if self.current_position > 0:
                self.execute_trade('S', abs(self.current_position))
            else:
                self.execute_trade('B', abs(self.current_position))
            print("All positions closed")
        
        print("Trading bot stopped successfully")

def main():
    """Main function with error handling"""
    try:
        print("OLAELEC Live Trading Bot")
        print("="*50)
        
        # Check if API secret is configured
        if API_SECRET == "your_api_secret_here":
            print("WARNING: Running in DEMO mode")
            print("To enable real trading, update API_SECRET in the code")
        
        # Initialize Flattrade API
        print("Setting up Flattrade connection...")
        flattrade_api = FlattradeAPI(
            user_id=USER_ID,
            password=FLATTRADE_PASSWORD,
            user_session=USER_SESSION,
            api_secret=API_SECRET
        )
        
        # Initialize Supabase if available
        supabase = None
        if SUPABASE_AVAILABLE:
            try:
                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("SUCCESS: Connected to Supabase")
            except Exception as e:
                print(f"WARNING: Supabase connection failed: {str(e)}")
        
        # Create and start trading bot
        trader = OLAELECLiveTrader(flattrade_api, supabase)
        trader.start_trading()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
    finally:
        print("Program ended")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"STARTUP ERROR: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")
