
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import requests
import json
import time
from datetime import datetime, timedelta
import threading
import hashlib
import hmac
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from collections import defaultdict
import numpy as np
import pytz # Add this import

# Supabase configuration
SUPABASE_URL = "https://zybakxpyibubzjhzdcwl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp5YmFreHB5aWJ1YnpqaHpkY3dsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ4OTQyMDgsImV4cCI6MjA3MDQ3MDIwOH0.8ZqreKy5zg_M-B1uH79T6lQXn62eRvvouo_OiMjwqGU"

# Flattrade configuration
FLATTRADE_CONFIG = {
    "USER_SESSION": "b1de36f2c66db606b5dbe5fe45dfb40b0a2f391d0265edd01af0ad972a9d48db",
    "USER_ID": "FZ03508",
    "PASSWORD": "Shubhi@2",
    "BASE_URL": "https://piconnect.flattrade.in/PiConnectTP"
}

class FlattradeAPI:
    def __init__(self):
        self.config = FLATTRADE_CONFIG
        self.session = requests.Session()
    
    def get_market_data(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get real-time market data from Flattrade"""
        try:
            # Note: This is a simplified example. You'll need to implement proper Flattrade API calls
            # based on their documentation. This includes proper authentication and API endpoints.
            
            url = f"{self.config['BASE_URL']}/GetQuotes"
            payload = {
                "uid": self.config["USER_ID"],
                "actid": self.config["USER_ID"],
                "exch": exchange,
                "token": symbol,
            }
            
            # Add authentication headers as required by Flattrade API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['USER_SESSION']}"
            }
            
            response = self.session.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def get_historical_data(self, symbol: str, exchange: str = "NSE", 
                          from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get historical candle data"""
        try:
            # Implement historical data fetching based on Flattrade API
            # This is a placeholder - you'll need to implement based on actual API
            url = f"{self.config['BASE_URL']}/TPSeries"
            payload = {
                "uid": self.config["USER_ID"],
                "exch": exchange,
                "token": symbol,
                "st": from_date or datetime.now().strftime("%d-%m-%Y"),
                "et": to_date or datetime.now().strftime("%d-%m-%Y"),
                "intrv": "1"  # 1 minute interval
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['USER_SESSION']}"
            }
            
            response = self.session.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
class SupabaseManager:
    def __init__(self):
        try:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self.connection_status = "Connected"
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.connection_status = f"Failed: {str(e)}"
            self.client = None
    
    def test_connection(self) -> tuple[bool, str]:
        """Test Supabase connection"""
        if not self.client:
            return False, "Client not initialized"
        
        try:
            # Try to access a system table to test connection
            result = self.client.from_('stock_candles').select('*').limit(1).execute()
            return True, "Connected successfully"
        except Exception as e:
            error_msg = str(e).lower()
            if "relation" in error_msg and "does not exist" in error_msg:
                return False, "Table 'stock_candles' does not exist. Please create it first."
            elif "invalid" in error_msg or "unauthorized" in error_msg:
                return False, f"Authentication failed: {str(e)}"
            else:
                return False, f"Connection error: {str(e)}"
    
    def create_candle_table(self) -> bool:
        """Create the candle data table if it doesn't exist"""
        try:
            if not self.client:
                st.error("Supabase client not initialized")
                return False
                
            # SQL schema for creating the table
            sql_schema = """
            CREATE TABLE IF NOT EXISTS stock_candles (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                open_price DECIMAL(10, 2) NOT NULL,
                high_price DECIMAL(10, 2) NOT NULL,
                low_price DECIMAL(10, 2) NOT NULL,
                close_price DECIMAL(10, 2) NOT NULL,
                volume BIGINT NOT NULL,
                vwap DECIMAL(10, 2),
                sdvwap1_plus DECIMAL(10, 2),
                sdvwap1_minus DECIMAL(10, 2),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_stock_candles_symbol_timestamp 
            ON stock_candles(symbol, timestamp DESC);
            """
            
            st.info("🔧 **Database Setup Required**")
            st.markdown("""
            **Step 1:** Go to your Supabase Dashboard → SQL Editor
            
            **Step 2:** Run this SQL command:
            """)
            st.code(sql_schema, language="sql")
            
            st.markdown("""
            **Step 3:** After running the SQL, click 'Test Connection' below to verify.
            """)
            
            # Try to execute the SQL directly (requires RLS to be disabled or proper policies)
            try:
                # This might not work if RLS is enabled without proper policies
                result = self.client.rpc('sql', {'query': sql_schema}).execute()
                st.success("✅ Table created successfully!")
                return True
            except Exception as e:
                st.warning(f"⚠️ Could not create table automatically: {str(e)}")
                st.info("Please run the SQL manually in your Supabase dashboard.")
                return False
                
        except Exception as e:
            st.error(f"Error in table creation process: {e}")
            return False
    
    def insert_candle_data(self, candle_data: Dict) -> bool:
        """Insert candle data into Supabase"""
        if not self.client:
            logger.error("Supabase client not initialized")
            return False
            
        try:
            # Adjust field names to match database schema
            formatted_data = {
                "symbol": candle_data.get("symbol"),
                "timestamp": candle_data.get("timestamp"),
                "open_price": candle_data.get("open"),
                "high_price": candle_data.get("high"),
                "low_price": candle_data.get("low"),
                "close_price": candle_data.get("close"),
                "volume": candle_data.get("volume"),
                "vwap": candle_data.get("vwap"),
                "sdvwap1_plus": candle_data.get("sdvwap1_plus"),
                "sdvwap1_minus": candle_data.get("sdvwap1_minus")
            }
            
            result = self.client.table('stock_candles').insert(formatted_data).execute()
            return True
        except Exception as e:
            logger.error(f"Error inserting candle data: {e}")
            return False
    
    def get_latest_candles(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get latest candle data for a symbol"""
        if not self.client:
            logger.error("Supabase client not initialized")
            return pd.DataFrame()
            
        try:
            result = self.client.table('stock_candles')\
                .select('*')\
                .eq('symbol', symbol)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                # Rename columns to match expected format
                column_mapping = {
                    'open_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'close_price': 'close'
                }
                df = df.rename(columns=column_mapping)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching candle data: {e}")
            return pd.DataFrame()
    
    def get_historical_data_for_indicators(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """Get historical data needed for indicator calculations"""
        if not self.client:
            return pd.DataFrame()
            
        try:
            result = self.client.table('stock_candles')\
                .select('timestamp, open_price, high_price, low_price, close_price, volume')\
                .eq('symbol', symbol)\
                .order('timestamp', desc=False)\
                .limit(limit)\
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        if df.empty or len(df) == 0:
            return None
        
        try:
            # VWAP = Sum(Price * Volume) / Sum(Volume)
            # Using typical price (HLC/3) for calculation
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            price_volume = typical_price * df['volume']
            
            vwap = price_volume.sum() / df['volume'].sum()
            return float(vwap)
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return None
    
    @staticmethod
    def calculate_sdvwap(df: pd.DataFrame, period: int = 14, multiplier: float = 1.0) -> tuple:
        """
        Calculate Standard Deviation VWAP bands
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for standard deviation calculation (default: 14)
            multiplier: Standard deviation multiplier (default: 1.0)
            
        Returns:
            tuple: (sdvwap_plus, sdvwap_minus)
        """
        if df.empty or len(df) < period:
            return None, None
        
        try:
            # Calculate VWAP first
            vwap = TechnicalIndicators.calculate_vwap(df)
            if vwap is None:
                return None, None
            
            # Calculate typical price
            typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
            
            # Take the last 'period' values for standard deviation calculation
            recent_prices = typical_price.tail(period)
            
            # Calculate standard deviation
            std_dev = recent_prices.std()
            
            # Calculate bands
            sdvwap_plus = vwap + (multiplier * std_dev)
            sdvwap_minus = vwap - (multiplier * std_dev)
            
            return float(sdvwap_plus), float(sdvwap_minus)
            
        except Exception as e:
            logger.error(f"Error calculating SDVWAP: {e}")
            return None, None

class FlattradeAPI:
    def __init__(self):
        self.config = FLATTRADE_CONFIG
        self.session = requests.Session()
    
    def get_market_data(self, symbol: str, exchange: str = "NSE") -> Optional[Dict]:
        """Get real-time market data from Flattrade"""
        try:
            # Note: This is a simplified example. You'll need to implement proper Flattrade API calls
            # based on their documentation. This includes proper authentication and API endpoints.
            
            url = f"{self.config['BASE_URL']}/GetQuotes"
            payload = {
                "uid": self.config["USER_ID"],
                "actid": self.config["USER_ID"],
                "exch": exchange,
                "token": symbol,
            }
            
            # Add authentication headers as required by Flattrade API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['USER_SESSION']}"
            }
            
            response = self.session.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def get_historical_data(self, symbol: str, exchange: str = "NSE", 
                          from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get historical candle data"""
        try:
            # Implement historical data fetching based on Flattrade API
            # This is a placeholder - you'll need to implement based on actual API
            url = f"{self.config['BASE_URL']}/TPSeries"
            payload = {
                "uid": self.config["USER_ID"],
                "exch": exchange,
                "token": symbol,
                "st": from_date or datetime.now().strftime("%d-%m-%Y"),
                "et": to_date or datetime.now().strftime("%d-%m-%Y"),
                "intrv": "1"  # 1 minute interval
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['USER_SESSION']}"
            }
            
            response = self.session.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []

class DataRecorder:
    def __init__(self):
        self.supabase_manager = SupabaseManager()
        self.flattrade_api = FlattradeAPI()
        self.technical_indicators = TechnicalIndicators()
        self.is_recording = False
        self.recording_threads = {}
        self.recording_stats = {}  # Changed from defaultdict to regular dict
        self.data_queue = queue.Queue()
        self.max_workers = 200  # Maximum concurrent threads for API calls
    
    def _init_symbol_stats(self, symbol: str):
        """Initialize statistics for a symbol"""
        self.recording_stats[symbol] = {
            'last_update': None,
            'total_records': 0,
            'errors': 0,
            'status': 'Stopped'
        }
    
    def start_recording(self, symbols: List[str], interval_seconds: int = 60):
        """Start recording data for given symbols"""
        if self.is_recording:
            return False, "Recording already in progress"
        
        if not symbols:
            return False, "No symbols provided"
        
        self.is_recording = True
        
        # Initialize stats for all symbols
        for symbol in symbols:
            self._init_symbol_stats(symbol)
            self.recording_stats[symbol]['status'] = 'Starting...'
        
        # Start main recording thread
        self.main_recording_thread = threading.Thread(
            target=self._main_recording_loop,
            args=(symbols, interval_seconds),
            daemon=True
        )
        self.main_recording_thread.start()
        
        # Start database writer thread
        self.db_writer_thread = threading.Thread(
            target=self._database_writer_loop,
            daemon=True
        )
        self.db_writer_thread.start()
        
        return True, f"Started recording for {len(symbols)} symbols"
    
    def stop_recording(self):
        """Stop recording data"""
        if not self.is_recording:
            return "Recording was not active"
        
        self.is_recording = False
        
        # Update all symbols status
        for symbol in self.recording_stats:
            self.recording_stats[symbol]['status'] = 'Stopping...'
        
        # Wait for threads to finish
        if hasattr(self, 'main_recording_thread') and self.main_recording_thread.is_alive():
            self.main_recording_thread.join(timeout=5)
        
        if hasattr(self, 'db_writer_thread') and self.db_writer_thread.is_alive():
            self.db_writer_thread.join(timeout=5)
        
        # Update final status
        for symbol in self.recording_stats:
            self.recording_stats[symbol]['status'] = 'Stopped'
        
        return "Recording stopped successfully"
    
    def _main_recording_loop(self, symbols: List[str], interval_seconds: int):
        """Main recording loop using ThreadPoolExecutor for concurrent API calls"""
        logger.info(f"Starting recording for symbols: {symbols}")
        
        while self.is_recording:
            start_time = time.time()
            current_time = datetime.now()
            
            # Use ThreadPoolExecutor for concurrent API calls
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all API calls concurrently
                future_to_symbol = {
                    executor.submit(self._fetch_and_process_symbol, symbol, current_time): symbol
                    for symbol in symbols
                }
                
                # Process completed futures as they finish
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            self.recording_stats[symbol]['status'] = 'Recording'
                            self.recording_stats[symbol]['last_update'] = current_time
                        else:
                            self.recording_stats[symbol]['errors'] += 1
                            self.recording_stats[symbol]['status'] = 'Error'
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        self.recording_stats[symbol]['errors'] += 1
                        self.recording_stats[symbol]['status'] = 'Error'
            
            # Calculate how long to sleep to maintain interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Recording cycle took {elapsed_time:.2f}s, longer than interval {interval_seconds}s")
    
    def _fetch_and_process_symbol(self, symbol: str, timestamp: datetime) -> bool:
        """Fetch and process data for a single symbol"""
        try:
            # Get market data from Flattrade
            market_data = self.flattrade_api.get_market_data(symbol)
            
            if market_data:
                # Parse the data and create candle record
                candle_data = self._parse_market_data(symbol, market_data, timestamp)
                
                if candle_data:
                    # Add to queue for database insertion
                    self.data_queue.put((symbol, candle_data))
                    return True
            return False
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return False
    
    def _database_writer_loop(self):
        """Separate thread for database writes to avoid blocking API calls"""
        logger.info("Database writer thread started")
        
        while self.is_recording or not self.data_queue.empty():
            try:
                # Get data from queue with timeout
                symbol, candle_data = self.data_queue.get(timeout=1)
                
                # Insert into database
                success = self.supabase_manager.insert_candle_data(candle_data)
                
                if success:
                    if symbol not in self.recording_stats:
                        self._init_symbol_stats(symbol)
                    self.recording_stats[symbol]['total_records'] += 1
                    logger.info(f"Recorded data for {symbol}")
                else:
                    if symbol not in self.recording_stats:
                        self._init_symbol_stats(symbol)
                    self.recording_stats[symbol]['errors'] += 1
                    logger.error(f"Failed to record data for {symbol}")
                
                self.data_queue.task_done()
                
            except queue.Empty:
                # No data in queue, continue
                continue
            except Exception as e:
                logger.error(f"Database writer error: {e}")
    
    def get_recording_stats(self) -> Dict:
        """Get current recording statistics"""
        return dict(self.recording_stats)
    
    def _parse_market_data(self, symbol: str, market_data: Dict, timestamp: datetime) -> Optional[Dict]:
        """Parse market data into candle format with technical indicators"""
        try:
            # Basic OHLCV data
            ohlcv_data = {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "open": float(market_data.get('o', 0)),
                "high": float(market_data.get('h', 0)),
                "low": float(market_data.get('l', 0)),
                "close": float(market_data.get('c', 0)),
                "volume": int(market_data.get('v', 0))
            }
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(symbol, ohlcv_data)
            
            # Merge OHLCV data with indicators
            ohlcv_data.update(indicators)
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Error parsing market data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, symbol: str, current_data: Dict) -> Dict:
        """Calculate technical indicators for the current candle"""
        try:
            # Get historical data for indicator calculations
            historical_df = self.supabase_manager.get_historical_data_for_indicators(symbol, limit=50)
            
            # Create current candle data in the same format
            current_candle = {
                'timestamp': pd.to_datetime(current_data['timestamp']),
                'open_price': current_data['open'],
                'high_price': current_data['high'],
                'low_price': current_data['low'],
                'close_price': current_data['close'],
                'volume': current_data['volume']
            }
            
            # If we have historical data, append current candle
            if not historical_df.empty:
                current_df = pd.DataFrame([current_candle])
                combined_df = pd.concat([historical_df, current_df], ignore_index=True)
            else:
                # If no historical data, use just current candle (indicators will be None)
                combined_df = pd.DataFrame([current_candle])
            
            # Calculate VWAP
            vwap = self.technical_indicators.calculate_vwap(combined_df)
            
            # Calculate SDVWAP with parameters: period=14, multiplier=1.0
            sdvwap_plus, sdvwap_minus = self.technical_indicators.calculate_sdvwap(
                combined_df, period=14, multiplier=1.0
            )
            
            return {
                'vwap': vwap,
                'sdvwap1_plus': sdvwap_plus,
                'sdvwap1_minus': sdvwap_minus
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {
                'vwap': None,
                'sdvwap1_plus': None,
                'sdvwap1_minus': None
            }


import streamlit as st
import time
from datetime import datetime, timedelta

# Add this after your existing configuration
DEFAULT_SYMBOLS = ["ACC", "APLAPOLLO", "AUBANK", "ATGL", "ABCAPITAL", "ABFRL", "ALKEM", "APOLLOTYRE", "ASHOKLEY", "ASTRAL", "AUROPHARMA", "BSE", "BANDHANBNK", "BANKINDIA", "MAHABANK", "BDL", "BHARATFORG", "BHEL", "BHARTIHEXA", "BIOCON", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CUMMINSIND", "DIXON", "ESCORTS", "EXIDEIND", "NYKAA", "FEDERALBNK", "GMRAIRPORT", "GLENMARK", "GODREJPROP", "HDFCAMC", "HINDPETRO", "HINDZINC", "HUDCO", "IDFCFIRSTB", "IRB", "INDIANB", "IRCTC", "IREDA", "IGL", "INDUSTOWER", "JUBLFOOD", "KPITTECH", "KALYANKJIL", "LTF", "LICHSGFIN", "LUPIN", "MRF", "M&MFIN", "MANKIND", "MARICO", "MFSL", "MAXHEALTH", "MAZDOCK", "MOTILALOFS", "MPHASIS", "MUTHOOTFIN", "NHPC", "NMDC", "NTPCGREEN", "NATIONALUM", "OBEROIRLTY", "OIL", "OLAELEC", "PAYTM", "OFSS", "POLICYBZR", "PIIND", "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET", "PHOENIXLTD", "POLYCAB", "PREMIERENE", "PRESTIGE", "RVNL", "SBICARD", "SJVN", "SRF", "SOLARINDS", "SONACOMS", "SAIL", "SUPREMEIND", "SUZLON", "TATACOMM", "TATAELXSI", "TATATECH", "TORNTPOWER", "TIINDIA", "UPL", "UNIONBANK", "VMM", "IDEA", "VOLTAS", "WAAREEENER", "YESBANK"]
# Streamlit Cloud specific configuration
PING_BASED_RECORDING = True  # Enable ping-based recording for Streamlit Cloud
RECORD_ON_STARTUP = True     # Record immediately when app starts

def is_market_hours() -> bool:
    """
    Check if the current time is within Indian market hours (IST),
    which are from 9:15 AM to 3:30 PM on weekdays (Monday to Friday).
    """
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)

    # Check if it's a weekday (Monday=0, Friday=4)
    if now_ist.weekday() >= 5: # Saturday or Sunday
        return False

    # Define market open and close times in IST
    market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    # Check if the current time is within the market hours
    if market_open_time <= now_ist <= market_close_time:
        return True
    else:
        return False
    
def should_record_now():
    """Check if we should record data now (avoid duplicates)"""
    current_time = datetime.now()
    
    # Create a unique key for this 5-minute window
    time_window = current_time.replace(second=0, microsecond=0)
    time_window = time_window - timedelta(minutes=time_window.minute % 5)
    record_key = f"recorded_{time_window.strftime('%Y%m%d_%H%M')}"
    
    # Check if we already recorded in this window
    if record_key in st.session_state:
        return False
    
    # Mark this window as recorded
    st.session_state[record_key] = True
    
    # Clean up old keys (keep only last 12 windows = 1 hour)
    keys_to_remove = []
    for key in st.session_state.keys():
        if key.startswith('recorded_') and key != record_key:
            try:
                key_time = datetime.strptime(key.split('_')[1] + '_' + key.split('_')[2], '%Y%m%d_%H%M')
                if (current_time - key_time).total_seconds() > 3600:  # Older than 1 hour
                    keys_to_remove.append(key)
            except:
                keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    return True

def ping_based_recording():
    """Record data immediately when UptimeRobot pings (Streamlit Cloud compatible)"""
    
    # Only proceed if we should record now
    if not should_record_now():
        st.info("⏭️ Already recorded in this time window")
        return
    
    # Only record during market hours
    if not is_market_hours():
        st.info("🕐 Outside market hours - skipping recording")
        return
    
    current_time = datetime.now()
    
    # Show recording status
    st.info(f"🎯 Ping-based recording started at {current_time.strftime('%H:%M:%S')}")
    
    # Initialize data recorder if not exists
    if 'data_recorder' not in st.session_state:
        st.session_state.data_recorder = DataRecorder()
    
    # Record data for all symbols immediately
    symbols = DEFAULT_SYMBOLS
    results = {'success': 0, 'errors': 0, 'details': []}
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Recording {symbol}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            # Get market data
            market_data = st.session_state.data_recorder.flattrade_api.get_market_data(symbol)
            
            if market_data:
                # Parse data with indicators
                candle_data = st.session_state.data_recorder._parse_market_data(
                    symbol, market_data, current_time
                )
                
                if candle_data:
                    # Insert into database
                    success = st.session_state.data_recorder.supabase_manager.insert_candle_data(candle_data)
                    
                    if success:
                        results['success'] += 1
                        results['details'].append(f"✅ {symbol}: ₹{candle_data['close']}")
                        logger.info(f"Ping recorded: {symbol} - ₹{candle_data['close']}")
                    else:
                        results['errors'] += 1
                        results['details'].append(f"❌ {symbol}: Database error")
                else:
                    results['errors'] += 1
                    results['details'].append(f"❌ {symbol}: Parse error")
            else:
                results['errors'] += 1
                results['details'].append(f"❌ {symbol}: API error")
                
        except Exception as e:
            results['errors'] += 1
            results['details'].append(f"❌ {symbol}: {str(e)[:50]}")
            logger.error(f"Ping recording error for {symbol}: {e}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    if results['success'] > 0:
        st.success(f"🎯 Ping Recording Complete: {results['success']}/{len(symbols)} successful")
        
        # Show details in expander
        with st.expander("📋 Recording Details"):
            for detail in results['details']:
                if detail.startswith('✅'):
                    st.success(detail)
                else:
                    st.error(detail)
    else:
        st.error(f"❌ Ping Recording Failed: {results['errors']} errors")
        
    # Store stats in session state
    st.session_state['last_ping_recording'] = {
        'time': current_time,
        'success': results['success'],
        'errors': results['errors'],
        'total': len(symbols)
    }

def show_ping_recording_stats():
    """Show statistics from ping-based recordings"""
    if 'last_ping_recording' in st.session_state:
        stats = st.session_state['last_ping_recording']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Last Recording", stats['time'].strftime('%H:%M:%S'))
        
        with col2:
            st.metric("Successful", stats['success'])
        
        with col3:
            st.metric("Errors", stats['errors'])
        
        with col4:
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

# Enhanced main function for Streamlit Cloud
def main():
    st.set_page_config(
        page_title="Stock Data Recorder - Streamlit Cloud",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock OHLC Data Recorder (Streamlit Cloud)")
    st.markdown("Ping-based recording every 5 minutes via UptimeRobot")
    st.sidebar.success("✅ Stock Recorder is running")
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone)
    st.write(f"🕐 Current Time: {current_time_ist.strftime('%H:%M:%S')}")
    st.sidebar.info(f"📅 Market Hours: {'✅' if is_market_hours() else '❌'}")
    
    # Initialize components
    if 'data_recorder' not in st.session_state:
        st.session_state.data_recorder = DataRecorder()
    
    # PING-BASED RECORDING - This runs on every UptimeRobot ping
    if PING_BASED_RECORDING and RECORD_ON_STARTUP:
        ping_based_recording()
    
    # Main dashboard
    st.subheader("🎯 Ping-Based Recording Dashboard")
    
    # Show current recording stats
    show_ping_recording_stats()
    
    # Market status
    if is_market_hours():
        st.success("🟢 Market is OPEN - Recording active")
    else:
        st.warning("🟡 Market is CLOSED - Recording paused")
    
    # Manual recording button
    st.subheader("🔧 Manual Controls")
    
    if st.button("🚀 Record Now (Manual)", type="primary"):
        ping_based_recording()
    
    # Show recent data
    st.subheader("📊 Recent Data")
    
    symbol_to_view = st.selectbox("Select Symbol", DEFAULT_SYMBOLS)
    
    if st.button("📈 View Recent Data"):
        df = st.session_state.data_recorder.supabase_manager.get_latest_candles(
            symbol_to_view, limit=20
        )
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            # Show latest price
            latest = df.iloc[0]
            st.metric(f"{symbol_to_view} Latest Price", f"₹{latest['close']:.2f}")
            
            # Show data table
            display_df = df[['timestamp', 'close', 'volume', 'vwap']].head(10)
            st.dataframe(display_df, use_container_width=True)
            
            # Simple chart
            if len(df) > 1:
                chart_data = df.set_index('timestamp')[['close', 'vwap']].tail(10)
                st.line_chart(chart_data)
        else:
            st.info(f"No data found for {symbol_to_view}")
    
    # Connection status
    st.subheader("🔗 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test database connection
        is_connected, message = st.session_state.data_recorder.supabase_manager.test_connection()
        if is_connected:
            st.success("✅ Database Connected")
        else:
            st.error(f"❌ Database: {message}")
    
    with col2:
        # API status
        st.info("🔄 Flattrade API Ready")
    
    # Instructions
    st.subheader("ℹ️ How It Works")
    st.info("""
    **Streamlit Cloud Ping-Based Recording:**
    
    1. 📡 **UptimeRobot pings every 5 minutes**
    2. 🚀 **App wakes up and runs ping_based_recording()**  
    3. 📊 **Records data for all symbols immediately**
    4. 💾 **Saves to Supabase database**
    5. 😴 **App goes to sleep until next ping**
    
    **Result:** You get fresh data every 5 minutes during market hours!
    
    ⚠️ **Note:** This is not continuous background recording, but timed data collection.
    """)
    
    # Show environment info
    with st.expander("🔧 Configuration"):
        st.write("**Symbols:**", DEFAULT_SYMBOLS)
        st.write("**Market Hours:** 9:15 AM - 3:30 PM (Mon-Fri)")
        st.write("**Recording Frequency:** Every 5 minutes via UptimeRobot")
        st.write("**Platform:** Streamlit Community Cloud")

if __name__ == "__main__":
    main()


# Streamlit App
def main():
    st.set_page_config(
        page_title="Stock Data Recorder",
        page_icon="📈",
        layout="wide"
    )
    # Call this in your main() function
    ping_based_recording()
    st.title("📈 Stock OHLC Data Recorder")
    st.markdown("Record 1-minute candle data for stocks using Flattrade API and store in Supabase")
    
    # Initialize components
    if 'data_recorder' not in st.session_state:
        st.session_state.data_recorder = DataRecorder()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Database setup
        st.subheader("Database Setup")
        
        # Test connection button
        if st.button("🔍 Test Connection"):
            is_connected, message = st.session_state.data_recorder.supabase_manager.test_connection()
            if is_connected:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        
        # Show current status
        status = st.session_state.data_recorder.supabase_manager.connection_status
        if "Connected" in status:
            st.success(f"Status: {status}")
        else:
            st.error(f"Status: {status}")
        
        if st.button("🔧 Setup Database Table"):
            st.session_state.data_recorder.supabase_manager.create_candle_table()
        
        # Recording configuration
        st.subheader("Recording Settings")
        
        # Symbol input
        symbols_input = st.text_area(
            "Stock Symbols (one per line)",
            value="ACC\nAPLAPOLLO\nAUBANK\nATGL\nABCAPITAL\nABFRL\nALKEM\nAPOLLOTYRE\nASHOKLEY\nASTRAL\nAUROPHARMA\nBSE\nBANDHANBNK\nBANKINDIA\nMAHABANK\nBDL\nBHARATFORG\nBHEL\nBHARTIHEXA\nBIOCON\nCOCHINSHIP\nCOFORGE\nCOLPAL\nCONCOR\nCUMMINSIND\nDIXON\nESCORTS\nEXIDEIND\nNYKAA\nFEDERALBNK\nGMRAIRPORT\nGLENMARK\nGODREJPROP\nHDFCAMC\nHINDPETRO\nHINDZINC\nHUDCO\nIDFCFIRSTB\nIRB\nINDIANB\nIRCTC\nIREDA\nIGL\nINDUSTOWER\nJUBLFOOD\nKPITTECH\nKALYANKJIL\nLTF\nLICHSGFIN\nLUPIN\nMRF\nM&MFIN\nMANKIND\nMARICO\nMFSL\nMAXHEALTH\nMAZDOCK\nMOTILALOFS\nMPHASIS\nMUTHOOTFIN\nNHPC\nNMDC\nNTPCGREEN\nNATIONALUM\nOBEROIRLTY\nOIL\nOLAELEC\nPAYTM\nOFSS\nPOLICYBZR\nPIIND\nPAGEIND\nPATANJALI\nPERSISTENT\nPETRONET\nPHOENIXLTD\nPOLYCAB\nPREMIERENE\nPRESTIGE\nRVNL\nSBICARD\nSJVN\nSRF\nSOLARINDS\nSONACOMS\nSAIL\nSUPREMEIND\nSUZLON\nTATACOMM\nTATAELXSI\nTATATECH\nTORNTPOWER\nTIINDIA\nUPL\nUNIONBANK\nVMM\nIDEA\nVOLTAS\nWAAREEENER\nYESBANK",
            help="Enter stock symbols, one per line"
        )
        
        symbols = [symbol.strip().upper() for symbol in symbols_input.split('\n') if symbol.strip()]
        
        # Recording interval
        interval = st.selectbox(
            "Recording Interval",
            options=[60, 300, 900, 1800, 3600],
            format_func=lambda x: f"{x//60} minute{'s' if x//60 != 1 else ''}",
            index=0
        )
        
        # Recording controls
        st.subheader("Recording Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Recording", type="primary"):
                if symbols:
                    success, message = st.session_state.data_recorder.start_recording(symbols, interval)
                    if success:
                        st.success(message)
                        st.session_state.recording_status = "Recording"
                    else:
                        st.error(message)
                else:
                    st.error("Please enter at least one symbol!")
        
        with col2:
            if st.button("⏹️ Stop Recording", type="secondary"):
                message = st.session_state.data_recorder.stop_recording()
                st.success(message)
                st.session_state.recording_status = "Stopped"
        
        # Status
        status = getattr(st.session_state, 'recording_status', 'Stopped')
        st.write(f"**Status:** {status}")
        
        if st.session_state.data_recorder.is_recording:
            st.write("🟢 Recording Active")
        else:
            st.write("🔴 Recording Inactive")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Real-time Recording Dashboard")
        
        # Show recording statistics if recording is active
        if st.session_state.data_recorder.is_recording:
            stats = st.session_state.data_recorder.get_recording_stats()
            
            if stats:
                # Create metrics columns
                metrics_cols = st.columns(min(len(stats), 4))
                
                for idx, (symbol, stat) in enumerate(stats.items()):
                    col_idx = idx % 4
                    with metrics_cols[col_idx]:
                        # Color code based on status
                        if stat['status'] == 'Recording':
                            status_color = "🟢"
                        elif stat['status'] == 'Error':
                            status_color = "🔴"
                        elif 'Starting' in stat['status'] or 'Stopping' in stat['status']:
                            status_color = "🟡"
                        else:
                            status_color = "⚪"
                        
                        st.metric(
                            label=f"{status_color} {symbol}",
                            value=f"{stat['total_records']} records",
                            delta=f"{stat['errors']} errors" if stat['errors'] > 0 else "No errors"
                        )
                        
                        if stat['last_update']:
                            st.caption(f"Last: {stat['last_update'].strftime('%H:%M:%S')}")
                        else:
                            st.caption("No data yet")
                
                # Detailed statistics table
                st.subheader("📈 Detailed Statistics")
                
                stats_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'Status': stat['status'],
                        'Records': stat['total_records'],
                        'Errors': stat['errors'],
                        'Success Rate': f"{((stat['total_records'] / max(stat['total_records'] + stat['errors'], 1)) * 100):.1f}%",
                        'Last Update': stat['last_update'].strftime('%H:%M:%S') if stat['last_update'] else 'Never'
                    }
                    for symbol, stat in stats.items()
                ])
                
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Auto-refresh every 30 seconds when recording
                time.sleep(0.1)  # Small delay to prevent excessive refreshing
                st.rerun()
        
        else:
            st.info("📴 Recording is stopped. Start recording to see real-time dashboard.")
        
        # Data viewer section
        st.subheader("📋 Historical Data Viewer")
        
        # Symbol selector for viewing data
        if symbols:
            selected_symbol = st.selectbox("Select Symbol to View", symbols, key="data_viewer_symbol")
            
            col_refresh, col_limit = st.columns([1, 1])
            
            with col_refresh:
                refresh_data = st.button("🔄 Refresh Data")
            
            with col_limit:
                data_limit = st.selectbox("Records to show", [20, 50, 100, 200], index=1)
            
            if refresh_data or st.session_state.get('auto_refresh_data', False):
                # Fetch latest data for selected symbol
                df = st.session_state.data_recorder.supabase_manager.get_latest_candles(
                    selected_symbol, limit=data_limit
                )
                
                if not df.empty:
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp', ascending=False)
                    
                    # Display summary metrics
                    latest_record = df.iloc[0]
                    col1_metrics, col2_metrics, col3_metrics, col4_metrics = st.columns(4)
                    
                    with col1_metrics:
                        st.metric("Latest Close", f"₹{latest_record['close']:.2f}")
                    with col2_metrics:
                        st.metric("Volume", f"{latest_record['volume']:,}")
                    with col3_metrics:
                        if len(df) > 1:
                            price_change = latest_record['close'] - df.iloc[1]['close']
                            st.metric("Change", f"₹{price_change:.2f}", delta=f"{price_change:.2f}")
                        else:
                            st.metric("Change", "N/A")
                    with col4_metrics:
                        st.metric("Total Records", len(df))
                    
                    # Display data table
                    display_df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'sdvwap1_plus', 'sdvwap1_minus']].head(20)
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Price chart
                    if len(df) > 1:
                        st.subheader(f"📈 {selected_symbol} Price Chart")
                        chart_df = df.set_index('timestamp')[['open', 'high', 'low', 'close']].tail(50)
                        st.line_chart(chart_df)
                        
                        # Volume chart
                        st.subheader(f"📊 {selected_symbol} Volume Chart")
                        volume_df = df.set_index('timestamp')[['volume']].tail(50)
                        st.bar_chart(volume_df)
                else:
                    st.info(f"No data found for {selected_symbol}")
        else:
            st.info("Enter symbols in the sidebar to view data")
    
    with col2:
        st.subheader("ℹ️ System Information")
        
        # Recording status
        if st.session_state.data_recorder.is_recording:
            st.success("🟢 **Recording Active**")
            
            # Show queue size
            queue_size = st.session_state.data_recorder.data_queue.qsize()
            if queue_size > 0:
                st.info(f"📥 Queue: {queue_size} pending")
            else:
                st.success("📥 Queue: Empty")
        else:
            st.error("🔴 **Recording Inactive**")
        
        # Performance settings
        st.subheader("⚙️ Performance Settings")
        
        max_workers = st.slider(
            "Max Concurrent API Calls",
            min_value=1,
            max_value=200,
            value=st.session_state.data_recorder.max_workers,
            help="Higher values = faster data collection but more API load"
        )
        
        if max_workers != st.session_state.data_recorder.max_workers:
            st.session_state.data_recorder.max_workers = max_workers
            st.success(f"Updated to {max_workers} workers")
        
        # Auto-refresh toggle
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.toggle(
            "Auto-refresh dashboard",
            value=False,
            help="Automatically refresh data every 30 seconds"
        )
        st.session_state.auto_refresh_data = auto_refresh
        
        st.markdown("""
        **Features:**
        - ✅ Simultaneous multi-stock recording
        - ✅ Real-time performance monitoring  
        - ✅ Concurrent API calls for speed
        - ✅ Separate database thread
        - ✅ Error tracking per symbol
        - ✅ Queue-based data processing
        
        **Performance Tips:**
        - More workers = faster collection
        - Monitor queue size during heavy load
        - Check error rates for API limits
        - Use shorter intervals for active trading
        """)
        
        # Connection status
        st.subheader("🔗 Connection Status")
        
        # Test Supabase connection
        is_connected, conn_message = st.session_state.data_recorder.supabase_manager.test_connection()
        if is_connected:
            st.success(f"✅ Supabase: {conn_message}")
        else:
            st.error(f"❌ Supabase: {conn_message}")
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common Issues & Solutions:**
                
                1. **Table doesn't exist:**
                   - Click "Setup Database Table" in sidebar
                   - Run the SQL in your Supabase dashboard
                
                2. **Authentication failed:**
                   - Check your Supabase URL and API key
                   - Ensure the key has proper permissions
                
                3. **RLS (Row Level Security) issues:**
                   - Disable RLS for testing: `ALTER TABLE stock_candles DISABLE ROW LEVEL SECURITY;`
                   - Or create proper RLS policies
                
                4. **Network issues:**
                   - Check internet connection
                   - Verify Supabase service status
                """)
        
        # Test Flattrade connection (placeholder)
        st.info("🔄 Flattrade API - Ready (requires market hours for testing)")
# Add this to your existing trade_app.py for Streamlit Cloud compatibility

