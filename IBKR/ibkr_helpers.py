# -*- coding: utf-8 -*-
"""
Helpers to connect to and utilize the IBKR provided data.
"""

import threading
import pytz
import time as time_module # To not confuse it with the time arg in the realtimeBar() method below. 
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from datetime import time as dt_time
from typing import List, Dict, Tuple, Union, DefaultDict

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
      
###############################################################################
# Class definitions for historical bars and real time bid-ask data.
###############################################################################
class IBAppHistoricalBars(EWrapper, EClient):
    '''
    A client application for retrieving historical bar data from the Interactive Brokers API.

    This class extends both `EWrapper` and `EClient` to manage historical data requests,
    including minute and daily bars. It processes incoming data into pandas DataFrames
    and tracks request metadata, connection status, and error handling.

    Attributes
    ----------
    #######################################################################################
    Values you may need to initialize outside the script before calling reqHistoricalData.
    #######################################################################################
    reqId_to_requested_bars : dict
        Required: Maps request IDs to the expected number of bars for each historical data request.
    data_reqId_to_symbol : dict
        Required: Maps data request IDs to ticker symbols.
    is_request_complete : dict
        Required: Tracks whether each data request has completed.
    contract_reqId_to_symbol : dict
        Required if calling contractDetails method.: Maps contract request IDs to ticker symbols (used with contractDetails).
    reqId_to_acceptable_buffer : dict
        Optional buffer for acceptable missing bars per request.
        
    #######################################################################################
    These values are internal, so you do not need to set them in your script.
    #######################################################################################
    is_reconnecting : bool
        Indicates whether the app is attempting to reconnect.
    connected_event : threading.Event
        Event object used to track connection status.
    need_to_reestablish_connection : bool
        Flag indicating whether a broken connection needs to be reestablished.
    data : dict
        Stores retrieved ticker data.
    _next_req_id : int
        Internal counter for assigning new data request IDs.
    _next_contract_req_id : int
        Internal counter for assigning new contract request IDs.
    reqId_to_confirmed_bar_count_received : dict
        Tracks how many bars have been received for each request.
    ticker_to_exchange : dict
        Captures exchange metadata for each ticker.
    ticker_to_industry : dict
        Captures industry metadata for each ticker.
    reqId_to_errors : collections.defaultdict
        Logs errors associated with each request.
    '''
    def __init__(self):
        EClient.__init__(self, self)
        
        self.reqId_to_requested_bars: Dict[int, int] = {}
        self.data_reqId_to_symbol: Dict[int, str] = {}
        self.is_request_complete: Dict[int, bool] = {}
        self.contract_reqId_to_symbol: Dict[int, str] = {}
        self.reqId_to_acceptable_buffer: Dict[int, int] = {}

        self.is_reconnecting: bool = False
        self.connected_event: threading.Event = threading.Event()
        self.need_to_reestablish_connection: bool = False
        self.data: Dict[str, list] = {}
        self._next_req_id: int = 1000
        self._next_contract_req_id: int = 100000
        self.reqId_to_confirmed_bar_count_received: Dict[int, int] = {}
        self.ticker_to_exchange: Dict[str, str] = {}
        self.ticker_to_industry: Dict[str, str] = {}
        self.reqId_to_errors: DefaultDict[int, List[Dict[str, Union[int, str]]]] = defaultdict(list)        
        
    def get_next_req_id(self):
        self._next_req_id += 1
        return self._next_req_id
    
    def get_next_contract_req_id(self):
        self._next_contract_req_id += 1
        return self._next_contract_req_id
        
    def error(self, reqId, errorCode, errorString, *args):
                    
        # List of known non-error messages.
        ignored_messages = [
            "Market data farm connection is OK",
            "HMDS data farm connection is OK",
            "Sec-def data farm connection is OK"
            ]
    
        # Ignore only if both the error code and message match.
        if any([msg in errorString for msg in ignored_messages]):
            return  # Do nothing for these codes.
             
        if reqId in self.data_reqId_to_symbol:
            errored_symbol = self.data_reqId_to_symbol[reqId]
        else:
            errored_symbol = None
            
        if self.is_reconnecting:
            print(f"\tError while reconnecting: {errorCode} - {errorString}, reqId = {reqId}, symbol = {errored_symbol}")
            
        self.reqId_to_errors[reqId].append({'code': errorCode, 'message': errorString})
        
        cause_for_disconnect_errors = [504, 502, 501, 1100, 1102, 2119, 162, 10147]
        if errorCode in cause_for_disconnect_errors:
            # 162 can be any number of errors, but also a pacing violation.
            print(f"\tCause for initiating disconnection and reconnecting detected: {errorCode} - {errorString}, reqId = {reqId}, symbol = {errored_symbol}")
            self.need_to_reestablish_connection = True
         
    def nextValidId(self, orderId: int):
        """Callback triggered when connection is established"""
        print("\tConnected to IBKR API event for IBAppHistoricalBars.")
        self.connected_event.set()  # Mark connection as successful.
        self.need_to_reestablish_connection = False
                
    def connectionClosed(self):
        """Callback triggered when connection is closed"""
        print("\tConnection to IBKR API closed for IBAppHistoricalBars")
        self.connected_event.clear()  # Reset connection event.
        self.need_to_reestablish_connection = True

    def historicalData(self, reqId, bar):
        """Callback to store retrieved historical data"""
        symbol = self.data_reqId_to_symbol[reqId]  # Get ticker symbol.
        
        if symbol not in self.data:
            self.data[symbol] = []  # Initialize list.
            
        self.data[symbol].append([str(bar.date), bar.open, bar.high, bar.low, bar.close, int(bar.volume)])
        
        if reqId not in self.reqId_to_confirmed_bar_count_received:
            self.reqId_to_confirmed_bar_count_received[reqId] = 1
        else:
            self.reqId_to_confirmed_bar_count_received[reqId] += 1
            
    def historicalDataEnd(self, reqId, start, end):
        """Callback to signal when a historical data request is complete"""  
        self.is_request_complete[reqId] = True
        
    def get_dataframe_for_symbol(self, symbol):
        if symbol in self.data:
            return pd.DataFrame(self.data[symbol], columns=["Date", "open", "high", "low", "close", "volume"])
        return pd.DataFrame()
    
    def get_dataframes(self):
        """Convert stored data into Pandas DataFrames"""
        return {symbol: pd.DataFrame(data, columns=["Date", "open", "high", "low", "close", "volume"]) for symbol, data in self.data.items()}

    def contractDetails(self, reqId, contractDetails):
        symbol = self.contract_reqId_to_symbol[reqId]
        self.ticker_to_exchange[symbol] = str(contractDetails.contract.primaryExchange)
        self.ticker_to_industry[symbol] = str(contractDetails.industry)
        
    def is_data_transmission_complete(self, reqId):
        '''
        This method checks if data transmission is complete.
        The question is, how long do we want to wait for this to return True? The answer depends on 
        how old the data is (older = longer time to retrieve), internet connection, api throttling, server busy, etc.
        Idea: We will accept the transmission as complete providing we get approx. the right number of bars as dictated
        by the value in reqId_to_acceptable_buffer. When 0, we tolerate no missing data.
        '''
        
        # Return False if we havent even finished the request yet. Note this being True will often, but not
        # necessarily, imply the data is fully obtained.
        if not self.is_request_complete[reqId]:
            return False

        # If no bars received yet, return False.
        if reqId not in self.reqId_to_confirmed_bar_count_received:
            return False
        
        if self.reqId_to_confirmed_bar_count_received[reqId] == 0:
            return False
        
        num_bars_received = self.reqId_to_confirmed_bar_count_received[reqId]
        num_bars_required = self.reqId_to_requested_bars[reqId]
        
        if reqId in self.reqId_to_acceptable_buffer:
            acceptable_buffer = self.reqId_to_acceptable_buffer[reqId]
        else:
            acceptable_buffer = 0
            
        # I add a buffer in case it doesn't trade on a given day or is too illiquid to register on a chart etc.
        # So, if i wanted say 4 days, I'll be ok with 2 or more when buffer is set to 2.
        if num_bars_received > max(num_bars_required - (acceptable_buffer + 1), 0):
            return True
        else:
            return False
        
    def start(self, host = "127.0.0.1", port = 7497, clientId = 1):
        """
        Start connection and event loop
        Use 7496 for live accounts, 7497 for paper, make sure TWS is openor the Gateway.
        """
        self.connect(host, port, clientId)
        self._run_thread = threading.Thread(target=self.run, daemon=True)
        self._run_thread.start()
    
    def reconnect(self, host = "127.0.0.1", port = 7497, clientId = 1):
        """Reconnect to TWS"""
        print("\tClosing connection and Reconnecting to IBKR...")
        
        self.is_reconnecting = True
        self.need_to_reestablish_connection = False

        self.disconnect()
        time_module.sleep(10)  # Allow socket to close
        self.connected_event.clear()
        self.start(host, port, clientId)
        self.connected_event.wait(timeout = 30)
    
        if not self.connected_event.is_set():
            raise Exception("Reconnection failed.")
            
        print("\tProviding 30 sec breathing room for the successful reconnection...")
        time_module.sleep(30)
        print("\tBreathing room complete.")
        self.is_reconnecting = False
        
class IBAppBidAskStreamer(EWrapper, EClient):
    '''    
    A client application for streaming real-time bid and ask data from the Interactive Brokers API.

    This class subclasses `EWrapper` and `EClient` to receive real-time bar updates via the
    `realtimeBar()` method. It is configured to record bid and ask values at 5-second intervals
    and stores the data in pandas DataFrames for further analysis or visualization.
    
    Attributes
    ----------
    realtimebars_bid_reqId : int
        Request ID used to stream bid data.
    realtimebars_ask_reqId : int
        Request ID used to stream ask data.
    error_happened : bool
        Flag indicating whether an error occurred during streaming.
    connected_event : threading.Event
        Event object used to track connection status.
    bid_df : pandas.DataFrame
        DataFrame storing streamed bid data.
    ask_df : pandas.DataFrame
        DataFrame storing streamed ask data.
    '''
    
    def __init__(self,
                 realtimebars_bid_reqId: int,
                 realtimebars_ask_reqId: int) -> None:
        
        EClient.__init__(self, self)
        
        
        self.realtimebars_bid_reqId: int = realtimebars_bid_reqId
        self.realtimebars_ask_reqId: int = realtimebars_ask_reqId
    
        self.error_happened: bool = False
        self.connected_event: threading.Event = threading.Event()
    
        self.bid_df: pd.DataFrame = pd.DataFrame()
        self.ask_df: pd.DataFrame = pd.DataFrame()
        
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        """
        Handles incoming real-time bar data.
        Notes:
            1. volume, wap, count are only returned when using 'TRADES'
            2. time returned is unix time e.g. 1750281818
        """
        
        est_zone = pytz.timezone("America/New_York")  # Eastern Time (ET).
        utc_time = datetime.fromtimestamp(time, tz=timezone.utc)
        est_time = utc_time.astimezone(est_zone)
        time_est_str = est_time.strftime("%Y-%m-%d %H:%M:%S %z")
                
        if reqId == self.realtimebars_bid_reqId:
            new_data = {
                "time": pd.to_datetime(time_est_str).round('1s'),
                "open_bid": float(open_),
                "high_bid": float(high),
                "low_bid": float(low),
                "close_bid": float(close)
            }
            
            if len(self.bid_df) == 0:
                self.bid_df = pd.DataFrame([new_data], columns=["time", "open_bid", "high_bid", "low_bid", "close_bid"])
            else:
                self.bid_df = pd.concat([self.bid_df, pd.DataFrame([new_data])], ignore_index=True)   
                self.bid_df = self.bid_df.sort_values(by="time", ascending=True) # Sort by time, most recent is last.
            
        elif reqId == self.realtimebars_ask_reqId:
            new_data = {
                "time": pd.to_datetime(time_est_str).round('1s'),
                "open_ask": float(open_),
                "high_ask":float(high),
                "low_ask": float(low),
                "close_ask": float(close)
            }
            
            if len(self.ask_df) == 0:
                self.ask_df = pd.DataFrame([new_data], columns=["time", "open_ask", "high_ask", "low_ask", "close_ask"])
            else:
                self.ask_df = pd.concat([self.ask_df, pd.DataFrame([new_data])], ignore_index=True)   
                self.ask_df = self.ask_df.sort_values(by="time", ascending=True) # Sort by time, most recent is last.
            
    def error(self, reqId, errorCode, errorString, *args):
        """Handles API errors."""
        
        # List of known non-error messages
        ignored_messages = [
            "Market data farm connection is OK",
            "HMDS data farm connection is OK",
            "Sec-def data farm connection is OK"
        ]
    
        # Ignore only if both the error code and message match.
        if any([msg in errorString for msg in ignored_messages]):
            return  # Do nothing for these codes.

        print(f"\tAn error occured in IBAppBidAskStreamer: Code: {errorCode}, Message: {errorString}") 
        self.error_happened = True
            
    def nextValidId(self, orderId: int):
        """Callback triggered when connection is established"""
        print("\tConnected to IBKR API event for IBAppBidAskStreamer.")
        self.connected_event.set()  # Mark connection as successful.

    def connectionClosed(self):
        """Callback triggered when connection is closed"""
        print("\tConnection to IBKR API closed event for IBAppBidAskStreamer")
        self.connected_event.clear()  # Reset connection event.
        
    def save_stream_df(self, save_to):
        """Save the 5 second bars we have streamed so far."""
        print("\tSaving the dataframe produced from streaming 5 second bid and ask values.")
        # Merge and save them.
        merged_df = pd.merge(self.bid_df, self.ask_df, on='time', how='outer')
        
        # Add some columns: spread and percent increase of ask above bid for good measure.
        merged_df['close_spread'] = merged_df.apply(
            lambda row: row['close_ask'] - row['close_bid'] if pd.notna(row['close_ask']) and pd.notna(row['close_bid']) else None,
            axis=1
        )

        merged_df['close_spread_perc'] = merged_df.apply(
            lambda row: str(((row['close_ask'] - row['close_bid']) / row['close_bid']) * 100) + '%'
            if pd.notna(row['close_ask']) and pd.notna(row['close_bid']) else None,
            axis=1
        )        
        merged_df.to_csv(save_to)
        
    def disconnect_app_and_stream(self):
        '''
        First cancel real time bar subscriptions. Disconnection already does this but 
        this approach is considered good practice. Then, close connection.
        '''
        try:
            print('\tCancelling real time bid and ask bars for IBAppBidAskStreamer...')
            self.cancelRealTimeBars(self.realtimebars_bid_reqId)
            self.cancelRealTimeBars(self.realtimebars_ask_reqId)
        except Exception as e:
            print(f'\tFailed to cancel real time bid and ask streams for IBAppBidAskStreamer: error: {e}')
        else:
            print('\tCancelled real time bid and ask streams for IBAppBidAskStreamer.')

        try:
            if self.isConnected():
                print('\tDisconnecting app for IBAppBidAskStreamer...')
                self.disconnect()
        except Exception as e:
            print(f'\tFailed to disconnect the app for IBAppBidAskStreamer: {e}')
        else:
            print('\tApp disconnected for IBAppBidAskStreamer.')
            
###############################################################################
# Wrapper functions to call the below two classes.
###############################################################################

def get_last_n_days_ohlcv(n: int,
                          symbols: List[str],
                          bar_size: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    
    '''
    Retrieves OHLCV data for the last `n` trading days for a list of symbols.

    For each symbol, the function returns a pandas DataFrame containing OHLCV bars
    of the specified `bar_size`. Symbols that fail to return valid data are collected
    and returned as a separate list for error handling or retry logic.

    Parameters
    ----------
    n : int
        Number of trading days to retrieve.
    symbols : list of str
        List of ticker symbols to query.
    bar_size : str
        Bar size for each OHLCV entry (one of ["1 day", "1 min"]).
    
    Returns
    -------
    dict of str to pandas.DataFrame
        Dictionary mapping each symbol to its corresponding OHLCV DataFrame.
    list of str
        List of symbols for which data retrieval was unsuccessful.
    '''
    assert bar_size in ['1 day', '1 min'], 'bar size must be "1 day" or "1 min"'    
    assert type(n) == int and n > 0, 'n must be a positive integer'
    assert type(symbols) == list and len(symbols) > 0, 'symbols must be a non-empty list'
    
    symbols = [ss.upper() for ss in symbols]
    
    # Lets get a recommended buffer depending on the input. You can decide whatever is appropriate for
    # your own use case.
    if bar_size == '1 day':
        num_expected_bars_per_req = n
        if n < 256:
            suggested_buffer_per_req = 0
        else:
            suggested_buffer_per_req = 2*(n//256)
    else:
        num_expected_bars_per_req = n*390
        suggested_buffer_per_req = 0 

    app = IBAppHistoricalBars()
    app.start() 
    app.connected_event.wait(timeout = 20)
        
    if not app.connected_event.is_set():
        app.disconnect()
        raise Exception('Failed to get the last n trading days in get_last_n_days_ohlcv_bar() because could not connect to TWS. Was it running?')
        
    req_ids_for_symbols = []
    symbols_with_incomplete_data = []
    res = dict()
    for i, symbol in enumerate(symbols):  
        
        contract = make_contract(symbol)   
        req_id = app.get_next_req_id()
        app.data_reqId_to_symbol[req_id] = symbol
        app.is_request_complete[req_id] = False
        app.reqId_to_acceptable_buffer[req_id] = suggested_buffer_per_req
        app.reqId_to_requested_bars[req_id] = num_expected_bars_per_req
        
        req_ids_for_symbols.append(req_id)
        
        try:
            # Enforce max 50 requests per second
            time_module.sleep(1.1/50) 
            app.reqHistoricalData(
                reqId = req_id,
                contract = contract,
                endDateTime = "", # Up to current date.
                durationStr = f"{n} D", # Will always aim for exctly n trading days automatically, no need to worry about weekeds or closures
                barSizeSetting = bar_size, 
                whatToShow = "TRADES",
                useRTH = 1,
                formatDate = 1,
                keepUpToDate = False,
                chartOptions = [])
        except Exception as e:
            app.disconnect()
            raise Exception(f'A request for historical data for {symbol} failed with error: {e}')
            
        max_wait_time_sec = 90 if i == 0 else 30 # Tailor to what you want.
        
        st_time = time_module.time()
        complete = False
        while (time_module.time() - st_time < max_wait_time_sec):
            
            if app.is_data_transmission_complete(req_id):
                complete = True
                break
            elif len(app.reqId_to_errors[req_id]) > 0:
                break
            
            time_module.sleep(0.5)
        
        if complete:
            df = app.get_dataframe_for_symbol(symbol) 
            df['day'] = df['Date'].map(lambda row: row[:8]).astype(str)
            
            if bar_size == '1 day':
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            else:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M:%S %Z')
                
            df.set_index('Date', inplace=True)
            df = df.sort_index(ascending=True) # most recent last
    
            last_n_dates = df['day'].drop_duplicates().iloc[-n:]
            df = df[df['day'].isin(last_n_dates)]
            
            df.drop_duplicates(inplace=True)
            df = df.sort_index(ascending=True) # most recent last

            res[symbol] = df
        else:
            symbols_with_incomplete_data.append(symbol)
                  
    for req_id_to_cancel in req_ids_for_symbols:
        try:
            app.cancelHistoricalData(req_id_to_cancel) # These should not count towards pacing limits.
        except:
            pass
        
    # Disconnect
    app.disconnect()
    
    return res, symbols_with_incomplete_data

def get_streamer_bid_ask_app(symbol: str,
                             bid_req_id: int,
                             ask_req_id: int) -> Union[IBAppBidAskStreamer, bool]:
    '''    
    The function creates an instance of `IBAppBidAskStreamer`, which can be used to
    access real-time bid and ask data for the specified symbol. If initialization fails
    due to invalid parameters or connection issues, the function returns `False`.

    Parameters
    ----------
    symbol : str
        Ticker symbol for which to stream bid/ask data.
    bid_req_id : int
        Unique request ID for the bid stream.
    ask_req_id : int
        Unique request ID for the ask stream.
    
    Returns
    -------
    IBAppBidAskStreamer or bool
        An instance of `IBAppBidAskStreamer` if successful; otherwise, `False`.
    '''

    symbol = symbol.upper()
        
    def run_loop():
        app.run()

    # Initialize IB API
    app = IBAppBidAskStreamer(bid_req_id,
                              ask_req_id)
    
    app.connect("127.0.0.1", 7497, clientId = 1)  # Use 7496 for live accounts, 7497 for paper, make sure TWS is open and running.

    # Start API loop in a separate thread.
    api_thread = threading.Thread(target = run_loop, daemon=True)
    api_thread.start()

    # Wait until the connection is established.
    print("\tWaiting for connection inside get_streamer_bid_ask_app()...")

    app.connected_event.wait(timeout = 20)  # Wait for connection (max 20 seconds).
        
    if app.connected_event.is_set():
        print("\tConnection to IBKR success inside get_streamer_bid_ask_app()")
    else:
        print("\tConnection unsuccessful to IBKR inside get_streamer_bid_ask_app()")
        return False
    
    try:
        contract = make_contract(symbol)
    except Exception as e:
        print(f"\tFailed to make the contract inside get_streamer_bid_ask_app(): {e}")
        app.disconnect()
        return False

    try:
        # Request the bid stream.
        time_module.sleep(1.1/50)
        
        app.reqRealTimeBars(
                bid_req_id, 
                contract, 
                5, 
                "BID", 
                1, 
                [])
        
        time_module.sleep(1.1/50)
        
        # Request the ask stream.
        app.reqRealTimeBars(
                ask_req_id, 
                contract, 
                5, 
                "ASK", 
                1, 
                [])
        
    except Exception as e:
        print(f"\tFailed to reqRealTimeBars inside get_streamer_bid_ask_app(): {e}")
        app.disconnect()
        return False
        
    st_time = time_module.time()
    curr_time = time_module.time()
    max_wait_time = 20 # Seconds: we will not wait more than this length of time in seconds to get the first two bars.
    while not app.error_happened and len(app.bid_df) < 2 and len(app.ask_df) < 2 and curr_time - st_time <= max_wait_time:
        print("\tWaiting to get our first two bids and asks from get_streamer_bid_ask_app()...")
        time_module.sleep(1)
        curr_time = time_module.time()
    
    # At this point, we return False if time limit exceeded or an error happened.
    if app.error_happened:
        print("\tAn error occured trying to get the 5 second bids and asks, see print statements.")
        app.disconnect_app_and_stream()
        return False
    
    if curr_time - st_time > max_wait_time:
        print(f"\tExceeded max wait time of {max_wait_time} seconds. Could not get bid and ask stream. Is it after market hours?")
        app.disconnect_app_and_stream()
        return False
    
    if len(app.bid_df) >= 2 and len(app.ask_df) >= 2:
        print("\tWe obtained sufficient bid and ask data to proceed.")
        return app
    
    time_module.sleep(5) # Last chance to capture some bids and asks.
    # We must check to make sure both DataFrames have 2 or more bars.
    if len(app.bid_df) < 2 or len(app.ask_df) < 2:
        print("\tSecond attempt at getting bid and ask data failed.")
        app.disconnect_app_and_stream()
        return False
    else:
        return app
    
###############################################################################
# Miscellaneous helpers.
###############################################################################
    
def make_contract(symbol: str,
                  secType: str = 'STK',
                  exchange: str = 'SMART',
                  currency: str = 'USD') -> Contract:
    '''
    Constructs a Contract object for use with IBKR API calls.

    This function creates and returns a `Contract` instance that specifies the
    financial instrument to be queried. While default values work for most tickers,
    some symbols—particularly those listed on specific exchanges—may require an
    explicit exchange designation to avoid data retrieval errors.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol of the financial instrument (e.g., 'AAPL', 'TSLA').
    secType : str, optional
        Security type (e.g., 'STK' for stock, 'OPT' for option). Default is 'STK', which works for ETFs as well.
    exchange : str, optional
        Exchange to route the request through (e.g., 'SMART', 'NYSE'). Default is 'SMART'.
    currency : str, optional
        Currency in which the instrument is traded (e.g., 'USD', 'EUR'). Default is 'USD'.
    
    Returns
    -------
    Contract
        An IBKR-compatible `Contract` object configured with the specified parameters.
    '''
    
    symbol = symbol.upper()
    contract = Contract()
    contract.symbol = symbol
    contract.currency = currency

    # Symbols we need to set exchange and primary exchange to NYSE for.
    nyse_tickers = ['TDW',
                    'AGI',
                    'ITT',
                    'DK',
                    'IFS',
                    'ST',
                    'CTO',
                    'PFSI',
                    'HESM',
                    'KFS',
                    'IFS',
                    'LIN']

    # Define index symbols and their exchanges.
    indices = {
        'SPX': 'CBOE',
        'COMP': 'NASDAQ',
        'NDX': 'NASDAQ',
        'NYA': 'NYSE',
        'RUT': 'NYSE',
        'VIX': 'CBOE'
    }

    # ETF list with preferred exchanges (mostly ARCA).
    etfs = {
        'EIDO', 'INDA', 'QQQJ', 'SMH', 'INDY', 'MUB', 'ECH', 'SPY', 'IWM', 'XLI',
        'EWY', 'XLB', 'XLK', 'EWT', 'XLV', 'LIT', 'EWU', 'VNQ', 'XLF', 'ICLN',
        'EPU', 'EUFN', 'XLP', 'EWA', 'DIA', 'EWI', 'EWZ', 'URA', 'XHB', 'JETS',
        'KWEB', 'EWW', 'IXG', 'SOXX', 'TAN', 'EWG', 'ARGT', 'QCLN', 'XLC', 'EWP',
        'XLU', 'IXP', 'ARKK', 'XME', 'EWJ', 'XLE', 'QQQ', 'EWD', 'XBI', 'GDX', 'XLY',
        'QQQE', 'AIQ', 'KRE', 'VYM'
    }

    if symbol in indices:
        contract.secType = 'IND'
        contract.exchange = indices[symbol]
    elif symbol in etfs:
        contract.secType = 'STK'  # IBAPI treats ETFs as stocks.
        contract.exchange = 'ARCA'
    elif symbol in nyse_tickers:
        contract.secType = secType
        contract.exchange = "NYSE"
        contract.primaryExchange = "NYSE"
    else:
        contract.secType = secType
        contract.exchange = exchange

    return contract

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates Heikin-Ashi candle values from OHLC data.

    Args:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close'].

    Returns:
        pd.DataFrame: DataFrame with Heikin-Ashi columns ['open_h', 'high_h', 'low_h', 'close_h'].
    '''

    df_original_ix = df.index
    df = df.reset_index(drop=True)
    ha_df = pd.DataFrame(index=df.index)  # Create a new DataFrame for Heikin-Ashi
    
    # Calculate Heikin-Ashi close (average of open, high, low, close of regular candles)
    ha_df['close_h'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Heikin-Ashi open (average of previous Heikin-Ashi open and haikinashi close)
    ha_df['open_h'] = df['open']  # Initialize with original open values
    for i in range(1, len(df)):
        ha_df.loc[i, 'open_h'] = (ha_df.loc[i-1, 'open_h'] + ha_df.loc[i-1, 'close_h']) / 2
    
    
    # Heikin-Ashi high (max of HA open, HA close, regular high)
    ha_df['high_h'] = ha_df[['open_h', 'close_h']].join(df['high']).max(axis=1)
    
    # Heikin-Ashi low (min of HA open, HA close, regular low)
    ha_df['low_h'] = ha_df[['open_h', 'close_h']].join(df['low']).min(axis=1)

    ha_df.index = df_original_ix
    
    return ha_df

def count_completed_trading_days(last_recorded_date: str,
                                 closures: List[str],
                                 early_closures: Dict[str, str]) -> int:
    
    '''
    Calculates the number of completed trading days since a given date.

    This function computes the count of trading days that have occurred since
    `last_recorded_date`, excluding any full market closures and accounting for
    early closures. The date format should be a string in 'YYYYMMDD' format.
    Closure data should be sourced from `get_full_closures` and `get_early_market_closures`.
    
    Parameters
    ----------
    last_recorded_date : str
        Starting date in 'YYYYMMDD' format from which to begin counting.
    closures : list of str
        List of full market closure dates in 'YYYYMMDD' format.
    early_closures : dict of str to str
        Dictionary mapping dates to early closure times, such as 1300 ofr 1:00 PM Eastern.
    
    Returns
    -------
    int
        Number of completed trading days since `last_recorded_date`, excluding closures.
    '''
    # Convert input date string to date object
    last_date = datetime.strptime(last_recorded_date, "%Y%m%d").date()
    
    # Get current datetime in New York timezone
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)
    today_ny_date = now_ny.date()
    
    # If last_recorded_date is today, return 0
    if last_date >= today_ny_date:
        return 0
        
    # Start from the day after last_recorded_date
    current_date = last_date + timedelta(days=1)
    count = 0
    
    while current_date <= today_ny_date:
        date_str = current_date.strftime("%Y%m%d")
        
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Skip full closures
        if date_str in closures:
            current_date += timedelta(days=1)
            continue
        
        # Determine market close time
        close_str = early_closures.get(date_str, "1600")
        close_hour = int(close_str[:2])
        close_minute = int(close_str[2:])
        close_time = dt_time(close_hour, close_minute)
        
        # If today, check if market has closed yet
        if current_date == today_ny_date and now_ny.time() < close_time:
            break
        
        # Count as completed trading day
        count += 1
        current_date += timedelta(days=1)
    
    return count

def get_symbols_first_date(ticker: str) -> str:
    '''    
    Retrieves the earliest available date for which digital records exist for a given ticker.
    
    This function queries the data source to determine the first day that historical
    data is available for the specified symbol. The returned date is formatted as a
    string in 'YYYYMMDD' format.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol for which to retrieve the first available record date.
    
    Returns
    -------
    str
        Earliest available date for the ticker, formatted as 'YYYYMMDD'.
    '''
    ticker = ticker.upper()
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="max")  # Fetch full history
    if hist.empty:
        raise Exception(f'Ticker {ticker} has no history according to yfinance.')
        
    first_traded_date = str(hist.index.min().date()).replace('-','')
    
    return first_traded_date

def get_last_n_trading_days_from_now(n: int) -> List[str]:
    '''
    Retrieves the most recent `n` trading days up to the current date.

    This function returns a list of the last `n` trading days, including both
    full trading days and days with early market closures. Dates are returned
    as strings in 'YYYYMMDD' format.
    
    Parameters
    ----------
    n : int
        Number of recent trading days to retrieve.
    
    Returns
    -------
    list of str
        List of trading day dates in 'YYYYMMDD' format, ordered from earliest to most recent.
    '''
    # The least number of trading days in a year is 248, but we wil call it 235 to be safe.
    # This will give us an upper bound on the number of years, then 365 calendar days a year gives m,
    # ensuring we don't miss any full closure days.
    m = np.ceil(n/235)*365
    
    full_closures = get_full_closures(days_back = m, # big buffer to make sure we capture all hoildays
                                     days_forward = 1)
    
    market_tz = pytz.timezone("America/New_York")
    now = datetime.now()

    last_trading_dates = get_last_n_trading_days(now,
                                                 n,
                                                 full_closures,
                                                 market_tz)
    
    last_trading_dates = [str(i).replace('-', '') for i in last_trading_dates]
    last_trading_dates = last_trading_dates[::-1] # Now ordered from earliest to most recent.
    
    return last_trading_dates

def get_last_n_trading_days(now: datetime,
                            n: int,
                            full_closures: List[str],
                            market_tz: pytz.BaseTzInfo,
                            trading_days = None):
    '''
    Returns the last n trading days, a helper for get_last_n_trading_days_from_now above.
    Not to be called explicitly.
       
    Notes:
        1. If at time of calling it is in the middle of the trading day, it returns this day as well.
        2. the .hour attribute returns the hour: 0 for 12am midnight, 1 for 1am, ...12 for noon 
           13 for 1pm, ..., 23 for 11pm.
        3. The .weekday() method returns 0 for monday, 1 for tuesday, etc... 6 for sunday
           so a weekend will have value 5 or 6, less than 5 means it is a weekday.
    '''
        
    if trading_days is None:
        trading_days = []

    # Define market timezone (New York time)
    now = now.astimezone(market_tz)  # Ensure `now` is timezone-aware
    today = now.date()

    # If it's before market open (9:30 AM EST), adjust to previous trading day
    it_is_before_market_opens = now.hour < 9 or (now.hour == 9 and now.minute < 30)
    
    # is it during market hours
    is_during_market_hours = (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and (now.hour <= 15) 

    today_is_a_weekday = today.weekday() < 5
    today_is_closed = str(today).replace('-', '') in full_closures

    if today_is_closed or (today_is_a_weekday and it_is_before_market_opens) or (today_is_a_weekday and is_during_market_hours):
        today -= timedelta(days=1)

    # Find the most recent valid trading day
    while (today.weekday() >= 5) or (str(today).replace('-', '') in full_closures):
        today -= timedelta(days=1)

    # Add unique trading day
    if today not in trading_days:
        trading_days.append(today)
    
    # Stop when we have enough trading days
    if len(trading_days) == n:
        return trading_days

    # **Recurse using last found valid trading day**
    return get_last_n_trading_days(datetime(today.year, today.month, today.day, tzinfo=market_tz), n, full_closures, market_tz, trading_days = trading_days)

def get_full_closures(days_back: int = 365,
                      days_forward: int = 365) -> List[str]:
    '''
    This function returns a sorted list of dates (in 'YYYYMMDD' format) when the
    U.S. stock market was completely closed, based on the official holiday calendars
    of the NYSE and NASDAQ exchanges. The date range is defined by a window extending
    `days_back` into the past and `days_forward` into the future from the current date.
    
    Parameters
    ----------
    days_back : int, optional
        Number of calendar days to look back from today. Default is 365.
    days_forward : int, optional
        Number of calendar days to look forward from today. Default is 365.
    
    Returns
    -------
    list of str
        Sorted list of dates in 'YYYYMMDD' format when the market was fully closed.
    '''
    start_date = (datetime.now() - timedelta(days = days_back)).date()
    end_date = (datetime.now() + timedelta(days = days_forward)).date()
    nyse = mcal.get_calendar('NYSE')
    nasdaq = mcal.get_calendar('NASDAQ')
    nyse_closures = nyse.holidays().holidays
    nasdaq_closures = nasdaq.holidays().holidays
    nyse_filtered = [str(dt).replace('-', '') for dt in nyse_closures if start_date <= dt <= end_date]
    nasdaq_filtered = [str(dt).replace('-', '') for dt in nasdaq_closures if start_date <= dt <= end_date]
    closures = list(sorted(set(nyse_filtered + nasdaq_filtered)))
    
    return closures

def get_early_market_closures(days_back: int = 365,
                              days_forward: int = 365) -> Dict[str, str]:
    '''
    Retrieves early market closure times for NYSE and NASDAQ within a specified date range.

    This function returns a dictionary mapping dates (formatted as 'YYYYMMDD') to early
    market closing times in Eastern Time. A date is considered an early closure if the
    market closes before 4:00 PM ET. The date range is defined by a window extending
    `days_back` into the past and `days_forward` into the future from the current date.
    
    Parameters
    ----------
    days_back : int, optional
        Number of calendar days to look back from today. Default is 365.
    days_forward : int, optional
        Number of calendar days to look forward from today. Default is 365.
    
    Returns
    -------
    dict of str to str
        Dictionary where keys are dates in 'YYYYMMDD' format and values are early
        closing times in 'HHMM' 24-hour format (Eastern Time). For example,
        {'20250703': '1300'} indicates a 1:00 PM ET close on July 3, 2025.
    '''
    eastern = pytz.timezone("America/New_York")
    
    start_date = (datetime.now() - timedelta(days=days_back)).date()
    end_date = (datetime.now() + timedelta(days=days_forward)).date()

    nyse = mcal.get_calendar('NYSE')
    nasdaq = mcal.get_calendar('NASDAQ')

    # Get early closures from schedule
    nyse_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    nasdaq_schedule = nasdaq.schedule(start_date=start_date, end_date=end_date)

    def extract_early_closures(schedule):
        early = {}
        for dt, row in schedule.iterrows():
            close_utc = row['market_close']
            close_et = close_utc.astimezone(eastern)
            if close_et.hour < 16:
                key = dt.strftime('%Y%m%d')
                hhmm = f"{close_et.hour:02d}{close_et.minute:02d}"
                early[key] = hhmm
        return early

    early_closures = extract_early_closures(nyse_schedule)
    early_closures.update(extract_early_closures(nasdaq_schedule))

    # Sort by date
    return dict(sorted(early_closures.items()))