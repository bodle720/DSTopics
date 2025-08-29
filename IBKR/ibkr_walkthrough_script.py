# -*- coding: utf-8 -*-
"""
Python walkthrough of IBKR tools and functionality.
"""

import time
import pytz
from datetime import datetime, timedelta

import mplfinance as mpf
import ibkr_helpers

#%% Let's get 30 day's worth of daily data for the three tickers indicated.

symbols_daily = ['TSLA', 'GOOG', 'MSFT']
daily_results, failed_tickers_daily = ibkr_helpers.get_last_n_days_ohlcv(30,
                                                                         symbols_daily,
                                                                         '1 day')
 
for symb_daily in symbols_daily:
    if symb_daily not in failed_tickers_daily:
        print(f'Daily data for symbol {symb_daily}')
        print(daily_results[symb_daily].tail(5))
        print('-'*50)
#%% Let's get 1 day's worth of minute data for the two tickers indicated.

symbols_min = ['AA', 'KO']
minute_results, failed_tickers_min = ibkr_helpers.get_last_n_days_ohlcv(1,
                                                                        symbols_min,
                                                                        '1 min')

for symb_min in symbols_min:
    if symb_min not in failed_tickers_min:
        print(f'Minute data for symbol {symb_min}')
        print(minute_results[symb_min].tail(5))
        print('-'*50)

#%% Make your own connection to the class and grab contract details, such as the exchange or industry.

symbol = 'TSLA'

# Initialize IB API
app = ibkr_helpers.IBAppHistoricalBars()
app.start() # uses default paper args.

# Wait until the connection is established
print("Waiting for connection...")

app.connected_event.wait(timeout = 10)  # Wait for connection (max 10 seconds).
    
if not app.connected_event.is_set():
    print("Failed to establish connection.")
else:  
    contract = ibkr_helpers.make_contract(symbol)
    con_req_id = app.get_next_contract_req_id()
    app.contract_reqId_to_symbol[con_req_id] = symbol
    app.reqContractDetails(con_req_id, contract) # This makes ticker_to_exchange and ticker_to_industry dicts in the app internally.
    time.sleep(5) # Give it a few seconds to retrieve the information.
    print(f'{symbol} exchange = ', app.ticker_to_exchange[symbol])
    print(f'{symbol} industry = ', app.ticker_to_industry[symbol])
    app.disconnect()

#%% Stream 5 second real time bid and ask data for the indicated ticker.

# Get the timezone for Eastern time.
est = pytz.timezone('America/New_York')

# How long do you want to stream?
observe_for_sec = 30

symbol = 'TSLA'
bid_req_id = 1
ask_req_id = bid_req_id + 1

app = ibkr_helpers.get_streamer_bid_ask_app(symbol,
                                            bid_req_id,
                                            ask_req_id)

if app:
    st_time = time.time()
    while time.time() - st_time < observe_for_sec:
        print("Current time in EST:", datetime.now(est).strftime('%Y-%m-%d %H:%M:%S'))
        last_bid = round(app.bid_df['close_bid'].iloc[-1], 2)
        last_ask = round(app.ask_df['close_ask'].iloc[-1], 2)
        spread = round(last_ask - last_bid, 2)
        spread_perc = round(100*spread/last_bid, 2)
        print(f"Last bid was ${last_bid:,}")
        print(f"Last ask was ${last_ask:,}")
        print(f"Dollar Spread is ${spread:,}")
        print(f"Ask is {spread_perc}% above the bid.")
        print('-'*50)
        time.sleep(5) # 5 second bars, so wait for a new bar to be received.
      
    app.disconnect_app_and_stream()

#%% Let's make some Heikin-Ashi candles from regular candles using the results from the daily bars above
# and plot them.

tsla_daily_bars = daily_results['TSLA']
tsla_daily_bars_ha = ibkr_helpers.calculate_heikin_ashi(tsla_daily_bars)

# Rename columns to match mplfinance expectations.
ohlc = tsla_daily_bars.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
})

ha = tsla_daily_bars_ha.rename(columns={
    'open_h': 'Open', 'high_h': 'High', 'low_h': 'Low', 'close_h': 'Close'
})

# Create the Heikin-Ashi addplot on panel 1
ha_plot = mpf.make_addplot(
    ha[['Open', 'High', 'Low', 'Close']],
    type='candle',
    panel=1
)

# Plot both panels in one call
mpf.plot(
    ohlc,
    type = 'candle',
    addplot = ha_plot,
    panel_ratios = (3, 2),
    style = 'yahoo',
    title = 'TSLA: OHLC (Top) & Heikin-Ashi (Bottom)',
    show_nontrading = False,
    volume = False
)

#%% Sometimes you will need to know when a symbol was first made publicly available for trading.
# Note the following utility does this, but won't necessarily reflect the first IPO. The true date
# for KO opening is in 1919, before digital records, s oit returns the first date we can pull records from.

ko_first_date = ibkr_helpers.get_symbols_first_date('KO')
print(f'Coca-Cola first started trading on {ko_first_date} in YYYYMMDD format.')

#%% In order to get the last n trading days starting with and including today, you can use the function below.
# Note that it includes today providing the trading day has ended.

# Retrieves the last 60 trading days. Note it skips weekends and holidays but includes early closure trading days.
last_trading_days = ibkr_helpers.get_last_n_trading_days_from_now(60)
print('The last trading days were: ', last_trading_days)

#%% If you just want a list of full closures for equity markets, run the following by indicating how far forward and
# backward you want to search from today.

# Note that not all days will be holidays. For example, 20250109 is counted as a closure as it was 
# a National Day of Mourning for President Jimmy Carter and the markets were closed.
# This assumes you set days_back far back enough to include that date.
days_back = 90
days_forward = 7
closures = ibkr_helpers.get_full_closures(days_back = days_back,
                                          days_forward = days_forward)

print('The full closure days are: ', closures)

#%% Similarly, we can grab all early closure days.
# Note this will return a dictionary mapping date to time of closure in NY time, e.g. '1300' = 1:00 PM
days_back = 90
days_forward = 7
early_closures = ibkr_helpers.get_early_market_closures(days_back = days_back,
                                                        days_forward = days_forward)

print('The early closure days (YYYYMMDD format) and closure times are: ', early_closures)

#%% Let's say you want to know precisely how many trading days have occured since a given
# date in format 'YYYYMMDD', excluding that date. Make sure your closures and early closures
# cover the date range of interest for accurate results.

# We start 10 days ago as our sample date then format it correctly.
past_date = datetime.today() - timedelta(days = 10)
past_date_str = past_date.strftime('%Y%m%d')

# We use closures and early_closures above, which will work because days_back = 90 > 70.
completed_trading_days = ibkr_helpers.count_completed_trading_days(past_date_str,
                                                                   closures,
                                                                   early_closures)

print(f'The number of completed trading days since, but not including, {past_date_str} is {completed_trading_days}.')

