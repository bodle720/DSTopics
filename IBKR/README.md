# Interactive Brokers API Walkthrough

This folder contains a practical walkthrough for working with the Interactive Brokers API from Python.

The focus is not trading strategy or financial advice. The purpose is to demonstrate how to connect to a real-world external API, request historical and streaming market data, handle asynchronous callbacks, organize returned data into usable `pandas` structures, and build helper utilities around messy time-series data.

## Main Notebook

- [`ibkr_walkthrough.ipynb`](ibkr_walkthrough.ipynb)

## What This Demonstrates

- Working with a real external API from Python
- Using the lower-level Interactive Brokers API rather than only a high-level wrapper
- Handling asynchronous callback-based data delivery
- Running the API client in a separate thread so the notebook/script can continue execution
- Requesting historical OHLCV market bars
- Requesting minute-level and daily market data
- Streaming bid/ask data during market hours
- Retrieving contract metadata such as exchange and industry
- Converting returned API data into `pandas` DataFrames
- Building helper utilities for market calendars, trading days, full closures, and early closures
- Calculating and plotting Heikin-Ashi candles
- Structuring notebook examples around reusable helper functions

## Project Framing

This project uses Interactive Brokers because it provides a realistic example of programmatic data access through a production-style API.

The financial context is incidental. The main value of the project is the engineering work: connecting to an external service, managing asynchronous responses, cleaning returned data, handling market-calendar edge cases, and presenting the results in a clear notebook walkthrough.

This is not intended to be a trading bot, automated trading system, investment recommendation, or profitable strategy. It is an API/data-handling demonstration.

## Files

| File | Purpose |
|---|---|
| `ibkr_walkthrough.ipynb` | Main notebook walkthrough showing API usage and helper utilities |
| `ibkr_helpers.py` | Helper functions and API client classes used by the notebook |
| `ibkr_walkthrough_script.py` | Script version of the notebook workflow, if running outside Jupyter |
| `reg_and_ha_tsla.png` | Example visualization comparing regular OHLC candles with Heikin-Ashi candles |

## Notebook Highlights

The walkthrough covers several common tasks when working with market data APIs:

- Fetching recent daily OHLCV bars for multiple symbols
- Fetching minute-level OHLCV bars
- Looking up contract details such as exchange and industry
- Streaming bid/ask data in near real time
- Calculating bid/ask spread information
- Creating Heikin-Ashi candle data from regular OHLC candles
- Determining the earliest available historical data date for a symbol
- Finding recent completed trading days
- Identifying full market closure days
- Identifying early market closure days
- Counting completed trading days since a specified date

## Requirements

To run the notebook against the live API, you generally need:

- An Interactive Brokers account
- Trader Workstation or IB Gateway running locally
- API access enabled in the local IBKR application
- Any required market-data subscriptions for the symbols and data types requested
- A Python environment with the required packages installed

The notebook is written as a walkthrough, so some cells depend on an active IBKR connection and may only work during market hours, especially the streaming bid/ask examples.

## Notes

The notebook mentions `ib_insync`, a popular higher-level wrapper around the Interactive Brokers API. This project intentionally works closer to the lower-level API to show how callback-driven data access works under the hood.

For many practical projects, `ib_insync` may be the simpler choice. This folder is useful because it exposes the lower-level mechanics and provides reusable utilities for working with market-data time series.