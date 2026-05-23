# -*- coding: utf-8 -*-
"""
A script of auxiliary functions acting as a library.
"""

# Imports
import multiprocessing

from typing import Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function will calculate the Heikin-Ashi candle values and return them
    in the form of a pandas.DataFrame, with columns open_h, high_h, low_h, and
    close_h.
    
    Arguments
    ----------
    :param df: pandas.DataFrame with columns: open, high, low, close (case insensitive).
    :type df: pandas.DataFrame
    
    Returns
    ----------
    :return: pandas.DataFrame with Hekin-Ashi candle values.
    :rtype: pandas.DataFrame
    '''
    df.columns = df.columns.str.lower()

    df_original_ix = df.index
    df = df.reset_index(drop=True)
    ha_df = pd.DataFrame(index=df.index)  # Create a new DataFrame for Heikin-Ashi
    
    # Calculate Heikin-Ashi close (average of open, high, low, close)
    ha_df['close_h'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Heikin-Ashi open (average of previous Heikin-Ashi open and close)
    ha_df['open_h'] = df['open']  # Initialize with original open values
    for i in range(1, len(df)):
        ha_df.loc[i, 'open_h'] = (ha_df.loc[i-1, 'open_h'] + ha_df.loc[i-1, 'close_h']) / 2
    
    # Calculate Heikin-Ashi high (max of high, open, close)
    ha_df['high_h'] = df[['high', 'open', 'close']].max(axis=1)
    
    # Calculate Heikin-Ashi low (min of low, open, close)
    ha_df['low_h'] = df[['low', 'open', 'close']].min(axis=1)
    ha_df.index = df_original_ix
    
    return ha_df

def apply_DMD(X: np.ndarray,
              X_prime: np.ndarray,
              approach: str = 'iterative',
              forward_steps: int = 3,
              perc_cumul_var: float = 0.85) -> np.ndarray:
    
    '''
    Runs the DMD algorithm and chooses r (rank) based on the quantity of cumulative
    explained variance in the eigenvalues. X and X_prime contain snapshots of data in time,
    organized as columns (each column represents one snapshot).
    
    If forward_steps = 1, both approaches are identical (iterative and power).
    
    Arguments
    ----------
    :param X: An array with columns at timestamps 1 to n, inclusive.
    :param X_prime: An array with columns at timestamps 2 to n + 1, inclusive.
    :param approach: If iterative, then each new update is informed by the state (possibly predicted state) before it.
                     Otherwise, approach must be power, and the forecasts will only evolve from the last observed true state.
    :param forward_steps: How many steps forward to predict using the DMD algorithm.
    :param perc_cumul_var: The minimum amount of explained variance captured to determine r in the DMD algorithm.
    
    Returns
    ----------
    :return: An array of the future states.
    :rtype: np.ndarray
    '''
    approach = approach.lower()
    
    assert approach in ['iterative', 'power'], 'Approach for DMD future prediction must be one of iterative or power'
    assert type(forward_steps) == int, 'forward_steps must be an integer greater than 0'
    assert forward_steps > 0, 'forward_steps must be an integer greater than 0'
    
    # Singular Value Decomposition (SVD).
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    
    # Determine the ideal rank.
    sing_vals_squared = Sigma ** 2
    total_var = np.sum(sing_vals_squared)
    
    cumulative_exp_var = 0
    rank = None
    for i, sq_sing_val in enumerate(sing_vals_squared):
        exp_var = sq_sing_val/total_var
        cumulative_exp_var += exp_var
        if (i == 0) and (cumulative_exp_var >= perc_cumul_var):
            rank = 2
            break
        elif cumulative_exp_var >= perc_cumul_var:
            rank = i + 1
            break
    
    assert not (rank is None), 'Issue calculating r in DMD'
    
    U = U[:, :rank]
    Sigma = Sigma[:rank]
    VT = VT[:rank, :]

    # Low-rank approximation of A_tilde = U* Sigma * VT.
    A_tilde = U.T @ X_prime @ VT.T @ np.linalg.inv(np.diag(Sigma))

    # Eigenvalue decomposition of A_tilde.
    eigenvalues, W = np.linalg.eig(A_tilde) #eigenvalues length r
    dmd_modes_PHI = X_prime @ VT.T @ np.linalg.inv(np.diag(Sigma)) @ W # shape n by r
    
    # Predict future states k steps into the future.
    future_states = []

    b = np.linalg.pinv(dmd_modes_PHI) @ X_prime[:,-1] # A vector of size r, b in the DMD formula.

    for k in range(forward_steps):
        if approach == 'iterative':
            if k == 0:
                next_state = dmd_modes_PHI @ (np.diag(eigenvalues) @ b)
                future_states.append(next_state)
            else:
                next_state = dmd_modes_PHI @ (np.diag(eigenvalues) @ np.linalg.pinv(dmd_modes_PHI) @ future_states[-1])
                future_states.append(next_state)
        else:
            # In this approach, b remains constant and we start at the last timestep and
            # allow the system to evolve without iterative updates.
            p = k + 1
            next_state = dmd_modes_PHI @ (np.diag(eigenvalues**p) @ b)
            future_states.append(next_state)
            
    predictions = np.array(future_states).real.T
    
    return predictions

def get_features_one_day(args) -> tuple:
    '''
    Extracts the features for a single day for use in a model (one sample). Uses an args input because 
    later we will use multiprocessing, and imap requires single arguments to worker functions. We could use
    starmap, but that will be less efficient memory-wise. Both return results in the same order as the input.

    args = (df, date, look_back, forecast_steps, dmd_approach, dmd_perc_cumul_var, timeframes, macd_fast, macd_slow, macd_signal)
    
    df: DataFrame containing the date of interest and at least enough bars in the past to calculate 
        the required indicators.
            
    forecast_steps: This is provided for use in other utility functions. For the purpose 
                    of feature generation for model training, only the first forecast is used
                    and is present in the first returned value, 'feature_dict'. If forecast_steps
                    is greater than 1, the returned 'feature_dict' will not change, but the second
                    returned value 'forecasts' will contain 'forecast_steps' of forecasts. The 
                    first will be the same as the returned features. If using this function
                    for forecasting and utilizing 'forecasts', then the value of 'dmd_approach'
                    will be relevant.
    
    Recommended defaults:
        forecast_steps = 1,
        dmd_approach = 'iterative',
        dmd_perc_cumul_var = 0.85,
        timeframes = [1, 2, 3, 5],
        macd_fast = 12,
        macd_slow = 26,
        macd_signal = 9
       
    Arguments
    ----------
    :param df: DataFrame containing the date of interest and at least enough bars in the past to calculate the required indicators.
    :type df: pandas.DataFrame
    :param date: String date in format 'YYYY-MM-DD', e.g. "2025-03-02"
    :type date: str
    :param look_back: Indicates how many bars (same for each timeframe) to use from the past up to current day for DMD prediction.
    :type look_back: int
    :param forecast_steps: The number of steps forward to forecast. See above explanation.
    :type forecast_steps: int
    :param dmd_approach: Set to either iterative or power; it indicates how to produce forecasts.
    :type dmd_approach: str
    :param dmd_perc_cumul_var: Used in the DMD algrithm, see apply_DMD documentation.
    :type dmd_perc_cumul_var: float
    :param timeframes: A list of integers in units of days (1 or higher) indicating the timeframes to include in the model.
    :type timeframes: list
    :param macd_fast: The period for the fast moving average in the MACD calculation. Must be less than macd_slow.
    :type macd_fast: int
    :param macd_slow: The period for the slow moving average in the MACD calculation. Must be greater than macd_fast.
    :type macd_slow: int
    :param macd_signal: The period for the signal moving average in the MACD calculation.
    :type macd_signal: int
    
    Returns
    ----------
    :return: A tuple containing:
            - feature_dict: a dictionary mapping featurename to value for use in ML models.
            - dmd_predictions: an np.ndarray containing the DMD forecasts.
            - dmd_feat_names: A list of feature names, with 'dmd_' preceeding features from the DMD forecasts.
    :rtype: tuple
    '''

    df, date, look_back, forecast_steps, dmd_approach, dmd_perc_cumul_var, timeframes, macd_fast, macd_slow, macd_signal = args
    
    # Quick checks.
    assert date in df.index, 'Invalid date, pick a trading day.'
    assert look_back > 1
    assert macd_slow > macd_fast, 'MACD slow period must be greater than the fast period.'
    
    # Get the Heikin-Ashi candles.
    df_ha = calculate_heikin_ashi(df)
    df_with_ha = pd.concat([df, df_ha], axis=1)
    
    # Now we must chop off the dates after 'date' that we don't need, but also
    # be sure to have enough data in the past to calculate the metrics for every timeframe.
    target_date = pd.Timestamp(date)
    target_index = df_with_ha.index.get_loc(target_date)

    # How many bars are needed to look back?
    num_required_past_bars = (look_back + 1 + macd_slow + macd_signal)*max(timeframes)
    start_index = target_index - num_required_past_bars 
    assert start_index >= 0, f'Choose a date not so near to the beginning of the dataset: {np.abs(start_index)} days too early.'
    
    sliced_df = df_with_ha.iloc[start_index:(target_index + 1)].copy()

    feature_dict = dict()
    dmd_dfs = []
    for tf in timeframes:
        # Get the appropriate timeframe data. We require the most recent day ('date') be aggregated
        # with the bar(s) before it for each timeframe. So we reverse the dataframe.
        tf_slice = sliced_df.copy()
        
        rows_to_trim = len(tf_slice) % tf
        tf_slice = tf_slice.iloc[rows_to_trim:].copy()
        
        # Assign each their groups, frst the earlier dates.
        group_labels = tf_slice.index.to_series().reset_index(drop=True).index // tf
        tf_slice['group'] = group_labels
             
        tf_df = tf_slice.iloc[::-1].groupby('group').agg({'open_h': 'last',   
                                                                'high_h': 'max',  
                                                                'low_h': 'min',     
                                                                'close_h': 'first',
                                                                'open': 'last',   
                                                                'high': 'max',  
                                                                'low': 'min',     
                                                                'close': 'first'})
        
    
        tf_df['ha_ret'] = (tf_df['close_h'] - tf_df['close_h'].shift(1)) / tf_df['close_h'].shift(1).abs()
        tf_df['price_ret'] = (tf_df['close'] - tf_df['close'].shift(1)) / tf_df['close'].shift(1).abs()

        
        tf_df['rsi_h'] = ta.rsi(tf_df['close_h'], length = 14)
        tf_df['rsi_h_sma'] = ta.sma(tf_df['rsi_h'], length = 14) 
        tf_df['rsi_h_sma_ret'] = (tf_df['rsi_h_sma'] - tf_df['rsi_h_sma'].shift(1)) / tf_df['rsi_h_sma'].shift(1).abs()

        current_rsi = tf_df['rsi_h'].iloc[-1]
        current_ha_ret = tf_df['ha_ret'].iloc[-1]
        current_price_ret = tf_df['price_ret'].iloc[-1]
        current_rsi_sma_ret = tf_df['rsi_h_sma_ret'].iloc[-1]

        # MACD on HA candles, and use histogram returns
        macd_hist_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
        tf_df['macd_h_hist'] = ta.macd(tf_df['close_h'],
                                       fast = macd_fast,
                                       slow = macd_slow,
                                       signal = macd_signal)[macd_hist_col]
        
        tf_df['macd_h_hist_ret'] = (tf_df['macd_h_hist'] - tf_df['macd_h_hist'].shift(1)) / tf_df['macd_h_hist'].shift(1).abs()

        current_macd_h_hist_ret = tf_df['macd_h_hist_ret'].iloc[-1]

        # MACD on RSI, and use histogram returns 
        tf_df['macd_rsi_hist'] = ta.macd(tf_df['rsi_h'],
                                       fast = macd_fast,
                                       slow = macd_slow,
                                       signal = macd_signal)[macd_hist_col]
        
        tf_df['macd_rsi_hist_ret'] = (tf_df['macd_rsi_hist'] - tf_df['macd_rsi_hist'].shift(1)) / tf_df['macd_rsi_hist'].shift(1).abs()

        current_macd_rsi_hist_ret = tf_df['macd_rsi_hist_ret'].iloc[-1]

        # Get the supertrend as another feature
        supertrend_dir_col = "SUPERTd_10_0.9"
        tf_df['supertrend_ha'] = ta.supertrend(high = tf_df['high_h'],
                                               low = tf_df['low_h'],
                                               close = tf_df['close_h'],
                                               length = 10,
                                               multiplier = 0.9)[supertrend_dir_col]
        
        current_supertrend_direction = tf_df['supertrend_ha'].iloc[-1]

        feature_dict[f'c_macd_hist_ret_ha_{tf}d'] = current_macd_h_hist_ret
        feature_dict[f'c_macd_hist_ret_ha_rsi_{tf}d'] = current_macd_rsi_hist_ret
        feature_dict[f'c_ha_ret_{tf}d'] = current_ha_ret
        feature_dict[f'c_price_ret_{tf}d'] = current_price_ret
        feature_dict[f'c_rsi_sma_ret_{tf}d'] = current_rsi_sma_ret
        feature_dict[f'c_rsi_ha_{tf}d'] = current_rsi/100 # RSI is 0 - 100
        feature_dict[f'c_supertrend_{tf}d'] = current_supertrend_direction # -1 means down, +1 means up

        # Now we must gather the data points required for DMD for this particular timeframe
        # using the look back variable 'look_back'. We use 4 features for DMD:
            # 1. price_ret
            # 2. ha_ret
            # 3. macd_h_hist_ret
            # 4. macd_rsi_hist_ret
        # We then form this sinto an array and stack across timeframes, then perform DMD and append
        # the final features.
        
        # Capture the current day plus look_bak bars in the past
        dmd_subset = tf_df[['price_ret', 'ha_ret', 'macd_h_hist_ret', 'macd_rsi_hist_ret']].tail(look_back + 1)
        dmd_subset.columns = [col + f'_{tf}d' for col in dmd_subset.columns]
        dmd_subset = dmd_subset.reset_index(drop=True)
        dmd_subset_t = dmd_subset.T.reset_index()
        dmd_subset_t = dmd_subset_t.rename(columns={'index': 'feature_name'})
        
        # Most recent data is far right.
        dmd_dfs.append(dmd_subset_t)

    dmd_df = pd.concat(dmd_dfs, ignore_index = True)
    dmd_feat_names = list(dmd_df['feature_name'].copy())
    dmd_df = dmd_df.drop(columns=['feature_name'])
    dmd_arr = dmd_df.to_numpy() # Most recent data is on the right.

    X = dmd_arr[:,:-1]
    X_prime = dmd_arr[:,1:]

    dmd_predictions = apply_DMD(X,
                                X_prime,
                                approach = dmd_approach, # irrelevant for one forward step.
                                forward_steps = forecast_steps,
                                perc_cumul_var = dmd_perc_cumul_var)
    
    # Now add the DMD feature predictions to the feature dict.
    for i in range(len(dmd_predictions)):
        f_name = dmd_feat_names[i]
        feature_dict[f'dmd_{f_name}'] = dmd_predictions[i, 0]
        
    return feature_dict, dmd_predictions, dmd_feat_names

def plot_time_series(ax, title: str, values: list, forecasts: list) -> None:
    '''
    A helper function to plot timeseries.
    
    Arguments
    ----------
    :param ax: matplotlib.axes._axes.Axes object on which to plot.
    :type ax: matplotlib.axes._axes.Axes
    :param title: A title for the plot.
    :type title: str
    :param values:The true values to plot.
    :type values: list
    :param forecasts: The forecasted values to plot next to the true values.
    :type forecasts: list
    
    Returns
    ----------
    :rtype: None
    '''
    
    time = list(range(1, len(values) + 1))
    ax.plot(time, values, label="Actual Values", marker="o")
    forecast_time = time[-1*(len(forecasts)):]  # Time points for the forecasts
    ax.plot(forecast_time, forecasts, label="Forecasts", marker="x", linestyle="--", color="red")
    
    ax.axvline(x = time[-1*(len(forecasts) + 1)], color='green', linestyle='--', label='Now')                
    ax.axhline(y = 0, color='red', linestyle='--', linewidth = 3)                

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid()
    
def plot_DMD_forecasts(df: pd.DataFrame,
                        date: str,
                        look_back: int,
                        forecast_steps: int,
                        view_tf: int,
                        dmd_approach: str = 'iterative',
                        dmd_perc_cumul_var: float = 0.85,
                        timeframes: list = [1, 2, 3, 5],
                        macd_fast: int = 12,
                        macd_slow: int = 26,
                        macd_signal: int = 9,
                        save_to: Optional[str] = None,
                        figsize: tuple = (9,6)) -> pd.DataFrame:
    '''
    Plots MACD histogram truth and predictions for RSI and Heikin-Ashi closing prices.
    
    
    Arguments
    ----------
    :param df: DataFrame containing the date of interest and at least enough bars in the past to calculate the required indicators.
    :type df: pandas.DataFrame
    :param date: String date in format 'YYYY-MM-DD', e.g. "2025-03-02"
    :type date: str
    :param look_back: Indicates how many bars (same for each timeframe) to use from the past up to current day for DMD prediction.
    :type look_back: int
    :param forecast_steps: The number of steps forward to forecast. See above explanation.
    :type forecast_steps: int
    :param view_tf: An integer. It must belong to the list 'timeframes'. Dictates which timeframe is plotted.
    :type view_tf: int
    :param dmd_approach: Set to either iterative or power; it indicates how to produce forecasts.
    :type dmd_approach: str
    :param dmd_perc_cumul_var: Used in the DMD algrithm, see apply_DMD documentation.
    :type dmd_perc_cumul_var: float
    :param timeframes: A list of integers in units of days (1 or higher) indicating the timeframes to include in the model.
    :type timeframes: list
    :param macd_fast: The period for the fast moving average in the MACD calculation. Must be less than macd_slow.
    :type macd_fast: int
    :param macd_slow: The period for the slow moving average in the MACD calculation. Must be greater than macd_fast.
    :type macd_slow: int
    :param macd_signal: The period for the signal moving average in the MACD calculation.
    :type macd_signal: int
    :param save_to: A full path to save the plot to rather than show it. Optional with adefault of None.
    :type save_to: str
    :param figsize: A tuple to indicate the size of the plotted figure. Order: (width, height)
    :type figsize: tuple
    
    Returns
    ----------
    :return: pandas.DataFrame with forecasts.
    :rtype: pandas.DataFrame
    '''
    
    # Quick checks.
    assert view_tf in timeframes
    assert date in df.index, 'Invalid date, pick a trading day.'
    assert look_back > 1
    assert forecast_steps > 0
    assert macd_slow > macd_fast, 'MACD slow period must be greater than the fast period.'

    # First, we calculate theDMD forecasts in a manner consistent with how the model was trained.
    args = [df.copy(),
            date,
            look_back,
            forecast_steps,
            dmd_approach,
            dmd_perc_cumul_var,
            timeframes,
            macd_fast,
            macd_slow,
            macd_signal]
    
    _, dmd_predictions, dmd_feat_names = get_features_one_day(args)
    
    # Make a dataframe with the requisite features and timeframe indicated by view_tf.
    forecast_df = pd.DataFrame(data = dmd_predictions).T
    forecast_df.columns = dmd_feat_names
    forecast_df = forecast_df.loc[:, forecast_df.columns.map(lambda x: f'_{view_tf}d' in x)].copy()    
    
    # Now we must calculate the past observations leading up to 'date' and into the future by 'forecast_steps' units.
    
    # Get the Heikin-Ashi candles.
    df_ha = calculate_heikin_ashi(df)
    df_with_ha = pd.concat([df, df_ha], axis=1)
    
    # Now we must chop off the dates after 'date' that we don't need, but also
    # be sure to have enough data in the past to calculate the metrics for every timeframe.
    target_date = pd.Timestamp(date)
    target_index = df_with_ha.index.get_loc(target_date)

    # How many bars are needed to look back?
    num_required_past_bars = (look_back + 1 + macd_slow + macd_signal)*view_tf
    start_index = target_index - num_required_past_bars 
    assert start_index >= 0, f'Choose a date not so near to the beginning of the dataset: {np.abs(start_index)} days too early.'
    
    # Make sure there is enough data to compare truth and forecasts.
    num_required_future_bars = view_tf*forecast_steps
    end_index = target_index + num_required_future_bars + 1
    assert end_index <= len(df), 'Choose a date not so close to the end of the data, or a smaller number of forecast steps'
    
    # To switch timeframes, we need to only utilize data up to present day ('date')
    sliced_df_all = df_with_ha.iloc[start_index:end_index].copy()
    
    # In case of uneven group counts, we ensure the dataframe length in a multiple of view_tf
    rows_to_trim = len(sliced_df_all) % view_tf
    sliced_df_all = sliced_df_all.iloc[rows_to_trim:].copy()
        
    # Assign each their groups, frst the earlier dates.
    group_labels = sliced_df_all.index.to_series().reset_index(drop=True).index // view_tf
    sliced_df_all['group'] = group_labels
    
    # Sanity check, it should always bethe case that the date after the target date 
    # starts a new group.
    new_target_index = sliced_df_all.index.get_loc(target_date)
    assert group_labels[new_target_index] + 1 == group_labels[new_target_index + 1], 'Something went wrong grouping the timeframes.'
         
    # The last 'forecast_steps' of these rows will be 'in the future' from our target date.
    tf_df = sliced_df_all.iloc[::-1].groupby('group').agg({'open_h': 'last',   
                                                            'high_h': 'max',  
                                                            'low_h': 'min',     
                                                            'close_h': 'first',
                                                            'open': 'last',   
                                                            'high': 'max',  
                                                            'low': 'min',     
                                                            'close': 'first'})
 
    
    # Now calculate truth values to compare to forecasts.
    tf_df['ha_ret'] = (tf_df['close_h'] - tf_df['close_h'].shift(1)) / tf_df['close_h'].shift(1).abs()
    tf_df['price_ret'] = (tf_df['close'] - tf_df['close'].shift(1)) / tf_df['close'].shift(1).abs()
    tf_df['rsi_h'] = ta.rsi(tf_df['close_h'], length = 14)
    
    macd_hist_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
    tf_df['macd_h_hist'] = ta.macd(tf_df['close_h'],
                                   fast = macd_fast,
                                   slow = macd_slow,
                                   signal = macd_signal)[macd_hist_col]
    
    tf_df['macd_h_hist_ret'] = (tf_df['macd_h_hist'] - tf_df['macd_h_hist'].shift(1)) / tf_df['macd_h_hist'].shift(1).abs()

    tf_df['macd_rsi_hist'] = ta.macd(tf_df['rsi_h'],
                                   fast = macd_fast,
                                   slow = macd_slow,
                                   signal = macd_signal)[macd_hist_col]
    
    tf_df['macd_rsi_hist_ret'] = (tf_df['macd_rsi_hist'] - tf_df['macd_rsi_hist'].shift(1)) / tf_df['macd_rsi_hist'].shift(1).abs()

    # Trim tf_df to correct look_nack distance.
    tf_df_trim = tf_df.iloc[-1*(forecast_steps + 1 + look_back):].copy()
    has_nan = tf_df_trim.isna().any().any()
    assert not has_nan, 'Error with NaNs in trimmed df for plotting.'
    
    # Retrieve and plot them.

    price_ret_values = tf_df_trim['price_ret'].to_list()
    ha_ret_values = tf_df_trim['ha_ret'].to_list()
    macd_h_hist_ret_values = tf_df_trim['macd_h_hist_ret'].to_list()
    macd_rsi_hist_ret_values = tf_df_trim['macd_rsi_hist_ret'].to_list()
    
    price_ret_forecasts = forecast_df[f'price_ret_{view_tf}d'].to_list()
    ha_ret_forecasts = forecast_df[f'ha_ret_{view_tf}d'].to_list()
    macd_h_hist_ret_forecasts = forecast_df[f'macd_h_hist_ret_{view_tf}d'].to_list()
    macd_rsi_hist_ret_forecasts = forecast_df[f'macd_rsi_hist_ret_{view_tf}d'].to_list()
    
    fig, axes = plt.subplots(2, 2, figsize = figsize)  # 2 rows, 2 columns

    plot_time_series(axes[0, 0], "Price Returns", price_ret_values, price_ret_forecasts)
    plot_time_series(axes[0, 1], "Heikin-Ashi Close Returns", ha_ret_values, ha_ret_forecasts)
    plot_time_series(axes[1, 0], "MACD on Heikin-Ashi Hist. Returns", macd_h_hist_ret_values, macd_h_hist_ret_forecasts)
    plot_time_series(axes[1, 1], "MACD on RSI Hist. Returns", macd_rsi_hist_ret_values, macd_rsi_hist_ret_forecasts)
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Show the plot
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)
    
    return forecast_df

def generate_data_and_labels(df,
                             look_back,
                             dmd_perc_cumul_var,
                             timeframes,
                             macd_fast,
                             macd_slow,
                             macd_signal,
                             label_freq,
                             frac_cpu_to_use) -> tuple:
    '''
    Generate unscaled data for model input. Utilizes multiprocessing, so must becalled within a if __name__ == '__main__' block.
    
    Arguments
    ----------
    :param df: A DataFrame containing open, high, low, close data from which to generate features and labels. Each day
                possible is used and generates a set of features and corresponding labels.
    :type df: pd.DataFrame
    :param look_back: Indicates how many bars (same for each timeframe) to use from the past up to current day for DMD prediction.
    :type look_back: int
    :param dmd_perc_cumul_var: Used in the DMD algrithm, see apply_DMD documentation.
    :type dmd_perc_cumul_var: float
    :param timeframes: A list of integers in units of days (1 or higher) indicating the timeframes to include in the model.
    :type timeframes: list
    :param macd_fast: The period for the fast moving average in the MACD calculation. Must be less than macd_slow.
    :type macd_fast: int
    :param macd_slow: The period for the slow moving average in the MACD calculation. Must be greater than macd_fast.
    :type macd_slow: int
    :param macd_signal: The period for the signal moving average in the MACD calculation.
    :type macd_signal: int
    :param label_freq: How many days into the furture are used to determine if the price decreased,
                        increased, or remains still. E.g. 5 means compare the open and close of the 
                        bar formed by the next 5 days after the target date.
    :type label_freq: int
    :param frac_cpu_to_use: A float greater than 0 and less than or equal to 1. Indicates how much CPU utilization
                            for parallel processing during data generation.
    :type frac_cpu_to_use: float
    
    Returns
    ----------
    :return: A tuple containing:
            - feat_arr: A numpy array of features.
            - labels_arr: A numpy array of labels.
            - dates: A list of string dates if required.
            - feat_df: A dataframe version of the features if required.
    :rtype: tuple
    '''
    
    forecast_steps = 1
    dmd_approach = 'iterative'
    
    assert 0 < frac_cpu_to_use <= 1, 'Invalid fraction of CPU cores to use, try 0.5, 0.8, 1, 0.2, etc...'
    
    start_ix = (look_back + 1 + macd_slow + macd_signal)*max(timeframes)
    end_ix = len(df) - label_freq - 1
    
    # Given a day, how many historical bars do we need to calculatethe indicators?
    num_required_past_bars = (look_back + 1 + macd_slow + macd_signal)*max(timeframes)

    # Store the labels for the samples.
    labels = []
    
    # Store a list of tuples, each element is input into 'get_features_one_day'.
    inputs_ls = [] 
    
    # Store the dates used.
    dates = []
    
    # Calculate the number of data points if needed.
    # num_pts = end_ix - start_ix + 1
        
    # Collect the inputs for multiprocessing.
    for ix, target_date in enumerate(list(df.index.copy())):
        if (ix >= start_ix) and (ix <= end_ix):
            date = str(target_date).split(' ')[0]
            dates.append(date)

            # First, get the label.
            # target_date = pd.Timestamp(date)
            target_index = df.index.get_loc(target_date)
            label_df = df.iloc[(target_index + 1):(target_index + 1 + label_freq), :].copy()
            
            bar_ret = (label_df['Close'].iloc[-1] - label_df['Open'].iloc[0])/label_df['Open'].iloc[0]
            
            if bar_ret <= 0:
                label = 0
            else:
                label = 1
                
            labels.append(label)
            
            # Now get the subset dataframe required for feature calculation in 'get_features_one_day'
            start_index = target_index - num_required_past_bars 
            assert start_index >= 0, f'Choose a date not so near to the beginning of the dataset: {np.abs(start_index)} days too early.'
            
            sub_df = df.iloc[start_index:(target_index + 1), :].copy()
            
            args = (sub_df,
                    date,
                    look_back,
                    forecast_steps,
                    dmd_approach,
                    dmd_perc_cumul_var,
                    timeframes,
                    macd_fast,
                    macd_slow,
                    macd_signal)
                    
            inputs_ls.append(args)
            
    # Now, calculte features in parallel.
    cpu_count = multiprocessing.cpu_count()    
    num_processes = max(1, int(frac_cpu_to_use * cpu_count))
    chunksize = max(1, len(inputs_ls)//num_processes)
    with multiprocessing.Pool(processes = num_processes) as pool:  # 2 worker processes
       # Use imap for lazy, ordered processing
       results = list(tqdm(pool.imap(get_features_one_day, inputs_ls, chunksize = chunksize), total = len(inputs_ls)))

    feat_dicts = [i[0] for i in results]
    
    feat_df = pd.DataFrame(feat_dicts)
    feat_arr = feat_df.to_numpy()
    labels_arr = np.array(labels)
    
    return feat_arr, labels_arr, dates, feat_df