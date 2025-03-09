# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:58:05 2025

@author: brian
"""
import os
import json
import numpy as np
import yfinance as yf
from datetime import datetime

from helpers import generate_data_and_labels

if __name__ == '__main__':
    
    ###########################################################################
    # Parameters.
    ###########################################################################
    
    # Where to save the data for future use.
    save_to = r"C:/Users/brian/Desktop/output_DMD"
    
    # Define parameters for the data to use.
    ticker = "AAPL"
    start_date = "2005-01-01"
    end_date = "2025-03-01"
    
    # Define parameters for DMD feature generation.
    look_back = 8
    dmd_perc_cumul_var = 0.95
    timeframes = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Labeling parameter to determine of successful going short or long.
    label_freq = 20
    
    # How many of your CPU cores to use to process the data in parallel.
    frac_cpu_to_use = 0.7
    ###########################################################################
    
    assert os.path.exists(os.path.dirname(save_to)), 'Save to path does not exist'
    
    # Create the data.
    df = yf.download(ticker,
                    start = start_date,
                    end = end_date,
                    multi_level_index = False,
                    interval="1d")
    
    # Generate the features.
    feat_arr, labels_arr, dates, feat_df = generate_data_and_labels(df.copy(),
                                                                     look_back,
                                                                     dmd_perc_cumul_var,
                                                                     timeframes,
                                                                     macd_fast,
                                                                     macd_slow,
                                                                     macd_signal,
                                                                     label_freq,
                                                                     frac_cpu_to_use)
    # Save out the data.
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    
    now = datetime.now()
    formatted_timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")
    final_save_to = os.path.join(save_to, formatted_timestamp + '_run')        
    os.mkdir(final_save_to)
    
    feat_df.to_csv(os.path.join(final_save_to, 'df_feats.csv'), index = False)
    
    np.save(os.path.join(final_save_to, 'feats.npy'), feat_arr)
    np.save(os.path.join(final_save_to, 'labels.npy'), labels_arr)
    
    meta_info = {'ticker':ticker,
                 'start_date': start_date,
                 'end_date':end_date,
                 'look_back': look_back,
                 'dmd_perc_cumul_var': dmd_perc_cumul_var,
                 'timeframes': timeframes,
                 'macd_fast': macd_fast,
                 'macd_slow': macd_slow,
                 'macd_signal': macd_signal,
                 'label_freq': label_freq,
                 'dates': dates,
                 'feat_names': list(feat_df.columns)}
    
    with open(os.path.join(final_save_to, 'meta_info.json'), 'w') as f:
        json.dump(meta_info, f, indent = 3)