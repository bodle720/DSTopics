# -*- coding: utf-8 -*-
"""
A file to make the GIF as described in the notebook.
To plot many at once, the code loops thorugh days and saves the plots anf GIF to a folder
saved in a new folder called <save_to> in the current working directory.
"""

# Imports
import os
import imageio
import yfinance as yf
import pandas as pd

from helpers import plot_DMD_forecasts

if __name__ == '__main__':
    
    ###########################################################################
    # Parameters.
    ###########################################################################
    save_to = r"plots"
    
    if not os.path.exists(save_to):
        os.mkdir(save_to)
        
    save_to = os.path.abspath(save_to)
    
    ticker = "AAPL"
    start_date = "2005-01-01"
    end_date = "2025-03-02"
    st_date = '2012-07-13'
    
    look_back = 10
    forecast_steps = 5
    view_tf = 1 
    dmd_approach = 'iterative'
    dmd_perc_cumul_var = 0.95
    timeframes = [1, 2]
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    steps_forward = 30
    ###########################################################################

    ###########################################################################
    # Do the work.
    ###########################################################################

    # Create the data.
    df = yf.download(ticker,
                    start = start_date,
                    end = end_date,
                    multi_level_index = False,
                    interval = "1d")
    
    target_date = pd.Timestamp(st_date)
    target_index = df.index.get_loc(target_date)
    
    dates = [str(i).split(' ')[0] for i in df.index][target_index:(target_index + steps_forward)]
    images = []
    
    # Modify the loop to plot only 1 or 2 plots if you want.
    for ix, date in enumerate(dates):
        if ix % 10 == 0:
            print(f'On date {ix+1}')
        try:
            save_path = os.path.join(save_to, f"{date}.png")
            _ = plot_DMD_forecasts(df,
                                    date,
                                    look_back,
                                    forecast_steps,
                                    view_tf,
                                    dmd_approach = dmd_approach,
                                    dmd_perc_cumul_var = dmd_perc_cumul_var,
                                    timeframes = timeframes,
                                    macd_fast = macd_fast,
                                    macd_slow = macd_slow,
                                    macd_signal = macd_signal,
                                    save_to = save_path)
            
            images.append(imageio.v2.imread(save_path))
    
        except:
            pass
    
    # Comment these lines out to not make the GIF.
    gif_save_path = os.path.join(save_to, f"{steps_forward}_steps_starting_at_{st_date}.gif")
    imageio.mimsave(gif_save_path, images, fps = 1)