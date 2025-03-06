# -*- coding: utf-8 -*-
"""
A script of auxiliary functions.
"""

import numpy as np
import pandas as pd

def calculate_heikin_ashi(df):
    
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

def apply_DMD(X,
              X_prime,
              approach = 'iterative',
              forward_steps = 3,
              perc_cumul_var = 0.85):
    
    '''
    Runs the DMD algorithm and chooses r (rank) based on the quantity of cumulative
    explained variance in the eigenvalues. X and X_prime contain snapshots of data in time,
    organized as columns (each column represents one snapshot). This function will always use
    the most recent snapshot in time for future predictions.
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    