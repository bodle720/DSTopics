# -*- coding: utf-8 -*-
"""
This is the main script you can run from the command line to demonstrate the multiprocesssing
example described in the README file.
"""

import time
import multiprocessing
from tqdm import tqdm
from pprint import pprint

def my_worker(args):
    '''
    This will simulate doing a unit of work that takes 5 seconds to complete.
    '''
    i, j = args
    
    # Your work goes here. Wait 5 seconds to simulate doing this work.
    # You can save results from inside if desired, but logging to the same file 
    # leads to race conditions and messy output.
    time.sleep(5)
    
    return {'i': i, 'j': j}

if __name__ == '__main__':
        
    cpu_count = multiprocessing.cpu_count()  
    num_processes = 25
    print('Starting the multiprocessing example.')
    print(f'We have {cpu_count} logical processors, but using {num_processes}.')

    # Form the inputs_ls as described in the README file.
    inputs_ls = []
    for i in range(50):
        for j in range(10):
            inputs_ls.append((i, j))
            
    chunksize = 10 # Experiment with this. Lower gives more frequent updates from tqdm.
    default_chunksize = max(1, len(inputs_ls) // (num_processes * 4)) # Default Python behavior.
    print(f'The default recommended chunksize is {default_chunksize}, but using {chunksize}.')

    print('Beginning imap call.')
    st_time = time.time()
    with multiprocessing.Pool(processes = num_processes) as pool:
       # Use imap for lazy, ordered processing.
       results = list(tqdm(pool.imap(my_worker, inputs_ls, chunksize = chunksize), total = len(inputs_ls)))
      
    total_time = round(time.time() - st_time, 2)
    
    print(f'Done with multiprocessing example, took {total_time} seconds.')
    print('The first 15 results in order are:')
    pprint(results[:15])