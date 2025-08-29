# Python Multiprocessing
---

Multiprocessing is a technique for parallel computing that uses multiple CPU cores to perform required tasks simultaneously. It's useful when you have a worker function that you need applied multiple times with possibly varying arguments, but to perform it sequentially would be prohibitively time-consuming.

Python offers a module called *multiprocessing* that allows you to utilize the CPU cores on your machine (more specifically, the 'logical processors' that reside on the CPU cores) to queue up a set of tasks and tackle them simultaneously. For example, my machine has 24 CPU cores and 32 logical processors. There are more processors than cores because some of the cores support simultaneous multithreading, allowing them to handle two threads via two logical processors (sometimes referred to as 'virtual cores').

The number of available logical processors on your machine can be determined by running the following command in Python:

```
import multiprocessing
processor_count = multiprocessing.cpu_count()
```

In the script *multiprocessing_example.py*, I will build a worker function that simulates a task that takes 5 seconds to complete. My goal is to run this task 500 times. Clearly, doing this consecutively would result in a runtime of 2,500 seconds, or about 41.7 minutes. Using multiprocessing, we should be able to cut that down considerablely.

The *multiprocessing* module offers many different functionalities, and if you're curious, I highly recommend you read through it for more details. A short description of the workflow is as follows:

- You define a worker function that takes the necessary arguments and performs the work. This should be defined in such a way that it represents the smallest independent unit of work you need to complete. So, don't combine multiple repeat tasks into the same call to your worker function.
- Next, you define a list of inputs (we will call this variable *inputs_ls*) to the worker that you would like pocessed. The list in my example will be of length 500 and contain tuples indicating the input arguments to the worker.
- Finally, you pass this into your chosen multiprocessing function and await completion.
- Be sure not to allocate too many CPU resources to the tasks; if the workload is intense, it may crash your system. So, use with caution and think through what makes sense for your situation.

There are two key parameters you must decide on: the number of processes (we will call this *num_processes*) to use and the chunk size (we will call this *chunksize*).

*num_processes*: This is how many processors you want to utilize. In my case, it's best not to use the full 32 and leave some to the side for general system operations. In my example, I will use 25 as the value for this parameter.

*chunksize*: This is how many individual calls to the worker constitutes one 'chunk', which are all sent to each worker process at one time. The default recommendation for this is to set:

```
chunksize = max(1, len(inputs_ls) // (num_processes * 4))
```

In our case, this works out to a value of 5. However, you can vary this parameter based on your needs and it can affect performance dramatically. If each worker call takes roughly the same length of time and are CPU bound (as will be the case in our example), then you can set this higher. Doing so will reduce overhead time of batch assignments and keep your processors busy working. If your tasks vary widely in duration or you want faster access to results, you can set *chunksize* to a lower value. Note this will provide more frequent outputs and updates to the progress of the queue but also increase overhead. In our case, we will set this value to 10.

Behind the scenes, Python’s multiprocessing queues up our chunks of tasks and dynamically assigns them to available worker processes. Each chunk contains 10 tasks, and with 500 total tasks, we end up with 50 chunks. The pool of 25 worker processes begins by picking up the first 25 chunks. As each process finishes its chunk (which takes 5 seconds, assuming perfect uniformity), it immediately picks up another from the queue. In this ideal scenario, the first wave of 25 chunks completes in 5 seconds, and the second wave begins immediately, finishing another 25 chunks in the next 5 seconds. Thus, all 50 chunks are processed in approximately 10 seconds, assuming no overhead and perfect load balancing.

To 'visualize' the output, I will log each task completion to a *logs.txt* file with each pair of arguments. In my code, the order of results are guaranteed to be preserved per the order of *inputs_ls*, but we can inspect the log file to see if they are *completed* in that order.

As a final note, a common pitfall is not wrapping the parallel call in a
```
if __name__ == '__main__':
```
block. This structure is required for the proper handling of parallel tasks.
