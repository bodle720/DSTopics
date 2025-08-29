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
- Next, you define a list of inputs (we will call this variable *inputs_ls*) to the worker that you would like pocessed. The list in my example will be of length 500 and contain tuples indicating the input arguments to the worker. It is constructed as follows:

```
inputs_ls = []
for i in range(50):
    for j in range(10):
        inputs_ls.append((i, j))

```
- Pass this into your chosen multiprocessing function and await completion. The results returned will be the output of yourworker function. My output will be a dictionary mapping argument names ('i' and 'j') to their respective value. E.g. {'i': 0, 'j': 0} is the result of the first task in *inputs_ls*.
- Be sure not to allocate too many CPU resources to the tasks; if the workload is intense, it may crash your system. So, use with caution and think through what makes sense for your situation.

There are two key parameters you must decide on: the number of processes (we will call this *num_processes*) to use and the chunk size (we will call this *chunksize*).

*num_processes*: This is how many processors you want to utilize. In my case, it's best not to use the full 32 and leave some to the side for general system operations. In my example, I will use 25 as the value for this parameter.

*chunksize*: This is how many individual calls to the worker constitutes one 'chunk', which are all sent to each worker process at one time. The default recommendation for this is to set:

```
chunksize = max(1, len(inputs_ls) // (num_processes * 4))
```

In our case, this works out to a value of 5. However, you can vary this parameter based on your needs and it can affect performance dramatically. If each worker call takes roughly the same length of time and are CPU-bound (as will be the case in our example), then you can set this higher. Doing so will reduce overhead time for batch assignments and keep your processors busy working. If your tasks vary widely in duration or you want faster access to results, you can set *chunksize* to a lower value. Note this will provide more frequent outputs and updates to the progress of the queue but also increase overhead. In our case, we will set this value to 10.

Behind the scenes, Python’s multiprocessing queues up our chunks of tasks and dynamically assigns them to available worker processes. Each chunk contains 10 tasks, and with 500 total tasks, we end up with 50 chunks. The pool of 25 worker processes begins by picking up the first 25 chunks. As each process finishes its chunk (which takes 5 seconds, assuming perfect uniformity), it immediately picks up another from the queue. In this ideal scenario, the first wave of 25 chunks completes in 5 seconds, and the second wave begins immediately, finishing another 25 chunks in the next 5 seconds. Thus, all 50 chunks are processed in approximately 10 seconds, assuming no overhead and perfect load balancing.

In my code, the order of results are guaranteed to be preserved per the order of *inputs_ls*, and we print the final results to verify. The output of the print statement (the first 15 returned results) will be:

```
[{'i': 0, 'j': 0},
 {'i': 0, 'j': 1},
 {'i': 0, 'j': 2},
 {'i': 0, 'j': 3},
 {'i': 0, 'j': 4},
 {'i': 0, 'j': 5},
 {'i': 0, 'j': 6},
 {'i': 0, 'j': 7},
 {'i': 0, 'j': 8},
 {'i': 0, 'j': 9},
 {'i': 1, 'j': 0},
 {'i': 1, 'j': 1},
 {'i': 1, 'j': 2},
 {'i': 1, 'j': 3},
 {'i': 1, 'j': 4}]
```

Note that a common pitfall is not wrapping the parallel call in a
```
if __name__ == '__main__':
```
block. This structure is required for the proper handling of parallel tasks.

I encourage experimentation with the *chunksize* parameter as it greatly affects visual tqdm update frequency. Using a chunksize of 10 provides very few updates before a sudden completion of all the tasks. Using a chunksize of 2 however provides more steady updates without loss in time.

At the end of the code the total execution time is provided in seconds. Over multiple runs, runtime was fairly consistently around 100 seconds, or 1 minute 40 seconds. 1 minute 40 seconds is significantly better than the approximately 42 minutes sequential execution would take, making multiprocessing certainly worthwhile. However, it is far from the earlier ideaized estimate of 10 seconds. There are many reasons for this and it is to be expected. Some reasons include OS scheduling being inefficient, overhead time of spawning processes takes time, and the time it takes to serialize each object can add up as well (pickle is used under the hood and can be slow).

I hope this code helps you save some time in your future workflows. Thank you for reading!
