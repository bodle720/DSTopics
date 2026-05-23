# Python Multiprocessing

Multiprocessing is a technique for parallel computing that uses multiple CPU cores to perform independent tasks simultaneously. It is useful when a worker function needs to be applied many times, often with varying arguments, and running those tasks sequentially would be prohibitively slow.

Python provides a built-in module called `multiprocessing` that can use the logical processors available on a machine. Logical processors are the execution units exposed to the operating system. A CPU may have more logical processors than physical cores when it supports simultaneous multithreading.

The number of available logical processors can be checked with:

```python
import multiprocessing

processor_count = multiprocessing.cpu_count()
print(processor_count)
```

In `multiprocessing_example.py`, a worker function simulates a task that takes 5 seconds to complete. The script runs this simulated task 500 times. Sequential execution would take:

```text
500 tasks × 5 seconds/task = 2,500 seconds
```

That is about 41.7 minutes. Multiprocessing can reduce the wall-clock runtime by distributing independent tasks across multiple worker processes.

## Basic Workflow

The general workflow is:

1. Define a worker function that performs one independent unit of work.
2. Build a list of input arguments for the worker.
3. Create a process pool.
4. Map the worker function across the input list.
5. Collect the returned results.

The worker function should represent the smallest independent unit of work. Avoid combining multiple repeat tasks into a single worker call unless there is a specific reason to do so.

In this example, the input list contains 500 `(i, j)` argument pairs:

```python
inputs_ls = []

for i in range(50):
    for j in range(10):
        inputs_ls.append((i, j))
```

Each input tuple is passed to the worker function. The returned result is a dictionary containing the input values:

```python
{'i': 0, 'j': 0}
```

## Number of Processes

One important parameter is the number of worker processes:

```python
num_processes = 25
```

This controls how many separate Python worker processes are created. It is usually not necessary, or desirable, to use every logical processor on the machine. Leaving some capacity available for the operating system and other applications can make the system more stable and responsive.

The appropriate value depends on the workload, machine, memory use, and whether the tasks are CPU-bound, I/O-bound, or mostly waiting.

## Chunk Size

Another important parameter is `chunksize`.

When using `Pool.imap`, the input iterable is split into chunks. Each chunk contains some number of individual worker calls. A chunk is assigned to a worker process, and that process works through the calls in that chunk.

A common default-style heuristic is:

```python
chunksize = max(1, len(inputs_ls) // (num_processes * 4))
```

For 500 inputs and 25 worker processes, this gives:

```text
500 // (25 × 4) = 5
```

The example script uses:

```python
chunksize = 10
```

With 500 tasks and a chunk size of 10, the work is divided into 50 chunks:

```text
500 tasks / 10 tasks per chunk = 50 chunks
```

Since each task sleeps for 5 seconds, each chunk takes about 50 seconds of worker time:

```text
10 tasks/chunk × 5 seconds/task = 50 seconds/chunk
```

With 25 worker processes, the first 25 chunks run in parallel, then the remaining 25 chunks run in a second wave. Ignoring overhead, the expected runtime is therefore approximately:

```text
2 waves × 50 seconds/wave = 100 seconds
```

This matches the observed runtime much better than assuming each chunk takes only 5 seconds.

## Choosing a Chunk Size

The best `chunksize` depends on the task.

A larger `chunksize` can reduce scheduling overhead because fewer chunks need to be assigned to worker processes. This can work well when individual tasks take similar amounts of time.

A smaller `chunksize` can improve load balancing when task runtimes vary. It also makes progress bars update more frequently, because results are returned in smaller groups. The tradeoff is increased scheduling overhead.

In this example, lowering `chunksize` can make the `tqdm` progress bar update more smoothly. Increasing `chunksize` can make updates less frequent, sometimes appearing as though the progress bar is stuck before completing in larger jumps.

## Ordered Results with `imap`

The example uses:

```python
pool.imap(my_worker, inputs_ls, chunksize=chunksize)
```

`imap` returns results lazily and preserves the order of the input iterable. This means the returned results appear in the same order as `inputs_ls`.

The first 15 returned results are:

```python
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

If result order is not important, other multiprocessing methods may be more appropriate. For example, `imap_unordered` can return results as soon as they are ready.

## Important Guard for Script Execution

A common multiprocessing pitfall is forgetting to wrap the process-pool code in:

```python
if __name__ == '__main__':
    ...
```

This guard is especially important on Windows, where new worker processes import the main module. Without the guard, the script can recursively start new processes or fail unexpectedly.

The example script uses this structure so it can be run safely from the command line.

## Running the Example

From the `Python_Multiprocessing/` folder:

```bash
python multiprocessing_example.py
```

The script will:

1. Print the number of logical processors.
2. Build 500 input tasks.
3. Run the worker function using a process pool.
4. Display progress with `tqdm`.
5. Print the total runtime.
6. Print the first 15 returned results.

## Notes

Multiprocessing is useful when the work can be split into independent tasks. It is not always faster than sequential execution. Process creation, task scheduling, inter-process communication, and object serialization all add overhead.

Python uses serialization, commonly through `pickle`, to send data between the main process and worker processes. Large inputs or outputs can reduce the benefit of multiprocessing.

For best results, keep worker inputs and outputs reasonably small, avoid shared mutable state, and test different values of `num_processes` and `chunksize` for the specific workload.