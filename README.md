- compute capability:
    nvidia-smi --query-gpu=compute_cap --format=csv

Check
------

* Terminology
  - Thread * Thread Block * Thread Cluster * Grid
  - Warp - Bank - SM
* Tail effect
* Shared memory usage
* Occupancy Calculator usage
* Bank conflict
* Tiling
* Paralel programming Patterns
* Cuda API usage
* Arithmetic intensity -> It is the main goal.
    Maximize the Arithmetic intensity = (amount of math operation) / (amount of memory accessed)
    In order to do that we can;
        - maximize the compute operations per thread.
        - minimize the time spent on memory per thread.


+ Thread divergence
  Thread execution within Warp executed together. It there for loop or if condition statements in kernel. It is possible that some threads
  executes different path from the others. It blocks all threads in warp even some of them complete the execution. It causes some slowness in 
  execution.
  
+ Coalesce vs Strided memory access
  It  is efficient that if threads with contiguous ids use contigiuous memory locations. The reason is read operation of a thread also triggers the retrieving contiguous data to cache. So that, contiguous thread don't need to fet global memory again.
  Strided access is inefficient because each threads needs to access global memory.

+ Memory access data speed;
  local < shared <<  global memory
  |
  |-> register on L1 cache

+ Parallel Communication Patterns
    - map: 1-to-1
    - gather: n-to-1
    - stencil: several-to-one - similar to gather read operation is based on a common pattern  Von Neumann, Moore stencils
    - scatter: 1-to-n - tasks compute where to write output
    - Transpose: 1-to-1 -  Tasks reorder the data elements in memory
    - reduce: all-to-one
    - scan/sort: all-to-all

+ GPU is responsible for allocating blocks to SMs.
+ Cuda guarantees that all threads in a block run in the same SM at the same time.
+ Cuda guarantees that all the blocks in a kernel finish before any blocks from the next kernel run.