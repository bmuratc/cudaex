- compute capability:
    nvidia-smi --query-gpu=compute_cap --format=csv

Check
------

* Shared memory usage
* Bank conflict
* Tiling
* Paralel programming Patterns
* Cuda API usage
* Arithmetic intensity -> It is the main goal.
    Maximize the Arithmetic intensity = (amount of math operation) / (amount of memory accessed)
    In order to do that we can;
        - maximize the compute operations per thread.
        - minimize the time spent on memory per thread.
* nsight and nvvp usage
* Occupancy Calculator usage
* Tail effect
* What is local memory on GPU
  L1 and SMEM using same area. It is configurable
* What is L2 cache
* Fundamental algorithms;
  - Reduce
  - Scan
  - Histogram
* What is work efficient?
* What is Brent's theorem?

- Reduce: 
  Serial reduce algorithm work and step complexities are linear O(n).
  Paralel reduce algorithm work complexity is linear O(n). step complexity is O(log n)

Serial Reduce:
i.e. : 1+2+3+4+5+6+7+8
    1   2   3   4    5        6        7        8
    \  /   /   /  
     \/   /   /  
      \  /   /
       \/   / 
        \  /
         \/
          ....

Parallel reduce:

    1   2   3   4    5        6        7        8
    \  /     \  / 
     \/       \/   
      \       /
       \     / 
        \   /
         \ /
          ....

  REDUCE : ( Set of Elements, Reduction operator)

  -  Reduction operator : should be binary ( a op b), associtive ((a op b) op c) == (a op (b op c))

+ Step complexity : number of steps to finis the job
+ Work complexity : total amount of work
  i.e.: In following operation : Step complexity is 3. work complexity is 7.
     X     X     X     X     X     X     X     X 
      \   /       \  /        \  /        \  /   
        X          X           X           X
          \       /              \       /
            \   /                  \    /
              X                       X
                \                   /
                  \               /
                    \           /
                      \       /
                        \   /
                          X

- SCAN Same as reduce that it takes list of data and an operation. output is running op of the set.
 i.e.
    INPUT: 1, 2, 3, 4 ...
    OPERATION : ADD
    OUTPUT: 1, 3, 6, 10 ...
    also needs IDENTITY OPERATOR such that [I op a == a]

  EXAMPLE of EXCLUSIVE SCAN
    operation : op
    input :[ a0, a1, a2,       a3,             a4, a5, a6, a7, a8 ... a(n-1)]
    output:[ I   a0, a0 op a1, a0 op a1 op a2, ....]

  EXAMPLE of INCLUSIVE SCAN
    operation : op
    input :[ a0, a1,       a2,             a3,             a4, a5, a6, a7, a8 ... a(n-1)]
    output:[ a0, a0 op a1, a0 op a1 op a2, ....]

  If the inclusive scan op implemets like reduction operation. step complexity will be O(log(n)) bu the work complexity will be O(n^2)
                   
   - Scan algorithms

                     MORE STEP EFFICIENT          MORe WORK EFFICIENT
  HILLIS&STEELE              X
  BLELLOCH                                                X

   - HILLIS&STEELE scan algorithm (inclusive scan)
   1        2        3        4        5        6        7        8
   1        3        5        7        9       11       13       15
   1        3        6       10       14       18       22       26
   1        3        6       10       15       21       28       36


   Work complexity is O(n), Step complexity is O(n * log (n))

   - BLELLOCH scan algorithm (exclusive scan)
   1        2        3        4        5        6        7        8
            3                 7                11                15
                             10                                  26
                                                                 36
                             10                                   0 (Identity Operator)
                              0                                  10
           3                  0                11                10
           0                  3                10                21
  1        0        3         3        5       10        7       21
  0        1        3         6       10       15       21       28

  There are 2 parts of this algorithm first part is reduce (ops till identity operator) and the 
  second phase is downsweep. Each part has same work complexity O(n) and step complexity O(lon(n)).
  
  - More Work than Processors -> Work Efficient -> Choose Blelloch 
  - More Processors than work -> Step Efficient -> Choose Hillis&Steele
  



+ Thread divergence
  Thread execution within Warp executed together. It there for loop or if condition statements in kernel. It is possible that some threads
  executes different path from the others. It blocks all threads in warp even some of them complete the execution. It causes some slowness in 
  execution.

* Bank Conflict
  - https://www.youtube.com/watch?v=qOCUQoF_-MM
  - SMEM is organized as 32 independent memory bank.
  - If two threads within a wrap try to access DIFFERENT ADDRESSES in a same bank, it is 2-way bank conflict. Because the read operation of the data from the addresses should be sequential. Bank cannot provide data from two different addresses in paralel way. To SAME ADDRESS
  accesses not a problem.
  - In order to avoid the bank conflict padding is used. 
  - Banks are organized in words(usually 4 bytes long)

+ Coalesce vs Strided memory access
  It is efficient that if the threads with contiguous ids use contigiuous memory locations. The reason is read operation of a thread also triggers the retrieving contiguous data to cache. So that, contiguous thread don't need to fet global memory again.
  Strided access is inefficient because each threads needs to access global memory.

+ Memory access data speed;
  local < shared <<  global memory
  |
  |-> registers on L1 cache

+ Parallel Communication Patterns
    - map: 1-to-1
    - gather: n-to-1 (No overlapping)
    - stencil: several-to-one - similar to gather read operation is based on a common pattern  Von Neumann, Moore stencils
    - scatter: 1-to-n - tasks compute where to write output
    - Transpose: 1-to-1 -  Tasks reorder the data elements in memory
    - reduce: all-to-one
    - scan/sort: all-to-all

+ GPU is responsible for allocating blocks to SMs.
+ Cuda guarantees that all threads in a block run in the same SM at the same time.
+ Cuda guarantees that all the blocks in a kernel finish before any blocks from the next kernel run.