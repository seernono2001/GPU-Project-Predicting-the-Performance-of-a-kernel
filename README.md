# GPU-Project-Predicting-the-Performance-of-a-kernel

## Setups

1. SSH TO A CUDA SERVER
    - Example (using GPU 2):
       ```
       ssh cuda2.cims.nyu.edu
       ```
2. LOAD CUDA MODULE
    - Example:
      ```
      module avail cuda
      module load cuda-12.2
      ```
      
## CUDA Benchmark Program — GPU_project.cu

Compile:

`nvcc -o project GPU_project.cu -lm`

Run commands:

`./project [PROBLEM_SIZE] [NUM_THREADS] [NUM_BLOCKS] [OPERATION]`

For example:

`./project 100 32 5 addition`

Available Operations:

`addtion`, `subtraction`, `multiplication`, and `reduction`

It should output something like

```
Number of devices: 2
Currently using Device: 0

Device 0:
name: NVIDIA GeForce RTX 2080 Ti
total global memory(KB): 11081664
shared mem per block: 49152
warp size: 32
clock rate(KHz): 1635000

Single thread time = 5.041408 secs
GPU: 5 blocks of 32 threads each
Kernel time = 0.039840 secs
Speedup: 126.54x
```
## Automation Script — experiment_automation.py

The automation script tests CUDA kernel performance across different configurations and GPUs. It repeatedly runs the compiled CUDA program (gpu_project) with varying parameters (problem size, threads per block, number of blocks, and operation type), records the output, and writes averaged timing results into a CSV file.

### What It Does
- Executes the CUDA binary using different configurations.
- Captures printed output (CPU time, GPU kernel time, and computed speedup).
- Parses those values using regular expressions.
- Repeats each experiment multiple times to average out timing noise.
- Saves all averaged results to a file named results_<GPU>.csv (e.g. results_cuda2.csv).

Each GPU node generates its own CSV file, which can later be merged into one master dataset for ML analysis.

### How to run the automation script:

 RUN THE AUTOMATION SCRIPT:
 
`python3 experiment_automation_(insert_name_of_operation).py`
replace the (insert_name_of_operation) with either `addition`, `subtraction`, `multiplication`, or `reduction`
