# GPU-Project-Predicting-the-Performance-of-a-kernel

Run commands:

`./vectorprog [PROBLEM_SIZE] [NUM_THREADS] [NUM_BLOCKS] [OPERATION]`

For example:

`./vectorprog 100 32 5 addition`

It should output something like

```
Single thread time = 24.055807 secs

GPU: 5 blocks of 32 threads each

Kernel time = 0.042080 secs

Speedup: 571.67x
```
