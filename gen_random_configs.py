#!/usr/bin/env python3
import math, random, csv

########################################################################
'''
RANDOM PARAMETER GENERATOR---
CHANGE THE FOLLOWING PARAMETERS AS NEEDED BETWEEN THESE LINES
'''
NUM_EXPERIMENTS = 200
PROBLEM_MIN, PROBLEM_MAX = 1000, 100000000 # 10K -> 100M elements
BLOCKS_MIN, BLOCKS_MAX = 1, 1024 # grid range
THREAD_MIN, THREAD_MAX = 32, 1024 # CUDA range (max 1024)

# number of problem sizes, threads, blocks
NUM_PROBLEMS = 10 
NUM_THREADS = 10 
NUM_BLOCKS = 5 
# sample size for each GPU = 500
########################################################################

configs = []

# Generates Lists
'''
Logarmithic + Uniform Sampling Mix
'''
problem_sizes = [int(round(10 ** random.uniform(math.log10(PROBLEM_MIN), math.log10(PROBLEM_MAX)))) for i in range(NUM_PROBLEMS)]
problem_sizes = sorted(set(problem_sizes))
threads_good = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 == 0], NUM_THREADS // 2)
threads_bad  = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 != 0], NUM_THREADS // 2)
threads_per_block = threads_good + threads_bad
random.shuffle(threads_per_block)

num_blocks = sorted(set(int(round(10 ** random.uniform(math.log10(BLOCKS_MIN), math.log10(BLOCKS_MAX)))) for i in range(NUM_BLOCKS)))


''' CODE FOR UNIFORM SAMPLING
problem_sizes = sorted(random.sample(range(PROBLEM_MIN, PROBLEM_MAX), NUM_PROBLEMS))
threads_good = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 == 0], NUM_THREADS // 2) # half from good
threads_bad  = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 != 0], NUM_THREADS // 2) # half from bad
threads_per_block = threads_good + threads_bad
random.shuffle(threads_per_block)
num_blocks = random.sample(range(BLOCKS_MIN, BLOCKS_MAX + 1), NUM_BLOCKS)
'''

# GENERATE COMBINATIONS
configs = []
configs = []
for ps in problem_sizes:
    for tpb in threads_per_block:
        for nb in num_blocks:
            if tpb % 32 == 0:
                alignment = "good"
            else:
                alignment = "bad"
            configs.append((ps, tpb, nb, alignment))

print(f"Generated {len(configs)} configurations ({NUM_PROBLEMS}×{NUM_THREADS}×{NUM_BLOCKS})")

# WRITE TO CSV FILE
with open("random_configs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ProblemSize", "ThreadsPerBlock", "NumBlocks", "Alignment"])
    writer.writerows(configs)

print("Generated file: random_configs.csv")