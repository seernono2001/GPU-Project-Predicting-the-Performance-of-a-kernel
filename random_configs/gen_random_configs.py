#!/usr/bin/env python3
import math, random, csv
import statistics, sys
from collections import Counter
import numpy as np

########################################################################
'''
RANDOM PARAMETER GENERATOR---
CHANGE THE FOLLOWING PARAMETERS AS NEEDED BETWEEN THESE LINES
'''
PROBLEM_MIN, PROBLEM_MAX = 1000, 100000000 # 10K -> 100M elements
BLOCKS_MIN, BLOCKS_MAX = 1, 65535 # grid range (65,535 max for CUDA)
BLOCK_MULTIPLIER = 2.0  # controls randomizations spread of blocks
THREAD_MIN, THREAD_MAX = 32, 1024 # CUDA range (max 1024)

# number of problem sizes, threads, blocks
NUM_PROBLEMS = 1000
# sample size for each GPU = 1000
########################################################################
configs = []

# Optional -- but cap for maxtrix multiplcation memory constraint
MAX_DIM = 2048
MAX_ELEMENTS = MAX_DIM * MAX_DIM

# Generates Lists
'''
Logarmithic + Uniform Sampling Mix
'''
problem_sizes = [int(round(10 ** random.uniform(math.log10(PROBLEM_MIN), math.log10(PROBLEM_MAX)))) for i in range(NUM_PROBLEMS)]
problem_sizes = sorted(set(problem_sizes))
good_threads = [i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 == 0]
bad_threads  = [i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 != 0]

# from old version
# num_blocks = sorted(set(int(round(10 ** random.uniform(math.log10(BLOCKS_MIN), math.log10(BLOCKS_MAX)))) for i in range(NUM_PROBLEMS)))


''' CODE FOR UNIFORM SAMPLING
problem_sizes = sorted(random.sample(range(PROBLEM_MIN, PROBLEM_MAX), NUM_PROBLEMS))
threads_good = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 == 0], NUM_THREADS // 2) # half from good
threads_bad  = random.sample([i for i in range(THREAD_MIN, THREAD_MAX + 1) if i % 32 != 0], NUM_THREADS // 2) # half from bad
threads_per_block = threads_good + threads_bad
random.shuffle(threads_per_block)
num_blocks = random.sample(range(BLOCKS_MIN, BLOCKS_MAX + 1), NUM_BLOCKS)
'''

# GENERATE COMBINATIONS
for ps in problem_sizes:
    # Clamp for safe memory size (for matrix kernels)
    if ps > MAX_ELEMENTS:
        ps = MAX_ELEMENTS

    # Randomly decide alignment type (50% chance)
    if random.random() < 0.5:
        tpb = random.choice(good_threads)
        alignment = "good"
    else:
        tpb = random.choice(bad_threads)
        alignment = "bad"

    # compute minimum blocks needed to cover problem size
    blocks_min = math.ceil(ps / tpb)
    # ADJUSTMENT 1: If too many blocks, inc threads to reduce blocks
    if blocks_min > BLOCKS_MAX:
        tpb = math.ceil(ps / BLOCKS_MAX) # threads per block should be less then or rqual to 65,535
        # This is the CUDA THEAD LIMIT
        if tpb > THREAD_MAX:
            tpb = THREAD_MAX
        blocks_min = math.ceil(ps / tpb)  # Recalc blocks after increasing threads

    # Randomized block range
    blocks_max = int(min(blocks_min * BLOCK_MULTIPLIER, BLOCKS_MAX))
    nb = random.randint(max(1, blocks_min), max(1, blocks_max)) # choose rand num btwn min and max

    # old method
    # nb = int(round(10 ** random.uniform(math.log10(BLOCKS_MIN), math.log10(BLOCKS_MAX))))
    # Adjustment 2: Final check to make sure everything is covered
    if nb * tpb < ps:
        nb = math.ceil(ps / tpb)
        if nb > BLOCKS_MAX:
            nb = BLOCKS_MAX

    configs.append((ps, tpb, nb, alignment))


print(f"Generated {len(configs)} configurations")

# WRITE TO CSV FILE
with open("random_configs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index","ProblemSize", "ThreadsPerBlock", "NumBlocks", "Alignment"])
    for i, (ps, tpb, nb, align) in enumerate(configs, start =1):
         writer.writerow([i, ps, tpb, nb, align])

print("Generated file: random_configs.csv")


################# STATISTIC SUMMARY ######################
sys.stdout = open("stat_sum.txt", "w")

problem_values = [ps for ps, _, _, _ in configs]
thread_values  = [tpb for _, tpb, _, _ in configs]
block_values   = [nb for _, _, nb, _ in configs]
align_values   = [align for *_, align in configs]

def stat_sum(data, name):
    arr = np.array(data)
    print(f"\n=== {name} ===")
    print(f"Count: {len(arr)}")
    print(f"Min: {np.min(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Mean: {np.mean(arr):.2f}")
    print(f"Median: {np.median(arr)}")
    print(f"Std Dev: {np.std(arr, ddof=1):.2f}")
    print(f"25th percentile (Q1): {np.percentile(arr, 25)}")
    print(f"75th percentile (Q3): {np.percentile(arr, 75)}")
    print(f"IQR (Q3 - Q1): {np.percentile(arr, 75) - np.percentile(arr, 25)}")
    try:
        mode = statistics.mode(data)
        print(f"Mode: {mode}")
    except statistics.StatisticsError:
        print("Mode: No unique mode (multiple values equally common)")


# Print numeric summaries
stat_sum(problem_values, "Problem Sizes")
stat_sum(thread_values, "Threads Per Block")
stat_sum(block_values, "Num Blocks")

# Alignment distribution
align_counter = Counter(align_values)
print("\n=== Alignment Distribution ===")
for k, v in align_counter.items():
    print(f"{k}: {v} ({v/len(configs)*100:.1f}%)")
