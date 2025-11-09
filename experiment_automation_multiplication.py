# TESTING AUTOMATION SCRIPT FOR EXPERIMENTS
import itertools, subprocess, csv, re

########################################################################
'''
CHANGE THE FOLLOWING PARAMETERS AS NEEDED BETWEEN THESE LINES
'''
EXECUTABLE = "./gpu_project"
GPU = "GPU_2" # CHANGE THE NAME FOR EACH GPU
REPETITION = 2 # SMALL NUMBER FOR TESTING (to mitigate noise) - repetition for each configuration
OPERATION = "multiplication"

CONFIGURATION_FILE = "random_configs.csv"

# Updates the CSV FILE NAME WITH THE GPU NAME
CSV_FILE = f"results_multiplication_{GPU}.csv"
########################################################################


# CSV FILE
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", 
                     "GPU",
                     "ProblemSize",
                     "ThreadsPerBlock",
                     "NumBlocks",
                     "Alignment",
                     "Operation",
                     "Avg_SingleThreadTime(sec)",
                     "Avg_KernelTime(sec)",
                     "Avg_Speedup"])

with open(CONFIGURATION_FILE, "r") as f:
    reader = csv.DictReader(f)

    for count, row in enumerate(reader, start=1):
        idx = row["Index"]
        ps = int(row["ProblemSize"])
        tpb = int(row["ThreadsPerBlock"])
        nb = int(row["NumBlocks"])
        align = row["Alignment"]
        print(f"[{idx}] Executing: {EXECUTABLE} {ps} {tpb} {nb} {OPERATION} ({align} alignment)")

        single_thread_times = []
        kernel_times = []
        speedups = []

        # REPEAT FOR AVERAGING
        for _ in range(REPETITION):
            # RUN THE CUDA PROGRAM FROM INSIDE PYTHON ON THE COMMAND LINE
            result = subprocess.run([EXECUTABLE, str(ps), str(tpb), str(nb), OPERATION],
                                    capture_output=True, text=True)
            
            out = result.stdout.strip()

            # READING CUDAs OUTPUT (REGEX)
            cpu_time_match = re.search(r"Single thread time\s*=\s*([0-9.]+)", out)
            gpu_time_match = re.search(r"Kernel time\s*=\s*([0-9.]+)", out)
            speedup_match = re.search(r"Speedup:\s*([0-9.]+)", out)

            if cpu_time_match and gpu_time_match and speedup_match:
                single_thread_times.append(float(cpu_time_match.group(1)))
                kernel_times.append(float(gpu_time_match.group(1)))
                speedups.append(float(speedup_match.group(1)))
            else:
                print("ERROR- OUTPUT COULD NOT BE PARSED:\n", out)

        # SAVE AVERAGED RESULTS
        if single_thread_times:
            avg_single = sum(single_thread_times) / len(single_thread_times)
            avg_kernel = sum(kernel_times) / len(kernel_times)
            avg_speed  = sum(speedups) / len(speedups)

            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([
                                        idx, 
                                        GPU,
                                        ps,
                                        tpb,
                                        nb,
                                        align,
                                        OPERATION,
                                        avg_single,
                                        avg_kernel, 
                                        avg_speed])
            # PRINT AVERAGED RESULTS (CAN DELETE LATER) <------------------------
            print(f"Average Speedup = {avg_speed:.2f}x\n")
        else:
            print("No Valid Data\n")

print(f"\nAutomation Finished. Results saved to {CSV_FILE}")
