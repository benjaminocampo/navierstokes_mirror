import os
from os import popen
from os import chdir, makedirs
from os.path import isdir
from time import time
import argparse

SHOULD_RUN = True # Generates run.output and perfstat.output

printf = lambda s: print(s, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-batch",
    "-nb",
    action="store_true",
    help="This flag will override the default behaviour of submiting to a slurm squeue, and instead directly run the commands"
)
arguments = parser.parse_args()

error_count = 0
def cmd(c, prefix="", suffix=""):
    global error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(f'{prefix}{c}{suffix}'):
        printf(f"{red_color}>>> [ERROR] there was an error in command:{end_color}")
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        exit()
        error_count += 1

def batch(branch, flags, n, steps):
    job_name = f"{n}x{steps}"
    prefix = f"nohup srun --job-name='{job_name}' --ntasks=1 --cpus-per-task=1 --exclusive " # Dettach job from this terminal
    suffix = f" 2>/dev/null &" # Nulls stderr output, nohup tends to send to stdout otherwise
    return run(branch, flags, n, steps, prefix, suffix)

def run(branch, flags, n, steps, prefix="", suffix=""):
    underscored = lambda s: "_".join(s.split())
    run_name = f"{branch}_n{n}_steps{steps}_{underscored(flags)}"
    run_cmd = f"./headless {n} 0.1 0.001 0.0001 5.0 100.0 {steps} > runs/stdouts/{run_name}.output"
    perfstat_cmd = f"perf stat -o runs/perfstats/{run_name}.output -e cache-references,cache-misses,L1-dcache-stores,L1-dcache-store-misses,LLC-stores,LLC-store-misses,page-faults,cycles,instructions,branches,branch-misses -ddd"
    perfstat_run_cmd = f"{perfstat_cmd} {run_cmd}"

    start_time = time()
    cmd(f"git checkout l1-{branch}")
    cmd("make clean")
    cmd(f"make headless CFLAGS='-g {flags}'")
    if (SHOULD_RUN): cmd(perfstat_run_cmd, prefix, suffix)
    printf(f">>> [TIME] Run finished in {time() - start_time} seconds.")

def setup_run_folder():
    makedirs("runs/stdouts", exist_ok=True)
    makedirs("runs/perfstats", exist_ok=True)

prun = run if arguments.no_batch else batch # Pick appropiate run method
setup_run_folder()
itime = time()
printf(">>> [START]")
for n, steps in [(2048, 32), (512, 128), (128, 512)]:
    prun("baseline", "-O0", n, steps)
    prun("baseline", "-O1", n, steps)
    prun("baseline", "-O2", n, steps)
    prun("baseline", "-O3", n, steps)
    prun("baseline", "-Ofast", n, steps)
    prun("baseline", "-Os", n, steps)
    prun("baseline", "-O3 -floop-interchange -floop-nest-optimize", n, steps)
    prun("ijswap", "-O3", n, steps)

    prun("invc", "-O3", n, steps)
    prun("ijswap", "-O3 -freciprocal-math", n, steps)
    prun("ijswap", "-Ofast", n, steps)

    prun("invc", "-Ofast", n, steps)
    prun("invc", "-Ofast -march=native", n, steps)
    prun("invc", "-Ofast -march=native -funroll-loops", n, steps)
    prun("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize", n, steps)
    prun("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

    prun("bblocks", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

    prun(f"constn{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
    prun(f"zdiffvisc{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

printf(f"Done in {time() - itime} seconds with {error_count} errors.")
