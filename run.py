import os
import argparse
from os import popen
from os import chdir, makedirs
from os.path import isdir
from time import time
from inspect import cleandoc

from utils import save_git_state, restore_git_state

SHOULD_RUN = True # Generates run.output and perfstat.output

printf = lambda s: print(s, flush=True)

error_count = 0
def cmd(c):
    global error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(c): # Command returned != 0
        printf(f"{red_color}>>> [ERROR] there was an error in command:{end_color}")
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        exit()
        error_count += 1

def run(branch, flags, n, steps):
    underscored = lambda s: "_".join(s.split())
    run_name = f"{branch}_n{n}_steps{steps}_{underscored(flags)}"
    run_cmd = f"./headless {n} 0.1 0.001 0.0001 5.0 100.0 {steps} > runs/stdouts/{run_name}.output"
    perfstat_cmd = f"perf stat -o runs/perfstats/{run_name}.output -e cache-references,cache-misses,L1-dcache-stores,L1-dcache-store-misses,LLC-stores,LLC-store-misses,page-faults,cycles,instructions,branches,branch-misses -ddd"
    perfstat_run_cmd = f"{perfstat_cmd} {run_cmd}"

    start_time = time()
    cmd(f"git checkout l1-{branch}")
    cmd("make clean")
    cmd(f"make headless CFLAGS='-g {flags}'")
    if (SHOULD_RUN): cmd(perfstat_run_cmd)
    printf(f">>> [TIME] Run finished in {time() - start_time} seconds.")

def batch(branch, flags, n, steps):
    underscored = lambda s: "_".join(s.split())
    run_name = f"{branch}_n{n}_steps{steps}_{underscored(flags)}"
    run_cmd = f"./headless {n} 0.1 0.001 0.0001 5.0 100.0 {steps} > runs/stdouts/{run_name}.output"
    perfstat_cmd = f"perf stat -o runs/perfstats/{run_name}.output -e cache-references,cache-misses,L1-dcache-stores,L1-dcache-store-misses,LLC-stores,LLC-store-misses,page-faults,cycles,instructions,branches,branch-misses -ddd"
    perfstat_run_cmd = f"{perfstat_cmd} {run_cmd}"
    submission_filename = f"runs/submissions/{run_name}.sh"
    submission_text = cleandoc(f"""
    #!/bin/bash
    #SBATCH --job-name={n}x{steps}
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --exclusive
    #SBATCH -o runs/slurmout/{run_name}.out
    #SBATCH -e runs/slurmerr/{run_name}.err

    git checkout l1-{branch} &&
    make clean &&
    make headless CFLAGS='-g {flags}' &&
    {perfstat_run_cmd} ||
    echo "If you see this file then your run with this filename had a problem, inspect runs/ folder for more information" > {run_name}.error # If this is in the root then you know there was an error
    """)
    with open(submission_filename, "w") as submission:
        submission.write(submission_text)
    if (SHOULD_RUN): cmd(f"sbatch ./{submission_filename}")

def setup_run_folder():
    makedirs("runs/submissions", exist_ok=True)
    makedirs("runs/stdouts", exist_ok=True)
    makedirs("runs/perfstats", exist_ok=True)
    makedirs("runs/slurmout", exist_ok=True)
    makedirs("runs/slurmerr", exist_ok=True)

def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-batch",
        "-nb",
        action="store_true",
        help="This flag will override the default behaviour of submiting to a slurm squeue, and instead directly run the commands"
    )
    arguments = parser.parse_args()

    prun = run if arguments.no_batch else batch # Pick appropiate run method
    setup_run_folder()
    repo, initial_branch = save_git_state()
    itime = time()
    printf(">>> [START]")
    for n, steps in [(128, 512), (2048, 32), (512, 128)]:
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

    # Batch command to restore git state after all batches
    cmd(f"nohup srun --job-name=cleanup 'git checkout {initial_branch} && git stash pop' &")
    printf(f"Done in {time() - itime} seconds with {error_count} errors.")

if __name__ == "__main__":
    main()
