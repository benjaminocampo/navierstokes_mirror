import os
import argparse
from os import popen
from os import chdir, makedirs
from os.path import isdir
from time import time
from inspect import cleandoc

from utils import save_git_state, restore_git_state

printf = lambda s: print(s, flush=True)

cmd_error_count = 0


def cmd(c):
    global cmd_error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(c):  # Command returned != 0
        printf(f"{red_color}>>> [ERROR] there was an error in command:{end_color}")
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        cmd_error_count += 1
        exit()


class Run:
    is_run_folder_initialized = False
    should_run = True

    def __init__(
        self,
        name,
        n,
        steps,
        branch_prefix="",
        cflags="",
        previous_batch_cmds="",
        envvars={},
    ):
        self.name = name
        self.n = n
        self.steps = steps
        self.branch_prefix = branch_prefix
        self.cflags = cflags
        self.previous_batch_cmds = previous_batch_cmds  # useful for things like sourcing icc `source /opt/ipsxe/2019u1/bin/compilervars.sh intel64`
        self.envvars = envvars
        if not Run.is_run_folder_initialized:
            Run.setup_run_folder()
            Run.is_run_folder_initialized = True

    @staticmethod
    def setup_run_folder():
        makedirs("runs/submissions", exist_ok=True)
        makedirs("runs/stdouts", exist_ok=True)
        makedirs("runs/perfstats", exist_ok=True)
        makedirs("runs/slurmout", exist_ok=True)
        makedirs("runs/slurmerr", exist_ok=True)

    @property
    def run_name(self):
        underscored = lambda s: "_".join(s.split())
        return f"{self.name}_n{self.n}_steps{self.steps}_{underscored(self.cflags)}"

    @property
    def run_cmd(self):
        run_cmd = f"./headless {self.n} 0.1 0.0001 0.0001 5.0 100.0 {self.steps} > runs/stdouts/{self.run_name}.output"
        return run_cmd

    @property
    def perfstat_run_cmd(self):
        perfstat_cmd = f"perf stat -o runs/perfstats/{self.run_name}.output -e cache-references,cache-misses,L1-dcache-stores,L1-dcache-store-misses,LLC-stores,LLC-store-misses,page-faults,cycles,instructions,branches,branch-misses -ddd"
        perfstat_run_cmd = f"{perfstat_cmd} {self.run_cmd}"
        return perfstat_run_cmd

    def run(self):
        if Run.no_batch:
            self.run_inplace()
        else:
            self.run_sbatch()

    def run_inplace(self):
        envvars_cmds = " ".join(f"{k}={v}" for k, v in self.envvars.items())
        start_time = time()
        cmd(f"git checkout {self.branch_prefix}{self.name}")
        cmd("make clean")
        cmd(f"make headless CFLAGS='-g {self.cflags}'")
        if Run.should_run:
            cmd(f"{envvars_cmds} {self.perfstat_run_cmd}")
            printf(f">>> [TIME] Run finished in {time() - start_time} seconds.")

    def run_sbatch(self):
        submission_filename = f"runs/submissions/{self.run_name}.sh"
        envvars_cmds = ";".join(f"export {k}={v}" for k, v in self.envvars.items())
        submission_text = cleandoc(
            f"""
            #!/bin/bash
            #SBATCH --job-name={self.n}x{self.steps}
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=1
            #SBATCH --exclusive
            #SBATCH -o runs/slurmout/{self.run_name}.out
            #SBATCH -e runs/slurmerr/{self.run_name}.err

            # Environment variables initialization
            {envvars_cmds}

            # Previous batch cmds
            {self.previous_batch_cmds}

            # Actual run
            git checkout {self.branch_prefix}{self.name} &&
            make clean &&
            make headless CFLAGS='-g {self.cflags}' &&
            srun {self.perfstat_run_cmd} ||
            echo "If you see this file then your run with this filename had a problem, inspect runs/ folder for more information" > {self.run_name}.error # If this is in the project root then you know there was an error
        """
        )
        with open(submission_filename, "w") as submission:
            submission.write(submission_text)
        if Run.should_run:
            cmd(f"sbatch ./{submission_filename}")


def main():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-batch",
        "-nb",
        action="store_true",
        help="This flag will override the default behaviour of submiting to a slurm squeue, and instead directly run the commands",
    )
    parser.add_argument(
        "--dry-run",
        "-s",
        action="store_true",
        help="When set, runs the script without actually making the runs, just prints",
    )
    parsed_args = parser.parse_args()
    Run.no_batch = parsed_args.no_batch
    Run.should_run = not parsed_args.dry_run

    repo, initial_branch = save_git_state()
    itime = time()

    printf(">>> [START]")
    for n, steps in [(128, 512), (512, 128), (2048, 32), (4096, 16), (8192, 8)]:
        Run(
            "lab1",
            n,
            steps,
            envvars={
                "OMP_NUM_THREADS": 14,
                "OMP_PROC_BIND": "true",
                "OMP_PLACES": "{0:4:1}",
            },
        ).run()

    if Run.no_batch:
        restore_git_state(repo, initial_branch)
    else:
        # Batch command to restore git state after all batches
        cmd(
            f"nohup srun --job-name=cleanup -- git checkout {initial_branch} && git stash pop &"
        )
    printf(f"Done in {time() - itime} seconds with {cmd_error_count} errors.")


if __name__ == "__main__":
    main()
