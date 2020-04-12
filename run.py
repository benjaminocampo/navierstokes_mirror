import os
from os import popen
from os import chdir
from os.path import isdir
from time import time

SHOULD_RUN = True # Generates run.output and perfstat.output
SHOULD_PERFRECORD = False

printf = lambda s: print(s, flush=True)

error_count = 0
def cmd(c):
    global error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(c):
        printf(f"{red_color}>>> [ERROR] there was an error in command:{end_color}")
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        exit()
        error_count += 1

def run(branch, flags, n, steps):
    run_cmd = f"./headless {n} 0.1 0.001 0.0001 5.0 100.0 {steps} > run.output"
    perfstat_cmd = f"perf stat -o perfstat.output -e cache-references,cache-misses,L1-dcache-stores,L1-dcache-store-misses,LLC-stores,LLC-store-misses,page-faults,cycles,instructions,branches,branch-misses -ddd"
    perfstat_run_cmd = f"{perfstat_cmd} {run_cmd}"
    perfrecord_cmd = f"perf record -g {run_cmd}"

    underscored = lambda s: "_".join(s.split())
    start_time = time()
    directory = f"{branch}_n{n}_steps{steps}_{underscored(flags)}"
    if not isdir(f"./{directory}"):
        cmd(f"cp -r navierstokes {directory}")
    chdir(directory)
    cmd(f"git checkout l1-{branch}")
    cmd("make clean")
    cmd(f"make headless CFLAGS='-g {flags}'")
    if (SHOULD_RUN): cmd(perfstat_run_cmd)
    elif (SHOULD_PERFRECORD): cmd(perfrecord_cmd)
    chdir("..")
    printf(f">>> [TIME] Run finished in {time() - start_time} seconds.")


itime = time()
printf(">>> [START]")
cmd("git clone https://github.com/mateosss/navierstokes")
for n, steps in [(2048, 32), (512, 128), (128, 512)]:
    # baseline -O0, -O1, -O2, -O3, -Ofast, Os
    # baseline (-O3) -> baseline (-O3 -floop-interchange -floop-nest-optimize)
    # baseline (-O3) -> ijswap (-O3) (mostrar perf stat cache references)
    run("baseline", "-O0", n, steps)
    run("baseline", "-O1", n, steps)
    run("baseline", "-O2", n, steps)
    run("baseline", "-O3", n, steps)
    run("baseline", "-Ofast", n, steps)
    run("baseline", "-Os", n, steps)
    run("baseline", "-O3 -floop-interchange -floop-nest-optimize", n, steps)
    run("ijswap", "-O3", n, steps)

    # ijswap (-O3) -> invc (-O3)
    # ijswap (-O3 -freciprocal-math) -> ijswap (-Ofast) [Ver si ofast nos da alguna otra ventaja que no notamos ademas del reciprocal]
    # invc (-O3) -> ijswap (-Ofast) [nuestra optimizacion permitio no meter otras flags peligrosas?]
    run("invc", "-O3", n, steps)
    run("ijswap", "-O3 -freciprocal-math", n, steps)
    run("ijswap", "-Ofast", n, steps)

    # invc (-Ofast)
    # invc (-Ofast -march=native)
    # invc (-Ofast -march=native -funroll-loops -floop-nest-optimize) [con estas nada]
    # invc (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [y con flto tampoco]
    # constn2048 (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [pero si ponemos const n una banda]
    # diffvisc0 (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [una banda mas con mas constantes]
    run("invc", "-Ofast", n, steps)
    run("invc", "-Ofast -march=native", n, steps)
    run("invc", "-Ofast -march=native -funroll-loops", n, steps)
    run("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize", n, steps)
    run("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

    run("bblocks", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

    run(f"constn{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
    run(f"zdiffvisc{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)

printf(f"Done in {time() - itime} seconds with {error_count} errors.")
