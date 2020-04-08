import os
from os import popen
from os import chdir
from time import time
import pdb

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

def run(source, sflags, target, tflags, n, steps):
    underscored = lambda s: "_".join(s.split())
    cmditime = time()
    sdirectory = f"{source}_n{n}_steps{steps}_{underscored(sflags)}"
    tdirectory = f"{target}_n{n}_steps{steps}_{underscored(tflags)}"
    sfile = f"../runs/{sdirectory}/run.output"
    tfile = f"../runs/{tdirectory}/run.output"
    comparison = f"{sdirectory}_vs_{tdirectory}"
    cmd(f"mkdir {comparison}")
    chdir(comparison)
    sflags_label = sflags.replace(" ", "\n")
    tflags_label = tflags.replace(" ", "\n")
    cmd(f"python3 ../fcompare.py -sd {sfile} -td {tfile} -sl '{source} {sflags_label}' -tl '{target} {tflags_label}' -p plot > compare.output")
    
    chdir("..")
    printf(f">>> [TIME] Run finished in {time() - cmditime} seconds.")


itime = time()
printf(">>> [START]")
for n, steps in [(2048, 32), (512, 128), (128, 512)]:
    ## baseline -O0, -O1, -O2, -O3, -Ofast, Os
    #run("baseline", "-O0", "baseline", "-O1", n, steps)
    #run("baseline", "-O1", "baseline", "-O2",n, steps)
    #run("baseline", "-O2", "baseline", "-O3", n, steps)
    #run("baseline", "-O3", "baseline", "-Ofast", n, steps)
    #run("baseline", "-Ofast", "baseline", "-Os", n, steps)
#
    ## baseline (-O3) -> baseline (-O3 -floop-interchange -floop-nest-optimize)
    ## baseline (-O3) -> ijswap (-O3) (mostrar perf stat cache references)
    #run("baseline", "-O3", "baseline", "-O3 -floop-interchange -floop-nest-optimize", n, steps)
    #run("baseline","-O3", "ijswap", "-O3", n, steps)
    ##cmd(f"python3 fcompare.py")
    ## ijswap (-O3) -> invc (-O3)
    ## ijswap (-O3 -freciprocal-math) -> ijswap (-Ofast) [Ver si ofast nos da alguna otra ventaja que no notamos ademas del reciprocal]
    ## invc (-O3) -> ijswap (-Ofast) [nuestra optimizacion permitio no meter otras flags peligrosas?]
    #run("ijswap", "-O3", "invc", "-O3", n, steps)
    #run("ijswap", "-O3 -freciprocal-math", "ijswap", "-Ofast", n, steps)
    #run("invc", "-O3", "ijswap", "-Ofast", n, steps)
#
    ## invc (-Ofast)
    ## invc (-Ofast -march=native)
    ## invc (-Ofast -march=native -funroll-loops -floop-nest-optimize) [con estas nada]
    ## invc (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [y con flto tampoco]
    ## constn2048 (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [pero si ponemos const n una banda]
    ## diffvisc0 (-Ofast -march=native -funroll-loops -floop-nest-optimize -flto) [una banda mas con mas constantes]
    #run("invc", "-Ofast", "invc", "-Ofast -march=native", n, steps)
    #run("invc", "-Ofast -march=native", "invc", "-Ofast -march=native -funroll-loops", n, steps)
    #run("invc", "-Ofast -march=native -funroll-loops", "invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize", n, steps)
    #run("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize", "invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
    #run("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", f"constn{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
    #run(f"constn{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", f"zdiffvisc{n}", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
#
    #run("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", "bblocks", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)
#
    #run("bblocks", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", "baseline", "-O0", n, steps)
    run("baseline", "-O0", "bblocks", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto", n, steps)



#
printf(f"Done in {time() - itime} seconds with {error_count} errors.")
