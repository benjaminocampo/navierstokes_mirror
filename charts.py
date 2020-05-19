import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it
from os import makedirs
from os.path import exists
from itertools import count, cycle, islice

from run import Run

white = "#FFFFFF"
darkgray = "#263238"

plt.rcParams["font.family"] = "Fira Code"
plt.rcParams["text.color"] = darkgray
plt.rcParams["axes.labelcolor"] = darkgray
plt.rcParams["xtick.color"] = darkgray
plt.rcParams["ytick.color"] = darkgray

COLORS = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FFC107",  # amber
    "#E91E63",  # pink
    "#673AB7",  # deeppurple
    "#00BCD4",  # cyan
    "#CDDC39",  # lime
    "#FF5722",  # deeporange
    "#009688",  # teal
]

DARK_COLORS = [
    "#1976D2",  # dark blue
    "#388E3C",  # dark green
    "#FFA000",  # dark amber
    "#C2185B",  # dark pink
    "#512DA8",  # dark deeppurple
    "#0097A7",  # dark cyan
    "#AFB42B",  # dark lime
    "#E64A19",  # dark deeporange
    "#00796B",  # dark teal
]


def save_nscell_graph(
    names, means, ns, iterations, filename="plot.png", only_show=False
):
    CATEGORY_WIDTH_PX = 432  # Magic number that correlates to bar_width
    DPI = 150
    bar_fontsize = 12
    legend_fontsize = 6
    nof_runs = len(names)
    nof_groups = len(ns)
    img_width = 96 * nof_groups * nof_runs
    img_height = 720
    bar_width = CATEGORY_WIDTH_PX / img_width

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="center",
                color=white,
                fontsize=bar_fontsize,
            )

    x = np.arange(nof_groups)  # the label locations
    fig, ax = plt.subplots(figsize=(img_width / DPI, img_height / DPI), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(darkgray)
    ax.spines["left"].set_color(darkgray)

    rects = []
    starting_x = x - nof_runs / 2 + 0.5
    for i, name, mean, color in zip(count(), names, means, cycle(COLORS)):
        rect = ax.bar(
            starting_x + bar_width * i,
            mean.astype(int),
            bar_width,
            label=name,
            color=color,
        )
        rects.append(rect)
        autolabel(rect)

    ax.set_ylabel("Nanoseconds per Cell")
    ax.set_title("Compute Time per Cell")
    plt.figtext(
        0.965,
        0.5,
        "Lower is better",
        ha="center",
        va="center",
        fontsize=legend_fontsize,
        rotation=-90,
    )
    tick_xs = [(i - nof_runs / 2 + 0.5) + (nof_runs - 1) * (bar_width / 2) for i in x]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, iterations)])
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename)
    plt.close()


def save_cache_graph(
    names,
    refss,
    missess,
    ns,
    iterations,
    filename="plot.png",
    cache_name="L1",
    only_show=False,
):
    CATEGORY_WIDTH_PX = 432  # Magic number that correlates to bar_width
    DPI = 150
    bar_fontsize = 10
    legend_fontsize = 6
    nof_runs = len(names)
    nof_groups = len(ns)
    img_width = 96 * nof_groups * nof_runs
    img_height = 720
    bar_width = CATEGORY_WIDTH_PX / img_width

    def autolabel(rects, total=False):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{human_readable(height if not total else height + rect.get_y())}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height),
                xytext=(0, -16 if not total else 8),
                textcoords="offset points",
                ha="center",
                va="center",
                color=white if not total else darkgray,
                fontsize=bar_fontsize,
            )

    def human_readable(number):
        multiples = [1e3, 1e6, 1e9, 1e12]
        units = ["K", "M", "G", "T"]
        multiples.reverse()
        units.reverse()
        unit = ""
        for m, u in zip(multiples, units):
            if number >= m:
                number /= m
                unit = u
        if number >= 100:
            return f"{number:.0f}{unit}"
        elif number >= 10:
            return f"{number:.1f}{unit}"
        else:
            return f"{number:.2f}{unit}"

    x = np.arange(nof_groups)  # the label locations
    hitss = [refs - misses for refs, misses in zip(refss, missess)]
    scaled_hitss = [hits / ((ns ** 2) * iterations) for hits in hitss]
    scaled_missess = [misses / ((ns ** 2) * iterations) for misses in missess]

    fig, ax = plt.subplots(figsize=(img_width / DPI, img_height / DPI), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(darkgray)
    ax.spines["left"].set_color(darkgray)

    starting_x = x - nof_runs / 2 + 0.5
    misses_bars = []
    hits_bars = []
    cyclec = lambda c: islice(cycle(c), 2 if cache_name == "L1" else 4, None)
    for i, name, misses, hits, color, dark_color in zip(
        count(),
        names,
        scaled_missess,
        scaled_hitss,
        cyclec(COLORS),
        cyclec(DARK_COLORS),
    ):
        hit_bar = ax.bar(
            starting_x + bar_width * i,
            hits,
            bar_width,
            label=f"{name} hits",
            color=color,
        )
        hits_bars.append(hit_bar)
        autolabel(hit_bar)
        miss_bar = ax.bar(
            starting_x + bar_width * i,
            misses,
            bar_width,
            bottom=hits,
            label=f"{name} misses",
            color=dark_color,
        )
        misses_bars.append(miss_bar)
        autolabel(miss_bar, True)

    ax.set_ylabel(f"{cache_name} References")
    ax.set_title(f"Cache {cache_name} References per Cell Iteration")
    plt.figtext(
        0.965,
        0.5,
        "Lower is better",
        ha="center",
        va="center",
        fontsize=legend_fontsize,
        rotation=-90,
    )
    tick_xs = [(i - nof_runs / 2 + 0.5) + (nof_runs - 1) * (bar_width / 2) for i in x]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, iterations)])
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename)
    plt.close()


def save_scaling_graph(
    run_measurements, ns, steps, filename="plot.png", only_show=False
):
    DPI = 120
    img_width = 960 * 2
    img_height = 720 * 2
    fig, ax = plt.subplots(figsize=(img_width / DPI, img_height / DPI), dpi=DPI)

    colors = cycle(COLORS)
    markers = cycle(["o", "v", "s", "D", "H"])
    for cores, runs in run_measurements.items():
        nspcells = [-measurement["nspcell_mean"] for measurement in runs]
        ax.plot(ns, nspcells, marker=next(markers), label=f"{cores} threads", color=next(colors))
    plt.legend(loc="lower left")
    plt.grid(linestyle='-', linewidth=0.125)
    ax.set_title(f"Raw Scaling")
    ax.set_ylabel("Inverted Nanoseconds per Cell")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, steps)])
    ax.set_yticks(np.concatenate((ax.get_yticks(), ax.get_yticks() - 10)))

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename)


def read_perfstats(filename, stats, cast_to=int):
    "Given stats=[stat] returns { stat: value } with values read from file"
    regex = lambda s: rf"^\s*([\d\.,]+|.not supported.)\s*{s}.*$"
    with open(filename, "r") as file:
        content = file.read()
        read_stats = {}
        for stat in stats:
            matches = [
                a.groups()[0] for a in re.finditer(regex(stat), content, re.MULTILINE)
            ]
            assert (
                len(matches) == 1
            ), f"matches={matches} are not exactly one. stat={stat} searched in filename={filename}"
            read_stats[stat] = cast_to(matches[0])
    return read_stats


def get_run_measurement(run: Run):
    # Returns for a given existing measured run, the next dictionary
    # {
    #     "nspcell_mean": nspcell_mean ,
    #     "cache-references": cache_references,
    #     "cache-misses": cache_misses,
    #     "L1-dcache-loads": L1_dcache_loads,
    #     "L1-dcache-load-misses": L1_dcache_load_misses,
    # }
    filename = f"{run.run_name}.output"
    stdout_file = f"runs/stdouts/{filename}"
    perfstat_file = f"runs/perfstats/{filename}"
    stats = [
        "cache-references",
        "cache-misses",
        "L1-dcache-loads",
        "L1-dcache-load-misses",
    ]
    if exists(stdout_file):
        nspcells = pd.read_csv(f"runs/stdouts/{filename}")
    else:
        print(f"Stdout file does not exists, will use zero instead: {stdout_file}")
    if exists(perfstat_file):
        perfstats = read_perfstats(f"runs/perfstats/{filename}", stats)
    else:
        perfstats = {stat: 0 for stat in stats}
        print(f"Perfstat file does not exist, will use zero instead: {perfstat_file}")
    run_measurement = {}
    run_measurement["nspcell_mean"] = np.mean(nspcells["total_ns"])
    run_measurement.update(perfstats)
    return run_measurement


def scaling_main():
    RUN_NAME = "lab3"  # The run name to make the scaling graph against
    make_run = lambda n, steps, cores: Run(
        RUN_NAME,
        n,
        steps,
        envvars={
            "OMP_NUM_THREADS": cores,
            "OMP_PROC_BIND": "true",
            "BUILD": "intrinsics",
            "CFLAGS": "-fopenmp",
        },
    )
    makedirs("runs/graphs", exist_ok=True)
    ns = np.array([128, 512, 2048, 4096, 8192])
    steps = np.array([512, 128, 32, 16, 8])
    cores = [1, 2, 4, 8, 14, 16, 21, 24, 28]
    plotpath = "runs/graphs"
    plotid = RUN_NAME

    # in the next part, run_measurements is a dict with the following structure
    # {corecount: [
    #     { # A dict for  each (n, step) pair
    #          "nspcell_mean": nspcell_mean ,
    #          "cache-references": cache_references,
    #          "cache-misses": cache_misses,
    #          "L1-dcache-loads": L1_dcache_loads,
    #          "L1-dcache-load-misses": L1_dcache_load_misses,
    #      }
    # ]}

    run_measurements = {corecount: [] for corecount in cores}
    for (n, step), corecount in it.product(zip(ns, steps), cores):
        run = make_run(n, step, corecount)
        run_measurement = get_run_measurement(run)
        run_measurements[corecount].append(run_measurement)

    save_scaling_graph(
        run_measurements,
        ns,
        steps,
        filename=f"{plotpath}/scaling__{plotid}.png",
        only_show=True,
    )


def main():
    makedirs("runs/graphs", exist_ok=True)
    ns = np.array([128, 512, 2048, 4096, 8192])
    steps = np.array([512, 128, 32, 16, 8])

    make_run = lambda name, n, steps: Run(
        name, n, steps, envvars={"BUILD": "intrinsics"}, branch_prefix="l2/intrinsics/",
    )

    # array of run names
    runs = [
        "project",
        "linsolve",
        "baseline",
        "rb",
        "lab1",
        "shload",
    ]

    run_measurements = {}
    # run_mesaurements is a dict with the following structure
    # { name: [
    #     { # A dict for  each (n, step) pair
    #          "nspcell_mean": nspcell_mean ,
    #          "cache-references": cache_references,
    #          "cache-misses": cache_misses,
    #          "L1-dcache-loads": L1_dcache_loads,
    #          "L1-dcache-load-misses": L1_dcache_load_misses,
    #      }
    # ] }
    for name in runs:
        run_measurements[name] = []
        for n, step in zip(ns, steps):
            run = make_run(name, n, step)
            filename = f"{run.run_name}.output"
            stdout_file = f"runs/stdouts/{filename}"
            perfstat_file = f"runs/perfstats/{filename}"
            stats = [
                "cache-references",
                "cache-misses",
                "L1-dcache-loads",
                "L1-dcache-load-misses",
            ]
            if exists(stdout_file):
                nspcells = pd.read_csv(f"runs/stdouts/{filename}")
            else:
                print(
                    f"Stdout file does not exists, will use zero instead: {stdout_file}"
                )
            if exists(perfstat_file):
                perfstats = read_perfstats(f"runs/perfstats/{filename}", stats)
            else:
                perfstats = {stat: 0 for stat in stats}
                print(
                    f"Perfstat file does not exist, will use zero instead: {perfstat_file}"
                )
            run_measurement = {}
            run_measurement["nspcell_mean"] = np.mean(nspcells["total_ns"])
            run_measurement.update(perfstats)
            run_measurements[name].append(run_measurement)

    comparissons = [
        ("baseline", "ijswap"),
        ("linsolve", "project"),
        ("linsolve", "project", "shload"),
        ("linsolve", "project", "linsolve", "project"),
        ("linsolve", "project", "linsolve", "project", "linsolve"),
    ]
    for comparisson in comparissons:
        names = comparisson
        plotpath = "runs/graphs"
        plotid = "__vs__".join(names)

        means = []
        l1_refs = []
        l1_misses = []
        llc_refs = []
        llc_misses = []

        for run in comparisson:
            means.append(np.array([e["nspcell_mean"] for e in run_measurements[run]]))
            l1_refs.append(
                np.array([e["L1-dcache-loads"] for e in run_measurements[run]])
            )
            l1_misses.append(
                np.array([e["L1-dcache-load-misses"] for e in run_measurements[run]])
            )
            llc_refs.append(
                np.array([e["cache-references"] for e in run_measurements[run]])
            )
            llc_misses.append(
                np.array([e["cache-misses"] for e in run_measurements[run]])
            )

        save_nscell_graph(
            names,
            means,
            ns,
            steps,
            f"{plotpath}/nspcellgraph__{plotid}.png",
            only_show=True,
        )
        save_cache_graph(
            names,
            l1_refs,
            l1_misses,
            ns,
            steps,
            filename=f"{plotpath}/l1graph__{plotid}.png",
            cache_name="L1",
            only_show=True,
        )
        save_cache_graph(
            names,
            llc_refs,
            llc_misses,
            ns,
            steps,
            filename=f"{plotpath}/llcgraph__{plotid}.png",
            cache_name="LLC",
            only_show=True,
        )


if __name__ == "__main__":
    scaling_main()
