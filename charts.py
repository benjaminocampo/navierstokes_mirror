import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import makedirs
from os.path import exists
from itertools import count, cycle

white = "#FFFFFF"
darkgray = "#263238"

plt.rcParams["font.family"] = "Fira Code"
plt.rcParams["text.color"] = darkgray
plt.rcParams["axes.labelcolor"] = darkgray
plt.rcParams["xtick.color"] = darkgray
plt.rcParams["ytick.color"] = darkgray

COLORS = [
    "#2196F3", # blue
    "#4CAF50", # green
    "#FFC107", # amber
    "#E91E63", # pink
    "#673AB7", # deeppurple
    "#00BCD4", # cyan
    "#CDDC39", # lime
    "#FF5722", # deeporange
]

DARK_COLORS = [
    "#1976D2", # dark blue
    "#388E3C", # dark green
    "#FFA000", # dark amber
    "#C2185B", # dark pink
    "#512DA8", # dark deeppurple
    "#0097A7", # dark cyan
    "#AFB42B", # dark lime
    "#E64A19", # dark deeporange
]

def save_nscell_graph(names, means, ns, iterations, filename="plot.png", only_show=False):
    CATEGORY_WIDTH_PX = 432 # Magic number that correlates to bar_width
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
            ax.annotate("{}".format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -16),
                        textcoords="offset points",
                        ha="center", va="center", color=white, fontsize=bar_fontsize)

    x = np.arange(nof_groups)  # the label locations
    fig, ax = plt.subplots(figsize=(img_width / DPI, img_height / DPI), dpi=DPI)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(darkgray)
    ax.spines["left"].set_color(darkgray)

    rects = []
    starting_x = x - nof_runs / 2 + 0.5
    for i, name, mean, color in zip(count(), names, means, cycle(COLORS)):
        rect = ax.bar(starting_x + bar_width * i, mean.astype(int), bar_width, label=name, color=color)
        rects.append(rect)
        autolabel(rect)

    ax.set_ylabel("Nanoseconds per Cell")
    ax.set_title("Compute Time per Cell")
    plt.figtext(0.965, 0.5, 'Lower is better', ha='center', va="center", fontsize=legend_fontsize, rotation=-90)
    tick_xs = [(i - nof_runs / 2 + 0.5) + (nof_runs - 1) * (bar_width / 2) for i in x]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, iterations)])
    ax.legend(loc="lower left", fontsize=legend_fontsize)

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename)
    plt.close()

def save_cache_graph(source_name, target_name, source_refs, source_misses, target_refs, target_misses, ns, iterations, filename="plot.png", cache_name="L1", only_show=False):
    source_color = "#FFC107" if cache_name == "L1" else "#00BCD4"
    source_color_dark = "#FFA000" if cache_name == "L1" else "#0097A7"
    target_color = "#E91E63" if cache_name == "L1" else "#673AB7"
    target_color_dark = "#C2185B" if cache_name == "L1" else "#512DA8"
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
        return f"{number:.0f}{unit}" if number >= 100 else f"{number:.2f}{unit}"

    x = np.arange(len(ns))  # the label locations
    width = 0.45  # the width of the bars

    source_hits = source_refs - source_misses
    target_hits = target_refs - target_misses

    scaled_source_hits = source_hits / ((ns**2) * iterations)
    scaled_source_misses = source_misses / ((ns**2) * iterations)
    scaled_target_hits = target_hits / ((ns**2) * iterations)
    scaled_target_misses = target_misses / ((ns**2) * iterations)


    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(darkgray)
    ax.spines["left"].set_color(darkgray)
    source_misses_bar = ax.bar(x - width/2, scaled_source_misses, width, bottom=scaled_source_hits, label=f"{source_name} misses", color=source_color_dark)
    source_hits_bar = ax.bar(x - width/2, scaled_source_hits, width, label=f"{source_name} hits", color=source_color)
    target_misses_bar = ax.bar(x + width/2, scaled_target_misses, width, bottom=scaled_target_hits, label=f"{target_name} misses", color=target_color_dark)
    target_hits_bar = ax.bar(x + width/2, scaled_target_hits, width, label=f"{target_name} hits", color=target_color)
    ax.set_ylabel(f"{cache_name} References")
    ax.set_title(f"Cache {cache_name} References per Cell Iteration")
    plt.figtext(0.965, 0.5, 'Lower is better', ha='center', va="center", fontsize=6, rotation=-90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, iterations)])
    ax.legend(loc="lower left", fontsize=6)


    def autolabel(rects, total=False):
        """Attach a text label above each bar in *rects*, displaying its height."""
        FONTSIZE = 10
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{human_readable(height if not total else height + rect.get_y())}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height),
                        xytext=(0, -16 if not total else 8),
                        textcoords="offset points",
                        ha="center", va="center", color=white if not total else darkgray, fontsize=FONTSIZE)

    autolabel(source_hits_bar)
    autolabel(source_misses_bar, True)
    autolabel(target_hits_bar)
    autolabel(target_misses_bar, True)

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename, dpi=150)
    plt.close()

def read_perfstats(filename, stats, cast_to=int):
    "Given stats=[stat] returns { stat: value } with values read from file"
    regex = lambda s : rf'^\s*([\d\.,]+|.not supported.)\s*{s}.*$'
    with open(filename, "r") as file:
        content = file.read()
        read_stats = {}
        for stat in stats:
            matches = [a.groups()[0] for a in re.finditer(regex(stat), content, re.MULTILINE)]
            assert len(matches) == 1, f"matches={matches} are not exactly one. stat={stat} searched in filename={filename}"
            read_stats[stat] = cast_to(matches[0])
    return read_stats

def main():
    makedirs("runs/graphs", exist_ok=True)

    ns = np.array([128, 512, 2048, 4096, 8192])
    steps = np.array([512, 128, 32, 16, 8])
    # array of (branch, flags)
    runs = [
        ("project", ""),
        ("linsolve", ""),
        ("baseline", ""),
        ("rb", ""),
        ("lab1", ""),
        ("shload", ""),
    ]


    underscored = lambda s: "_".join(s.split())

    run_measuremets = {}
    for branch, flags in runs:
        run_measuremets[(branch, flags)] = []
        for n, step in zip(ns, steps):
            filename = f"{branch}_n{n}_steps{step}_{underscored(flags)}.output"
            stdout_file = f"runs/stdouts/{filename}"
            perfstat_file = f"runs/perfstats/{filename}"
            stats = ["cache-references", "cache-misses", "L1-dcache-loads", "L1-dcache-load-misses"]
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
            run_measuremets[(branch, flags)].append(run_measurement)

    comparissons = [
        # (("source_branch", "source_flags"), ("target_branch", "target_flags")),
        (("baseline", "-O3"), ("ijswap", "-O3")),
        (("linsolve", ""), ("project", ""),),
        (("linsolve", ""), ("project", ""), ("shload", "")),
        (("linsolve", ""), ("project", ""), ("linsolve", ""), ("project", "")),
        (("linsolve", ""), ("project", ""), ("linsolve", ""), ("project", ""), ("linsolve", ""))
    ]
    for comparisson in comparissons:
        names = [" ".join(run) for run in comparisson]
        plotpath = "runs/graphs"
        plotid = "__vs__".join(names)

        means = []
        l1_refs = []
        l1_misses = []
        llc_refs = []
        llc_misses = []

        for run in comparisson:
            means.append(np.array([e["nspcell_mean"] for e in run_measuremets[run]]))
            l1_refs.append(np.array([e["L1-dcache-loads"] for e in run_measuremets[run]]))
            l1_misses.append(np.array([e["L1-dcache-load-misses"] for e in run_measuremets[run]]))
            llc_refs.append(np.array([e["cache-references"] for e in run_measuremets[run]]))
            llc_misses.append(np.array([e["cache-misses"] for e in run_measuremets[run]]))

        save_nscell_graph(names, means, ns, steps, f"{plotpath}/nspcellgraph__{plotid}.png", only_show=False)

if __name__ == "__main__":
    main()
