import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import makedirs

# # Chart 1: ns/cell comparisson

# baseline -O3 N=128 L1 ns/cell
# ijswap -O3 N=128 L1 ns/cell

# baseline -O3 N=512 L1 ns/cell
# ijswap -O3 N=512 L1 ns/cell

# baseline -O3 N=2048 L1 ns/cell
# ijswap -O3 N=2048 L1 ns/cell

# # Chart 2: L1 Cache

# baseline -O3 N=128 (L1 refs + L1 hits) / N
# ijswap -O3 N=128 (L1 refs + L1 hits) / N

# baseline -O3 N=512 (L1 refs + L1 hits) / N
# ijswap -O3 N=512 (L1 refs + L1 hits) / N

# baseline -O3 N=2048 (L1 refs + L1 hits) / N
# ijswap -O3 N=2048 (L1 refs + L1 hits) / N

# # Chart 3: LLC Cache

# baseline -O3 N=128 (LLC refs + LLC hits) / N
# ijswap -O3 N=128 (LLC refs + LLC hits) / N

# baseline -O3 N=512 (LLC refs + LLC hits) / N
# ijswap -O3 N=512 (LLC refs + LLC hits) / N

# baseline -O3 N=2048 (LLC refs + LLC hits) / N
# ijswap -O3 N=2048 (LLC refs + LLC hits) / N

# TODO: Set colors: https://coolors.co/263238-e63462-fe5f55-3a506b-5bc0be
# TODO: Set font
# TODO: Draw graph 2
# TODO: Draw graph 3
# TODO: Get data for graph 1
# TODO: Get data for graph 2
# TODO: Get data for graph 3

white = "#FFFFFF"
darkgray = "#263238"

plt.rcParams["font.family"] = "Fira Code"
plt.rcParams["text.color"] = darkgray
plt.rcParams["axes.labelcolor"] = darkgray
plt.rcParams["xtick.color"] = darkgray
plt.rcParams["ytick.color"] = darkgray

def save_nscell_graph(source_name, target_name, sources, targets, ns, iterations, filename="plot.png", only_show=False):
    source_color = "#2196F3"
    source_color_dark = "#1976D2"
    target_color = "#4CAF50"
    target_color_dark = "#388E3C"
    x = np.arange(len(ns))  # the label locations
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(darkgray)
    ax.spines["left"].set_color(darkgray)
    rects1 = ax.bar(x - width/2, sources.astype(int), width, label=source_name, color=source_color)
    rects2 = ax.bar(x + width/2, targets.astype(int), width, label=target_name, color=target_color)
    ax.set_ylabel("Nanoseconds per Cell")
    ax.set_title("Compute Time per Cell")
    plt.figtext(0.965, 0.5, 'Lower is better', ha='center', va="center", fontsize=6, rotation=-90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={s} x{i}" for s, i in zip(ns, iterations)])
    ax.legend(loc="lower left", fontsize=6)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        FONTSIZE = 12
        for rect in rects:
            height = rect.get_height()
            ax.annotate("{}".format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height - FONTSIZE * 4 - 16),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="center", color=white, fontsize=12)


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show() if only_show else plt.savefig(filename, dpi=150)


# baseline -O3 N=128 (L1 refs + L1 hits) / N
# ijswap -O3 N=128 (L1 refs + L1 hits) / N

# baseline -O3 N=512 (L1 refs + L1 hits) / N
# ijswap -O3 N=512 (L1 refs + L1 hits) / N

# baseline -O3 N=2048 (L1 refs + L1 hits) / N
# ijswap -O3 N=2048 (L1 refs + L1 hits) / N
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
        FONTSIZE = 12
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

    ns = np.array([128, 512, 2048])
    steps = np.array([512, 128, 32])
    # array of (branch, flags)
    runs = [
        ("baseline", "-O3"),
        ("ijswap", "-O3"),
        ("baseline", "-O3 -floop-interchange -floop-nest-optimize"),
        ("invc", "-O3"),
        ("ijswap", "-Ofast"),
        ("ijswap", "-O3 -freciprocal-math"),
        ("invc", "-Ofast"),
        ("invc", "-Ofast -march=native"),
        ("invc", "-Ofast -march=native -funroll-loops"),
        ("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize"),
        ("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto"),
        # TODO # ("constn2048", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto"),
    ]


    underscored = lambda s: "_".join(s.split())

    run_measuremets = {}
    for branch, flags in runs:
        run_measuremets[(branch, flags)] = []
        for n, step in zip(ns, steps):
            filename = f"{branch}_n{n}_steps{step}_{underscored(flags)}.output"
            nspcells = pd.read_csv(f"runs/stdouts/{filename}")
            run_measurement = {}
            run_measurement["nspcell_mean"] = np.mean(nspcells["total_ns"])
            perfstats = read_perfstats(f"runs/perfstats/{filename}", ["cache-references", "cache-misses", "L1-dcache-loads", "L1-dcache-load-misses"])
            run_measurement.update(perfstats)
            run_measuremets[(branch, flags)].append(run_measurement)

    from pprint import pprint
    pprint(run_measuremets)

    comparissons = [
        # (("source_branch", "source_flags"), ("target_branch", "target_flags")),
        (("baseline", "-O3"), ("ijswap", "-O3")),
        (("baseline", "-O3"), ("baseline", "-O3 -floop-interchange -floop-nest-optimize")),
        (("ijswap", "-O3"), ("invc", "-O3")),
        (("invc", "-O3"), ("ijswap", "-Ofast")),
        (("ijswap", "-O3 -freciprocal-math"), ("ijswap", "-Ofast")),
        (("invc", "-Ofast"), ("invc", "-Ofast -march=native")),
        (("invc", "-Ofast -march=native"), ("invc", "-Ofast -march=native -funroll-loops")),
        (("invc", "-Ofast -march=native -funroll-loops"), ("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize")),
        (("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize"), ("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto")),
        # TODO # (("invc", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto"), ("constn2048", "-Ofast -march=native -funroll-loops -floop-nest-optimize -flto")),
    ]
    for source_run, target_run in comparissons:
        source_name = " ".join(source_run)
        target_name = " ".join(target_run)
        plotpath = "runs/graphs"
        plotid = f"{source_name}__vs__{target_name}"

        source_means = np.array([e["nspcell_mean"] for e in run_measuremets[source_run]])
        source_l1_refs = np.array([e["L1-dcache-loads"] for e in run_measuremets[source_run]])
        source_l1_misses = np.array([e["L1-dcache-load-misses"] for e in run_measuremets[source_run]])
        source_llc_refs = np.array([e["cache-references"] for e in run_measuremets[source_run]])
        source_llc_misses = np.array([e["cache-misses"] for e in run_measuremets[source_run]])
        target_means = np.array([e["nspcell_mean"] for e in run_measuremets[target_run]])
        target_l1_refs = np.array([e["L1-dcache-loads"] for e in run_measuremets[target_run]])
        target_l1_misses = np.array([e["L1-dcache-load-misses"] for e in run_measuremets[target_run]])
        target_llc_refs = np.array([e["cache-references"] for e in run_measuremets[target_run]])
        target_llc_misses = np.array([e["cache-misses"] for e in run_measuremets[target_run]])

        save_nscell_graph(source_name, target_name, source_means, target_means, ns, steps, f"{plotpath}/nspcellgraph__{plotid}.png", only_show=False)
        save_cache_graph(source_name, target_name, source_l1_refs, source_l1_misses, target_l1_refs, target_l1_misses, ns, steps, filename=f"{plotpath}/l1graph__{plotid}.png", cache_name="L1", only_show=False)
        save_cache_graph(source_name, target_name, source_llc_refs, source_llc_misses, target_llc_refs, target_llc_misses, ns, steps, filename=f"{plotpath}/llcgraph__{plotid}.png", cache_name="LLC", only_show=False)

if __name__ == "__main__":
    main()
