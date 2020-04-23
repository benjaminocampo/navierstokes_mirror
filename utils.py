import subprocess
import shlex
import git
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO, TextIOWrapper
from pathlib import Path

def dump_data(mu0,
              observed_value,
              p_value, ratio,
              significance_level,
              source,
              target):
    if p_value < significance_level:
        print(f"P value: {p_value} < {significance_level}")
        print("Assert in favour of the new approach.")
    else:
        print(f"P value: {p_value} >= {significance_level}")
        print("No evidence to reject the previous approach.")
    print(f"Source {source} mean:", mu0)
    print(f"Target {target} mean:", observed_value)
    print(f"Percentage improved from source to target: {ratio:.2f}%")


def test_hypotheses(sample_hyp1, sample_hyp2):
    """
    Use a ztest procedure to decide if the observed value
    given by sample_hyp2 rejects the null hypothesis.

    Parameters
    -------
    sample_hyp1: Panda Series object
        sample of null hypothesis.
    mnts_hyp2: Panda Series object
        sample of the observed value.

    Returns
    -------
    4-Upla:(
        mu0: float,
        observed_value: float,
        p_value: float,
        ratio: float)
    """

    mu0 = sample_hyp1.mean()
    observed_value = sample_hyp2.mean()
    test_statistic, p_value = sm.stats.ztest(
        sample_hyp2,
        value=mu0,
        alternative='smaller')
    ratio = ((mu0 - observed_value)/mu0)*100
    return (mu0, observed_value, p_value, ratio)


def get_data_from_stdin(make_cmds, exec_cmds):
    """
    Run the procedures given by make_cmds and exec_cmds.
    The output information is sent through stdin
    to this file (utils.py) and parsed as a Panda DataFrame.

    Parameters
    -------
    make_cmds: str
        make commands separated by spaces
    exec_cmds: str
        executable commands separated by spaces
    Returns
    -------
    data: Panda DataFrame object
        measurements obtained by stdout after running exec_cmds.
    """
    subprocess.run(['make', 'clean'])
    print(f">>> [MAKE] {make_cmds}")
    subprocess.run(shlex.split(make_cmds))
    print(f">>> [EXEC] {exec_cmds}")
    run = subprocess.Popen(shlex.split(exec_cmds), stdout=subprocess.PIPE)

    # Read while printing output
    output = StringIO()
    for line in TextIOWrapper(run.stdout, encoding="utf-8", line_buffering=True, write_through=True):
        output.write(line)
        print(line, end="", flush=True)
    output.seek(0)

    data = pd.read_csv(output)
    return data

def save_git_state():
    repo_abs_path = str(Path(".").resolve())
    repo = git.Repo(repo_abs_path)
    initial_branch = repo.active_branch.name
    repo.__utils_was_dirty = repo.is_dirty()
    if repo.__utils_was_dirty:
        repo.git.stash()
    return (repo, initial_branch)

def restore_git_state(repo, initial_branch):
    repo.git.checkout(initial_branch)
    if repo.__utils_was_dirty:
        repo.git.stash("pop")
    repo.__utils_was_dirty = False

def plot_charts(plotfile,
                mu0,
                observed_value,
                N,
                steps,
                p_value,
                ratio,
                source_label,
                target_label):
    bars = [f"Source:\n{source_label}", f"Target:\n{target_label}"]
    y_pos = np.arange(len(bars))
    height = [mu0, observed_value]
    limit = 1500

    plt.figure(figsize=[12, 6])
    plt.barh(y_pos, height, color=["darkblue", "orange"], height=0.8, alpha=0.75)

    plt.title(f"N={N} - Steps={steps} - P-Value={p_value:.8f} - Improvement of {ratio:.2f}%", fontsize=11)
    plt.xlabel("ns per cell")

    plt.xlim(0, limit)
    plt.yticks(y_pos, bars, color='black')

    plt.rc('grid', linestyle="-", color='black')
    plt.grid()

    plt.savefig(f"{plotfile}.png")
