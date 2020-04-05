import sys
import subprocess
import argparse
import git
import shlex
import pandas as pd
import statsmodels.api as sm
from io import StringIO, TextIOWrapper
from pathlib import Path

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

def  get_parser(active_branch, default_n=2048, default_steps=4):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sb',
        '--sbranch',
        nargs='?',
        default=active_branch,
        help='name of the source branch, current one by default'
    )
    parser.add_argument(
        '-smake',
        required=True,
        help='make-commands to be run by the source branch'
    )
    parser.add_argument(
        '-tb',
        '--tbranch',
        nargs='?',
        default=active_branch,
        help='name of the target branch, current one by default'
    )
    parser.add_argument(
        '-tmake',
        required=True,
        help='make-commands to be run by the target branch'
    )
    parser.add_argument(
        '-sexec',
        nargs='?',
        default=None,
        help='executable command to be run by the source branch'
    )
    parser.add_argument(
        '-texec',
        nargs='?',
        default=None,
        help='executable command to be run by the target branch'
    )
    parser.add_argument(
        '-n',
        nargs="?",
        default=default_n,
        help="length of the square grid"
    )
    parser.add_argument(
        '-steps',
        nargs="?",
        default=default_steps,
        help="amount of steps to simulate"
    )
    parser.add_argument(
        '-csv',
        action='store_true',
        help='[TO BE DONE] write measurements to a csv file.'
    )
    parser.add_argument(
        '-p',
        '--plot',
        action='store_true',
        help='[TO BE DONE] plot graphics.'
    )
    return parser

def main():
    repo_abs_path = str(Path(".").resolve())
    repo = git.Repo(repo_abs_path)
    initial_branch = repo.active_branch.name
    args = get_parser(initial_branch).parse_args()

    sbranch = args.sbranch
    tbranch = args.tbranch
    n = args.n
    steps = args.steps
    default_exec = f"./headless {n} 0.1 0.001 0.0001 5.0 100.0 {steps}"
    sexec = args.sexec or default_exec
    texec = args.texec or default_exec
    smake = args.smake
    tmake = args.tmake
    print(
        f"Running with the following configuration:\n"
        f"Source branch={sbranch}\n"
        f"Source make={smake}\n"
        f"Source exec={sexec}\n"
        f"Target branch={tbranch}\n"
        f"Target make={tmake}\n"
        f"Target exec={texec}\n"
    )

    failed = False
    was_dirty = repo.is_dirty()
    if was_dirty:
        repo.git.stash()
    try:
        repo.git.checkout(sbranch)
        source_info = get_data_from_stdin(smake, sexec)
        repo.git.checkout(tbranch)
        target_info = get_data_from_stdin(tmake, texec)
    except BaseException as e:
        failed = True
        print(f"There was an error in the execution: {repr(e)}")
    finally:
        repo.git.checkout(initial_branch)
        if was_dirty:
            repo.git.stash("pop")
    if failed:
        exit(1)

    significance_level = 0.01
    mu0, observed_value, p_value, ratio = test_hypotheses(source_info['total_ns'],
                                                          target_info['total_ns'])
    if p_value < significance_level:
        print(f"P value: {p_value} < {significance_level}")
        print("Assert in favour of the new approach.")
    else:
        print(f"P value: {p_value} >= {significance_level}")
        print("No evidence to reject the previous approach.")
    print("Source mean:", mu0)
    print("Target mean:", observed_value)
    print(f"Percentage improved from source to target: {ratio:.2f}%")


if __name__ == "__main__":
    main()
