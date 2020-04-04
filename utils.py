import sys
import subprocess
import argparse
import git
import shlex
import pandas as pd
import statsmodels.api as sm
from io import StringIO
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    '-sb',
    '--sbranch',
    nargs='?',
    help='name of the source branch')

parser.add_argument(
    '-smake',
    required=True,
    help='make-commands to be run by the source branch'
)

parser.add_argument(
    '-tb',
    '--tbranch',
    help='name of the target branch')

parser.add_argument(
    '-tmake',
    required=True,
    help='make-commands to be run by the target branch'
)

parser.add_argument(
    '-sexec',
    nargs='?',
    default='./headless',
    help='executable command to be run by the source branch')

parser.add_argument(
    '-texec',
    nargs='?',
    default='./headless',
    help='executable command to be run by the target branch')

parser.add_argument(
    '-csv',
    action='store_true',
    help='write measurements to a csv file.')

parser.add_argument(
    '-p',
    '--plot',
    action='store_true',
    help='plot graphics.')


import statsmodels.api as sm


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
        make commands separeted by spaces
    exec_cmds: str
        executable commands separeted by spaces
    Returns
    -------
    data: Panda DataFrame object
        measurements obtained by stout after running exec_cmds.
    """
    try:
        subprocess.run(['make', 'clean'])
        subprocess.run(shlex.split(make_cmds))
        result = subprocess.run(
            shlex.split(exec_cmds),
            stdout=subprocess.PIPE,
            shell=True).stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print('Fail execution of', e, file=sys.stderr)
        sys.exit(1)

    data = StringIO(result)
    data = pd.read_csv(data)
    return data


def main():

    repo_abs_path = str(Path(".").resolve())
    repo = git.Repo(repo_abs_path)

    args = parser.parse_args()
    sbranch = repo.active_branch.name if args.sbranch is None else args.sbranch
    tbranch = args.tbranch
    smake_cmds = args.smake
    sexec_cmds = args.sexec
    tmake_cmds = args.tmake
    texec_cmds = args.texec

    if tbranch is None:
        source_info = get_data_from_stdin(smake_cmds, sexec_cmds)
        target_info = get_data_from_stdin(tmake_cmds, texec_cmds)
    else:
        if repo.is_dirty():
            print('There are uncommited changes', file=sys.stderr)
            sys.exit(1)
        repo.git.checkout(sbranch)
        source_info = get_data_from_stdin(smake_cmds, sexec_cmds)
        repo.git.checkout(tbranch)
        target_info = get_data_from_stdin(tmake_cmds, texec_cmds)
        repo.git.checkout(sbranch)

    significance_level = 0.01
    mu0, observed_value, p_value, ratio = test_hypotheses(source_info['total_ns'], 
                                                          target_info['total_ns'])
    if p_value < significance_level:
        print('Assert in favour of the new approach.')
    else:
        print('No evidence to reject the previous approach.')
    print("Mu0:", mu0)
    print("Observed_value:", observed_value)
    print(f"Ratio: {ratio}%")


if __name__ == "__main__":
    main()
