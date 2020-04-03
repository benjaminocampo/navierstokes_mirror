import pdb
import sys
import subprocess
import argparse
from utils import test_hypotheses as test
import pandas as pd
import git
from io import StringIO 
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    '-sb', 
    '--sbranch', 
    required=True,
    help='name of the source branch')

parser.add_argument(
    '-scmd',
    '--scommand',
    required=True,
    help='command to be executed by the source branch'
)

parser.add_argument(
    '-tb',
    '--tbranch', 
    required=True,
    help='name of the target branch')

parser.add_argument(
    '-tcmd',
    '--tcommand',
    required=True,
    help='command to be executed by the target branch'
)

parser.add_argument(
    '-csv', 
    action='store_true',
    help='write measurements to a csv file.')

parser.add_argument(
    '-p',
    '--plot',
    action='store_true',
    help='plot graphics.')

def get_data_from_stdin(cmd):
    try:
        subprocess.run(['make', 'clean'], check=True)
        subprocess.run(['make'], check=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')
        subprocess.run(['make', 'clean'], check=True)
    except FileNotFoundError as err:
        print('Fail execution of headless:', err, 
            file=sys.stderr)
        sys.exit(1)
        
    data = StringIO(result)
    data = pd.read_csv(data)
    return data

def main():

    args = parser.parse_args()
    sbranch = args.sbranch
    tbranch = args.tbranch
    scmd = args.scommand
    tcmd = args.tcommand

    repo_abs_path = str(Path(".").resolve())

    repo = git.Repo(repo_abs_path)

    if repo.is_dirty():
        print('There are uncommited changes. You cannot switch between branches.', 
            file=sys.stderr)
        sys.exit(1)

    repo.git.checkout(sbranch)

    source_info = get_data_from_stdin(scmd)

    repo.git.checkout(tbranch)

    target_info = get_data_from_stdin(tcmd)

    test.test_hypotheses(source_info['total_ns'], target_info['total_ns'])

    repo.git.checkout('master')

if __name__ == "__main__":
    main()