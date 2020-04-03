import sys
import subprocess
import argparse
import pandas as pd
import git
import shlex
from utils import test_hypotheses as test
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

def get_data_from_stdin(make_cmds, exec_cmds):
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
        print(source_info)
        print(target_info)    
    else: 
        if repo.is_dirty():
            print('There are uncommited changes. You cannot switch between branches.', 
                file=sys.stderr)
            sys.exit(1)
        repo.git.checkout(sbranch)
        source_info = get_data_from_stdin(smake_cmds, sexec_cmds)
        repo.git.checkout(tbranch)
        target_info = get_data_from_stdin(tmake_cmds, texec_cmds)
        repo.git.checkout(sbranch)    

    print(source_info)
    print(target_info)
    if test.test_hypotheses(source_info['total_ns'], target_info['total_ns']):
        print('There are strong evidence to assert in favour of the new approach')
    else:
        print('There are no strong evidence to reject the previous approach')

if __name__ == "__main__":
    main()