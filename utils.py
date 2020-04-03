import pdb
import sys
import subprocess
import pandas as pd
import git
from io import StringIO 
from pathlib import Path

def get_data_from_stdin():
    subprocess.run(['make', 'clean'])
    subprocess.run(['make'])
    result = subprocess.run('./headless', stdout=subprocess.PIPE).stdout.decode('utf-8')
    data = StringIO(result)
    data = pd.read_csv(data)
    return data

def main():
    # Get absolute path of the repository.
    repo_abs_path = str(Path(".").resolve())

    # Track the repository by means of the absolute path.
    repo = git.Repo(repo_abs_path)
    
    if repo.is_dirty():
        print("There are uncommited changes. You cannot switch between branches.", 
            file=sys.stderr)
        sys.exit(1)

    # Get measurementes given by the current program on branch master
    current_info = get_data_from_stdin()

    # Checkout to another branch
    repo.git.checkout('test_branch')
    
    # Get measurements given by the new branch
    likely_improvement_info = get_data_from_stdin()

if __name__ == "__main__":
    main()