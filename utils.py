import pdb
import sys
import subprocess
import pandas as pd
import git
from io import StringIO 
from pathlib import Path

def main():
    repo_abs_path = str(Path(".").resolve())
    repo = git.Repo(repo_abs_path)
    subprocess.run(['make', 'clean'])
    subprocess.run(['make'])
    result = subprocess.run('./headless', stdout=subprocess.PIPE).stdout.decode('utf-8')
    data = StringIO(result)
    data = pd.read_csv(data)
    print(data)
    repo.git.checkout('test_branch')
    subprocess.run(['make', 'clean'])
    subprocess.run(['make'])
    result2 = subprocess.run('./headless', stdout=subprocess.PIPE).stdout.decode('utf-8')
    data2 = StringIO(result2)
    data2 = pd.read_csv(data2)
    print(data2)



if __name__ == "__main__":
    main()