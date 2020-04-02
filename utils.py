import pdb
import sys
import pandas as pd
import matplotlib as plt

def main():
    data = pd.read_csv(sys.stdin)
    total_ns = data["total_ns"]
    react = data["react"]
    vel_step = data["vel_step"]
    dens_step = data["dens_step"]


if __name__ == "__main__":
    main()