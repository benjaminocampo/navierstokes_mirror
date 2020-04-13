import argparse
import git
from pathlib import Path
from utils import dump_data, test_hypotheses, get_data_from_stdin

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
        help='filename to save the generated plot.'
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
    plotfile = args.plot
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

    dump_data(mu0,
              observed_value,
              p_value, ratio,
              significance_level,
              sbranch,
              tbranch)

    if plotfile:
        plot_charts(plotfile,
                    mu0,
                    observed_value,
                    N,
                    steps,
                    p_value,
                    ratio,
                    slabel,
                    tlabel)


if __name__ == "__main__":
    main()
