import argparse
import pandas as pd
import pdb
from utils import test_hypotheses, dump_data, plot_charts

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sd',
        '--sdata',
        required=True,
        type=argparse.FileType('r'),
        help='name of the source file.'
    )
    parser.add_argument(
        '-td',
        '--tdata',
        required=True,
        type=argparse.FileType('r'),
        help='name of the target file.'
    )
    parser.add_argument(
        '-sl',
        '--slabel',
        required=True
    )
    parser.add_argument(
        '-tl',
        '--tlabel',
        required=True
    )
    parser.add_argument(
        '-p',
        '--plot',
        help='filename to save the generated plot.'
    )
    return parser


def main():
    args = get_parser().parse_args()
    slabel = args.slabel
    tlabel = args.tlabel
    plotfile = args.plot
    sdata = pd.read_csv(args.sdata)
    tdata = pd.read_csv(args.tdata)
    operations = 65536
    steps = len(sdata.index)
    N = int(operations/steps)
    significance_level = 0.1
    mu0, observed_value, p_value, ratio = test_hypotheses(sdata['total_ns'],
                                                          tdata['total_ns'])
    dump_data(mu0,
              observed_value,
              p_value,
              ratio,
              significance_level,
              slabel,
              tlabel)

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
