
""" Convert result files to latex table entry. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import csv
from argparse import ArgumentParser

FILE_NAMES = ("lh_white_eval_ad_hd_summary_corrected.csv",
              "rh_white_eval_ad_hd_summary_corrected.csv",
              "lh_pial_eval_ad_hd_summary_corrected.csv",
              "rh_pial_eval_ad_hd_summary_corrected.csv",
              "eval_ad_hd_summary_corrected.csv")

if __name__ == '__main__':
    argparser = ArgumentParser(description="Mesh evaluation procedure")
    argparser.add_argument('exp_name',
                           type=str,
                           help="Name of experiment under evaluation.")
    argparser.add_argument('n_test_vertices',
                           type=int,
                           help="The number of template vertices for each"
                           " structure that was used during testing.")
    argparser.add_argument('dataset',
                           type=str,
                           help="The dataset.")

    args = argparser.parse_args()
    exp_name = args.exp_name
    n_test_vertices = args.n_test_vertices
    dataset = args.dataset

    exp_dir = f"../experiments/{exp_name}/test_template_{n_test_vertices}_{dataset}"

    out_file = os.path.join(exp_dir, "latex_out.txt")
    out_str = ""

    for f in FILE_NAMES:
        f_full = os.path.join(exp_dir, f)

        with open(f_full, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            # Read second row
            r1 = csv_reader.__next__()
            r2 = csv_reader.__next__()
            r2 = [float(i) for i in r2]
            out_str += f"{r2[0]:.3f} \\interval{{{r2[1]:.3f}}} & {r2[2]:.3f} \\interval{{{r2[3]:.3f}}}"
            if f != FILE_NAMES[-1]:
                out_str += " & "

    out_str += " \\\\ \n "
    with open(out_file, 'w') as f_out:
        f_out.write(out_str)
