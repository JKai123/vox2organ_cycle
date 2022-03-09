
""" Convert result files to latex table entry. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import csv
from argparse import ArgumentParser

FILE_NAMES = ("lh_white_eval_ad_hd_summary.csv",
              "rh_white_eval_ad_hd_summary.csv",
              "lh_pial_eval_ad_hd_summary.csv",
              "rh_pial_eval_ad_hd_summary.csv",
              "eval_ad_hd_summary.csv")

if __name__ == '__main__':

    out_file = "latex_out.txt"
    out_str = ""

    for f in FILE_NAMES:

        with open(f, 'r') as csv_file:
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
