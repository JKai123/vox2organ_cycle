#!/usr/bin/env python3

""" Visualization of 3D data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import argparse

from utils.visualization import show_pointcloud

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D data.")
    parser.add_argument('filenames',
                        nargs='+',
                        type=str,
                        help="The filenames or the name of one folder to visualize.")
    parser.add_argument('--backend',
                        metavar='LIB',
                        type=str,
                        default='pyvista',
                        help="The library used for visualization, 'open3d' or 'pyvista' (default).")

    args = parser.parse_args()
    if os.path.isdir(args.filenames[0]):
        filenames = args.filenames[0]
    else:
        filenames = args.filenames
    show_pointcloud(filenames, backend=args.backend)


if __name__ == "__main__":
    main()
