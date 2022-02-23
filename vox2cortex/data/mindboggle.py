
""" Mindboggle data."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random

import numpy as np

from data.parcdata import ParcDataset

class MindboggleDataset(ParcDataset):
    subdatasets = (
        'Extra-18',
        'MMRR-21',
        'NKI-RS-22',
        'NKI-TRT-20',
        'OASIS-TRT-20'
    )

    def __init__(self, raw_data_dir: str, split: str, atlas='DKT31'):
        random.seed(0)
        self.atlas = atlas
        super().__init__(raw_data_dir, split)

    def load_data(self, raw_data_dir):
        return MindboggleDataset.load_manual_surfaces(raw_data_dir, self.atlas)

    def select_split(self, data, split):
        ids = list(data.keys())

        train_portion = 0.8
        val_portion = 0.1
        test_portion = 0.1

        n_train = int(np.ceil(train_portion * len(ids)))
        n_val = int(np.floor(val_portion * len(ids)))
        n_test = len(ids) - n_train - n_val

        random.shuffle(ids)
        ids_train = ids[slice(0, n_train)]
        ids_val = ids[slice(n_train, n_train + n_val)]
        ids_test = ids[(n_train + n_val):]

        if split == 'train':
            return {k: v for k, v in data.items() if k in ids_train}
        if split == 'val':
            return {k: v for k, v in data.items() if k in ids_val}
        if split == 'test':
            return {k: v for k, v in data.items() if k in ids_test}

        raise ValueError(f"Unknown split {split}")

    @staticmethod
    def read_vtk(filename: str):
        points = []
        polys = []
        point_data = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                if 'POINTS' in line:
                    n_points = int(line.split()[1])
                    for _ in range(n_points):
                        points.append(np.array(
                            list(map(float, list(f.readline().split())))
                        ))
                    line = f.readline()
                elif 'POLYGONS' in line:
                    n_polys = int(line.split()[1])
                    for _ in range(n_polys):
                        polys.append(np.array(
                            list(map(int, f.readline().split(" ")))[1:]
                        ))
                    line = f.readline()
                elif 'POINT_DATA' in line:
                    n_pd = int(line.split()[1])
                    while not line.replace("\n","").replace(".", "", 1).isdigit():
                        line = f.readline()
                    # First line already read
                    point_data.append(np.array(float(line.split()[0])))
                    for _ in range(n_pd - 1):
                        point_data.append(np.array(float(f.readline().split()[0])))
                    line = f.readline()
                else:
                    line = f.readline()

            return np.stack(points), np.stack(polys), np.stack(point_data)

    @staticmethod
    def load_manual_surfaces(raw_data_dir: str, atlas: str):
        """ Load the mindboggle dataset, consisting of
        pial surfaces + vertex labels. The returned dict has the form
        {id: {structure: {verts: (V, D), faces: (F, D), verts_labels: (V, 1)} ...} ...}
        """
        data = {}

        # Iterate over sub-datasets
        for subd in MindboggleDataset.subdatasets[:1]:
            subd_dir = os.path.join(raw_data_dir, subd + "_surfaces")
            ids = os.listdir(subd_dir)

            # Iterate over ids
            for i in ids[:1]:
                id_dir = os.path.join(subd_dir, i)
                fns = os.listdir(id_dir)

                # Iterate over files
                data[i] = {}
                for fn in fns:
                    # Only load files of specified atlas
                    if not atlas in fn:
                        continue

                    # Load file
                    fn_full = os.path.join(id_dir, fn)
                    verts, faces, verts_labels = MindboggleDataset.read_vtk(fn_full)

                    data[i][fn] = {
                        "verts": verts,
                        "faces": faces,
                        # View as (V, 1)
                        "verts_labels": np.expand_dims(verts_labels, -1)
                    }

        return data
