
""" Check the implementation of some functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np

from data.dataset import offset_due_to_padding, img_with_patch_size

def check_offset_calculation():
    patch_size = (4,5,6)
    img = np.array([[[0,0,0],[0,0,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])
    coords = img.nonzero()
    offset = offset_due_to_padding(img.shape, patch_size)

    coord_new = np.array(img.nonzero()).squeeze() + offset

    new_img = img_with_patch_size(img, patch_size, is_label=True)

    ref_coords = np.array(new_img.numpy().nonzero()).squeeze()
    assert (coord_new == ref_coords).all()

    print("Test passed.")

if __name__ == '__main__':
    check_offset_calculation()
