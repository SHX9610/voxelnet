"""
*Preliminary* pytorch implementation.
data generators for voxelmorph
"""

import numpy as np
import sys

import SimpleITK as sitk

def example_gen(moving_names,batch_size=1):
    """
    generate examples
    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)
        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(moving_names), size=batch_size)
        X_data = []            #  Batch-size Moving names
        for idx in idxes:
            print('moving image:', moving_names[idx])
            X = sitk.ReadImage(moving_names[idx])
            X = sitk.GetArrayFromImage(X)[np.newaxis,  np.newaxis, ...]
            X_data.append(X)

        if batch_size > 1:
            return_movs = np.concatenate(X_data, 0)
        else:
            return_movs = X_data[0]

        return return_movs


if __name__ =='__main__':
    import glob
    import os
    import random
    batch_size = 1
    moving_dir= 'E:\peizhunsd\LPBA40\\train\\'
    moving_names = glob.glob(os.path.join(moving_dir, '*.gz'))
    for i in range(100):
        movs = example_gen(moving_names,batch_size=1)

