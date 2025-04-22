#!/usr/bin/env python3

import argparse
import numpy as np
import itk
import gatetools as gt


def main():
    print(args)

    projections_itk = itk.imread(args.projections)
    projections_array = itk.array_from_image(projections_itk)
    id_shift = args.shiftindex
    shift_mm = float(args.shift)
    all_shifted_projections_itk = gt.applyTransformation(projections_itk, keep_original_canvas=True, translation = [0, shift_mm, 0])

    all_shifted_projections_array = itk.array_from_image(all_shifted_projections_itk)

    projections_array[id_shift:,:,:] = all_shifted_projections_array[id_shift:,:,:]
    shifted_projections_itk = itk.image_from_array(projections_array)
    shifted_projections_itk.CopyInformation(projections_itk)

    itk.imwrite(shifted_projections_itk, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--shift")
    parser.add_argument("-i","--shiftindex", type = int)
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
