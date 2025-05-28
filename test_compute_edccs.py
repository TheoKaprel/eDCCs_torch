#!/usr/bin/env python3

# Welcome
# ------------------------------------------------------------------------------
# Load packages
import itk
import numpy as np
from matplotlib import pyplot as plt
import argparse
from exponential_projections_torch import ExponentialProjectionsTorch
import time
import torch

def main():
    print(args)
    projection_edcc_torch = ExponentialProjectionsTorch(projs_fn=args.projs,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name = args.device)

    fig,ax = plt.subplots()

    em_slice = [20,108]
    for projs_fn in args.projs:
        projections_img = itk.imread(projs_fn)
        projections_array = itk.array_from_image(projections_img)
        projections_tensor = torch.from_numpy(projections_array).to(projection_edcc_torch.device).to(torch.float64)
        edccs = projection_edcc_torch.compute_edcc_vectorized_input_proj(em_slice=em_slice,projections_tensor=projections_tensor,del_mask=True)
        ax.hist(edccs.cpu().numpy(), bins=100, label = projs_fn, alpha = 0.5)



    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--projs", nargs="+")
    parser.add_argument("--device", default = "cpu")
    args = parser.parse_args()

    main()

