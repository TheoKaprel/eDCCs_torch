#!/usr/bin/env python3

# Welcome
# ------------------------------------------------------------------------------
# Load packages
from matplotlib import pyplot as plt
import argparse
from exponential_projections_torch import ExponentialProjectionsTorch
import time

def main():
    print(args)

    # ------------------------------------------------------------------------------
    # Geometry and references
    # ------------------------------------------------------------------------------

    t0_torch = time.time()
    projection_edcc_torch = ExponentialProjectionsTorch(projs_fn=args.projs,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name = "cpu")
    t1_torch = time.time()
    em_slice = [40, 190]
    edcc_torch_fast = projection_edcc_torch.comute_edcc_maybe_faster(em_slice=em_slice)
    t2_torch = time.time()
    print("elapsed time: {} ({})".format(t2_torch-t0_torch,t2_torch-t1_torch))
    fig,ax = plt.subplots()
    ax.imshow(edcc_torch_fast.cpu().numpy())
    plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.hist(edcc_torch_fast, bins=100, alpha=0.5)
    # ax.set_xlabel("eDCC")
    # ax.set_ylabel("Frequency")
    # ax.set_xlim([0, 3])
    # ax.set_title('TEST')
    # plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--projs")
    args = parser.parse_args()

    main()

