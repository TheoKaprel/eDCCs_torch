#!/usr/bin/env python3

# Welcome
# ------------------------------------------------------------------------------
# Load packages
import time

from matplotlib import pyplot as plt
import argparse
from exponential_projections_torch import ExponentialProjectionsTorch

from itk import RTK as rtk
from edcc.exponential_projections import *
import torch
def main():
    print(args)
    itk.imread(args.attmap)

    em_slice = [90, 170]
    ##########################################################################
    t0 = time.time()
    geometryReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    geometryReader.SetFilename(args.geom)
    geometryReader.GenerateOutputInformation()
    geometry = geometryReader.GetGeometry()
    attenuation_map = itk.imread(args.attmap)
    K_region = VolumeClass(args.kregion)
    conversion_factor_path = args.cf
    conversion_factor = itk.imread(conversion_factor_path)
    ref_exponential_projection = ExponentialProjectionsGeometryClass(attenuation_map, geometry,voxelized_region=K_region, like=conversion_factor)
    ref_exponential_projection.read_conversion_factor(conversion_factor_path)
    projection_original = itk.imread(args.projs)
    projection = ProjectionsClass(projection_original, geometry)
    projection_edcc = ExponentialProjectionsClass(projection, ref_exponential_projection)

    t1 = time.time()
    edcc = projection_edcc.compute_edcc(em_slice, divide_by_variance=True)
    t2 = time.time()
    ##########################################################################
    t0_torch = time.time()
    projection_edcc_torch = ExponentialProjectionsTorch(projs_fn=args.projs,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name = "cpu")
    t1_torch = time.time()
    # edcc_torch_slow = projection_edcc_torch.compute_edcc(em_slice=em_slice)
    edcc_torch_fast = projection_edcc_torch.comute_edcc_maybe_faster(em_slice=em_slice)
    t2_torch = time.time()
    ##########################################################################

    print(f"Time numpy: {round(t2-t0, 4)} ({round(t2-t1, 4)})")
    print(f"Time torch: {round(t2_torch-t0_torch, 4)} ({round(t2_torch-t1_torch, 4)})")

    ##########################################################################

    print('Both edccs are equal:')
    print((np.abs(edcc_torch_fast.cpu().numpy()-projection_edcc.edcc)<1e-8).all())

    ##########################################################################

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(projection_edcc.edcc, bins=100, alpha=0.5, color = "blue")
    ax.hist(edcc_torch_fast.cpu().numpy(), bins=100, alpha=0.5, color = "orange")
    ax.set_xlabel("eDCC")
    ax.set_ylabel("Frequency")
    ax.set_title('TEST')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--projs")
    args = parser.parse_args()

    main()

