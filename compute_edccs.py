#!/usr/bin/env python3

# Welcome
# ------------------------------------------------------------------------------
# Load packages
import itk
import numpy as np
from itk import RTK as rtk
from edcc.projections import *
from edcc.exponential_projections import *
import sys
from matplotlib import pyplot as plt
import argparse


def main():
    print(args)

    # ------------------------------------------------------------------------------
    # Geometry and references
    # ------------------------------------------------------------------------------
    geometryReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    geometryReader.SetFilename(args.geom)
    geometryReader.GenerateOutputInformation()
    geometry = geometryReader.GetGeometry()
    rotation_angles = (np.asarray(geometry.GetGantryAngles()))
    N = len(rotation_angles)

    attenuation_map = itk.imread(args.attmap)

    K_region = VolumeClass(args.kregion)
    conversion_factor_path = args.cf
    conversion_factor = itk.imread(conversion_factor_path)
    ref_exponential_projection = ExponentialProjectionsGeometryClass(attenuation_map, geometry,
                                                                     voxelized_region=K_region, like=conversion_factor)
    ref_exponential_projection.read_conversion_factor(conversion_factor_path)

    # ------------------------------------------------------------------------------
    # Generate random spheres and random shifts
    # ------------------------------------------------------------------------------

    projection_original = itk.imread(args.projs)

    # ------------------------------------------------------------------------------
    # Correct motion
    # ------------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 4))
    em_slice = [90, 170]
    # em_slice = 142
    projection = ProjectionsClass(projection_original, geometry)
    # projection.normalize(projection_original)
    # projection.add_poisson_noise()
    # projection.add_gaussian_filter(sigma=2, two_dimensional=True)
    projection_edcc = ExponentialProjectionsClass(projection, ref_exponential_projection)
    edcc = projection_edcc.compute_edcc(em_slice, divide_by_variance=True)
    ax.hist(edcc, bins=100, alpha=0.5)
    ax.set_xlabel("eDCC")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 3])
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

