# Load packages
import itk
import numpy as np
from itk import RTK as rtk
from edcc.projections import *
from edcc.exponential_projections import *
import sys
import argparse
import time


def main():
    t0 = time.time()
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

    reference_projection = VolumeClass(args.likeproj)
    ref_exponential_projection = ExponentialProjectionsGeometryClass(attenuation_map, geometry, voxelized_region= K_region, like = reference_projection)

    conversion_factor = compute_conversion_factor_in_parallel(ref_exponential_projection, index_list= None)
    itk.imwrite(conversion_factor.itk_image, args.output)

    print(f'elsapsed time: {round(time.time()-t0, 3)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--geom")
    parser.add_argument("--likeproj")
    parser.add_argument("--output")
    args = parser.parse_args()
    main()
