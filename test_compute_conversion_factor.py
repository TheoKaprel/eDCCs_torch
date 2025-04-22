#!/usr/bin/env python3

import argparse
import itk
from itk import RTK as rtk
import numpy as np
import time
def main():
    print(args)
    t0 = time.time()
    pixelType = itk.F
    imageType = itk.Image[pixelType, 3]

    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(args.geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    projection_angles = geometry.GetGantryAngles()
    nprojs = len(projection_angles)

    K_region_img = itk.imread(args.kregion, pixelType)
    attmap_img = itk.imread(args.attmap)
    like_proj_img = itk.imread(args.likeproj)

    K_region_array = itk.array_from_image(K_region_img)
    attmap_array = itk.array_from_image(attmap_img)
    like_proj_array = itk.array_from_image(like_proj_img)

    # mu0 = attmap_array[K_region_array == 1].mean()
    # attmap_array[K_region_array == 1] = mu0

    mask = itk.binary_threshold_image_filter(K_region_img, lower_threshold=1e-10, outside_value=0, inside_value=1)
    mask = itk.cast_image_filter(mask, ttype=(itk.Image[itk.F, 3], itk.Image[itk.UC, 3]))
    stats = itk.LabelStatisticsImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
    stats.SetInput(attmap_img)
    stats.SetLabelInput(mask)
    stats.Update()
    mu0 = stats.GetMean(1)

    # attmap_img_mu0 = itk.image_from_array(attmap_array)
    # attmap_img_mu0.CopyInformation(attmap_img)

    K_region_fp_filter = rtk.JosephForwardProjectionImageFilter[imageType, imageType].New()
    K_region_fp_filter.SetGeometry(geometry)
    zero_proj_array = np.zeros_like(like_proj_array)
    zero_proj_img = itk.image_from_array(zero_proj_array)
    zero_proj_img.CopyInformation(like_proj_img)

    K_region_fp_filter.SetInput(0, zero_proj_img)
    K_region_fp_filter.SetInput(1, K_region_img)
    K_region_fp_filter.Update()
    projected_K_region_img = K_region_fp_filter.GetOutput()
    itk.imwrite(projected_K_region_img, "test_K_region_1/projected_K_region.mha")
    projected_K_region_array = itk.array_from_image(projected_K_region_img)


    conversion_factor_array = np.zeros_like(like_proj_array)

    ###
    origin = np.array(K_region_img.GetOrigin())
    size = np.array(itk.size(K_region_img))
    numpy_matrix_id_to_physical_coordinates = np.zeros((3, 3))
    for j in range(3):
        index = [0] * 3
        index[j] = 1
        point = K_region_img.TransformIndexToPhysicalPoint(index)
        for i in range(3):
            numpy_matrix_id_to_physical_coordinates[i, j] = point[i] - origin[i]
    i, j, k = np.arange(size[2]), np.arange(size[1]), np.arange(size[0])
    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    indices = np.stack((I, J, K)).transpose((1, 2, 0, 3))
    physical_position = np.dot(numpy_matrix_id_to_physical_coordinates, indices) + origin[::-1][:, None, None, None]
    physical_position_vector = np.reshape(physical_position, (3, -1)).transpose()

    sid = 280
    tol = 1e-6
    for ii in range(len(projection_angles)):
        i = (ii + 30) % 120
        theta_i = projection_angles[i]
        dir_det = itk.GetArrayFromMatrix(geometry.GetRotationMatrix(i))[2, :3]
        dir_det[2] = - dir_det[2]
        D = - np.dot(dir_det, sid * np.array([-np.sin(theta_i), 0, np.cos(theta_i)]))
        distances_to_detector = np.abs(np.dot(physical_position_vector, dir_det) + D)
        distances_to_detector = np.reshape(distances_to_detector, (physical_position.shape[1],
                                                                   physical_position.shape[2],
                                                                   physical_position.shape[3]))
        Ktilde_array = ((sid - distances_to_detector) * K_region_array).astype(np.float32)
        Ktilde_image = itk.image_from_array(Ktilde_array)
        Ktilde_image.CopyInformation(K_region_img)

        Ktilde_region_fp_filter = rtk.JosephForwardProjectionImageFilter[imageType, imageType].New()
        geometry_angle_i = rtk.ThreeDCircularProjectionGeometry.New()
        geometry_angle_i.AddProjectionInRadians(geometry.GetSourceToIsocenterDistances()[ii],
                                             geometry.GetSourceToDetectorDistances()[ii],
                                             geometry.GetGantryAngles()[ii], 0, 0, geometry.GetOutOfPlaneAngles()[ii])
        Ktilde_region_fp_filter.SetGeometry(geometry_angle_i)
        zero_proj_array = np.zeros_like(like_proj_array[0:1, :, :])
        zero_proj_img = itk.image_from_array(zero_proj_array)
        zero_proj_img.CopyInformation(like_proj_img)
        Ktilde_region_fp_filter.SetInput(0, zero_proj_img)
        Ktilde_region_fp_filter.SetInput(1, Ktilde_image)
        Ktilde_region_fp_filter.Update()
        projected_Ktilde_region_i_img = Ktilde_region_fp_filter.GetOutput()
        projected_Ktilde_region_i_array = itk.array_from_image(projected_Ktilde_region_i_img)
        projected_K_region_i_array = projected_K_region_array[ii:ii + 1]

        tau1_i_array = np.zeros_like(projected_Ktilde_region_i_array)

        tau1_i_array[projected_K_region_i_array>tol] = (1 / 2 * projected_K_region_i_array
                                                      + projected_Ktilde_region_i_array / projected_K_region_i_array)[projected_K_region_i_array > tol]
        tau0_i_array = tau1_i_array - projected_K_region_i_array
        tau_exit_i_array = np.maximum(tau0_i_array, tau1_i_array)
        inferior_clip_i_array = (tau_exit_i_array + sid ) / (2*sid)
        inferior_clip_i_array[projected_K_region_i_array<=tol] = 1

        inferior_clip_i_img = itk.image_from_array(inferior_clip_i_array.astype(np.double))
        inferior_clip_i_img.CopyInformation(like_proj_img)

        joseph_fp = rtk.JosephForwardProjectionImageFilter[imageType, imageType].New()
        joseph_fp.SetGeometry(geometry_angle_i)
        zero_proj_array = np.zeros_like(like_proj_array[0:1,:,:])
        zero_proj_img = itk.image_from_array(zero_proj_array)
        zero_proj_img.CopyInformation(like_proj_img)
        joseph_fp.SetInput(0, zero_proj_img)

        joseph_fp.SetInput(1, attmap_img)
        joseph_fp.SetInferiorClipImage(inferior_clip_i_img)
        joseph_fp.Update()
        conversion_factor_angle_i = joseph_fp.GetOutput()
        conversion_factor_array[ii:ii+1,:,:] = np.exp(itk.array_from_image(conversion_factor_angle_i) + mu0 * tau_exit_i_array)

    conversion_factor = itk.image_from_array(conversion_factor_array)
    conversion_factor.CopyInformation(like_proj_img)
    itk.imwrite(conversion_factor, args.output)
    print(f'elsapsed time: {round(time.time() - t0, 3)}')

    ###

    # for i,angle in enumerate(projection_angles):
    #     # Filters to be re-used
    #     bp_projected_K_region_filter = rtk.JosephBackProjectionImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    #     multiplyImageFilter = itk.MultiplyImageFilter[imageType, imageType, imageType].New()
    #     fp_masked_attmap_along_ray_crossing_K_region = rtk.JosephForwardProjectionImageFilter[imageType, imageType].New()
    #
    #     zero_img = itk.image_from_array(zero_img_array)
    #     zero_img.CopyInformation(attmap_img)
    #     bp_projected_K_region_filter.SetInput(0, zero_img)
    #     extract_size = list(itk.size(projected_K_region))
    #     extract_size[2] = 1
    #     # Set up extraction region
    #     region = itk.ImageRegion[3]()
    #     region.SetSize(extract_size)
    #     # Apply the ExtractImageFilter
    #     extract_filter = itk.ExtractImageFilter.New(projected_K_region, ExtractionRegion=region)
    #     extract_filter.SetDirectionCollapseToSubmatrix()  # Preserve correct orientation
    #     thresholdFilter = itk.BinaryThresholdImageFilter[imageType, imageType].New()
    #     thresholdFilter.SetLowerThreshold(1e-6)
    #     thresholdFilter.SetOutsideValue(0)
    #     thresholdFilter.SetInsideValue(1)
    #
    #
    #     # geometry
    #     geometry_angle_i = rtk.ThreeDCircularProjectionGeometry.New()
    #     geometry_angle_i.AddProjectionInRadians(geometry.GetSourceToIsocenterDistances()[i],
    #                                          geometry.GetSourceToDetectorDistances()[i],
    #                                          geometry.GetGantryAngles()[i], 0, 0, geometry.GetOutOfPlaneAngles()[i])
    #     bp_projected_K_region_filter.SetGeometry(geometry_angle_i)
    #
    #     # select projection angle i
    #     extract_index = [0, 0, i]
    #     region.SetIndex(extract_index)
    #     extract_filter.Update()
    #     projected_K_region_angle_i = extract_filter.GetOutput()
    #
    #     # back-project the projected K region
    #     bp_projected_K_region_filter.SetInput(1, projected_K_region_angle_i)
    #     bp_projected_K_region_filter.Update()
    #     bp_projected_K_region = bp_projected_K_region_filter.GetOutput()
    #
    #     # mask to keep only positive values
    #     thresholdFilter.SetInput(bp_projected_K_region)
    #     thresholdFilter.Update()
    #     mask_bp_projected_K_region = thresholdFilter.GetOutput()
    #
    #
    #     fp_masked_attmap_along_ray_crossing_K_region.SetGeometry(geometry_angle_i)
    #     zero_proj_array = np.zeros_like(like_proj_array[0:1,:,:])
    #     zero_proj = itk.image_from_array(zero_proj_array)
    #     zero_proj.CopyInformation(like_proj_img)
    #     fp_masked_attmap_along_ray_crossing_K_region.SetInput(0, zero_proj)
    #
    #     # keep in the attmap only the region for which the rays intersect the K region
    #     multiplyImageFilter.SetInput1(attmap_img_mu0)
    #     multiplyImageFilter.SetInput2(mask_bp_projected_K_region)
    #     multiplyImageFilter.Update()
    #     masked_attmap_along_ray_crossing_K_region = multiplyImageFilter.GetOutput()
    #
    #     # forward project the masked attmap only from the center to the detector
    #     fp_masked_attmap_along_ray_crossing_K_region.SetInput(1, masked_attmap_along_ray_crossing_K_region)
    #     fp_masked_attmap_along_ray_crossing_K_region.SetInferiorClip(0.5)
    #     fp_masked_attmap_along_ray_crossing_K_region.Update()
    #     conversion_factor_angle_i = fp_masked_attmap_along_ray_crossing_K_region.GetOutput()
    #     conversion_factor_array[i:i+1,:,:] = np.exp(itk.array_from_image(conversion_factor_angle_i))
    #
    # conversion_factor = itk.image_from_array(conversion_factor_array)
    # conversion_factor.CopyInformation(like_proj_img)
    # itk.imwrite(conversion_factor, 'testdir/conversion_factor.mha')
    #
    # print(f'elsapsed time: {time.time()-t0}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--geom")
    parser.add_argument("--likeproj")
    parser.add_argument("--output")
    args = parser.parse_args()
    main()
