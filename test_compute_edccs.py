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

    # ------------------------------------------------------------------------------
    # Geometry and references
    # ------------------------------------------------------------------------------

    t0_torch = time.time()
    projection_edcc_torch = ExponentialProjectionsTorch(projs_fn=args.projs,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name = args.device)

    translation = [20, 30, 40]
    shifted_array = np.zeros_like(projection_edcc_torch.projections_array)
    for i in range(shifted_array.shape[0]):
        print(i)
        shifted_array[i,:,:] = projection_edcc_torch.apply_translation_itk(translation=translation,index=i)

    shifted_itk = itk.image_from_array(shifted_array)
    shifted_itk.CopyInformation(projection_edcc_torch.projections_img)
    itk.imwrite(shifted_itk,"motion_correction/shifted_projections_itk.mha")

    shifted_array = projection_edcc_torch.apply_translation_torch_3(translation=torch.Tensor(translation)).cpu().numpy()

    shifted_itk = itk.image_from_array(shifted_array)
    shifted_itk.CopyInformation(projection_edcc_torch.projections_img)
    itk.imwrite(shifted_itk, "motion_correction/shifted_projections_torch_2.mha")

    # em_slice = [40, 190]

    # # t1_torch = time.time()
    # # edcc_torch_vec = projection_edcc_torch.compute_edcc_vectorized(em_slice=em_slice)
    # # t2_torch = time.time()
    # # print("*"*30+" vectorized "+"*"*30)
    # # print("elapsed time: {}".format(t2_torch-t1_torch))
    # # print("*" * 72)
    #
    # t1_torch = time.time()
    # edcc_torch_vec_2 = projection_edcc_torch.compute_edcc_vectorized_2(em_slice=em_slice)
    # t2_torch = time.time()
    # print("*"*30+" vectorized 2 "+"*"*30)
    # print("elapsed time: {}".format(t2_torch-t1_torch))
    # print("*" * 72)
    #
    #
    #
    # fig, ax = plt.subplots(figsize=(12, 4))
    # # ax.hist(edcc_torch_vec.ravel(), bins=100, alpha=0.5, color = "blue", label = "v1")
    # ax.hist(edcc_torch_vec_2.cpu().numpy().ravel(), bins=100, alpha=0.5, color = "red", label = "v2")
    # # ax.hist(edcc_torch_vec_m.cpu().numpy().ravel(), bins=100, alpha=0.5, color = "red", label = "m=60")
    # ax.set_xlabel("eDCC")
    # ax.set_ylabel("Frequency")
    # # ax.set_xlim([0, 3])
    # plt.legend()
    #
    # # fig,ax = plt.subplots()
    # # for m in range(1,119):
    # #     edcc_torch_vec_m = edcc_torch_vec_2[:m,m:].mean()
    # #     ax.scatter(m, edcc_torch_vec_m, color = "red")
    #
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--projs")
    parser.add_argument("--device", default = "cpu")
    args = parser.parse_args()

    main()

