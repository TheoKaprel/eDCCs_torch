#!/usr/bin/env python3

# Welcome
# ------------------------------------------------------------------------------
# Load packages
import itk
import numpy as np
from matplotlib import pyplot as plt
import argparse
from exponential_projections_torch import ExponentialProjectionsTorch,Motion
import torch
import os


def main():
    print(args)

    # ------------------------------------------------------------------------------
    # Geometry and references
    # ------------------------------------------------------------------------------

    projection_edcc_torch = ExponentialProjectionsTorch(projs_fn=args.projs,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name = args.device)
    em_slice = [40, 190]


    motion_model = Motion(device=projection_edcc_torch.device,projections_edcc=projection_edcc_torch, em_slice=em_slice)
    optimizer = torch.optim.Adam(motion_model.parameters(), lr=1)
    iteration = 0
    while iteration<100:
        optimizer.zero_grad()
        eDCCs = motion_model.forward()
        eDCCs.backward()
        optimizer.step()
        print('iteration: {} | loss: {} | m: {} | translation: {}'.format(iteration, round(eDCCs.item(), 4),motion_model.m.item(), [round(t,4) for t in motion_model.translation.tolist()]))
        iteration+=1


    # list_m = []
    # list_F = []
    # for m in range(2,118):
    #     translation = torch.nn.Parameter(torch.randn((3),device=projection_edcc_torch.device,requires_grad=True,dtype=torch.float32))
    #     optimizer = torch.optim.Adam([translation,], lr=1)
    #     iteration = 0
    #     while iteration<100:
    #         optimizer.zero_grad()
    #         projection_edcc_torch.re_init_projections_tensor()
    #         projection_edcc_torch.apply_translation_torch_m(translation=translation, m = m)
    #         eDCCs = projection_edcc_torch.compute_edcc_vectorized_2(em_slice=em_slice).mean()
    #         eDCCs.backward()
    #         optimizer.step()
    #         print('m: {} | iteration: {} | loss: {} | translation: {}'.format(m,iteration, round(eDCCs.item(), 4), [round(t,4) for t in translation.tolist()]))
    #         iteration+=1
    #
    #     list_m.append(m)
    #     list_F.append(eDCCs.item())
    #
    # np.save(os.path.join(args.outputfolder,'array_m.npy'),np.array(list_m))
    # np.save(os.path.join(args.outputfolder,'array_F.npy'),np.array(list_F))
    #
    # fig,ax = plt.subplots()
    # ax.scatter(list_m, list_F, color = "red")
    # ax.set_xlabel('m')
    # ax.set_ylabel('F(m, am)')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--projs")
    parser.add_argument("--device", default = "cpu")
    parser.add_argument("--outputfolder")
    args = parser.parse_args()

    main()

