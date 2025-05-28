#!/usr/bin/env python3

import argparse
import itk
import numpy as np
from matplotlib import pyplot as plt
from exponential_projections_torch import ExponentialProjectionsTorch
import time
import torch

def main():
    print(args)


    exp_projections = ExponentialProjectionsTorch(projs_fn=args.projections,
                                                  attmap_fn=args.attmap,
                                                  kregion_fn=args.kregion,
                                                  conversion_factor_fn=args.cf,
                                                  geometry_fn=args.geom,
                                                  device_name="gpu")
    em_slice = [32,95]

    # correction_k  = torch.ones_like(exp_projections.projections_tensor).to(exp_projections.device)
    correction_k  = torch.rand_like(exp_projections.projections_tensor).to(exp_projections.device)/10+1
    correction_k.requires_grad_(True)

    optimizer = torch.optim.Adam([correction_k,], lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    for epoch in range(args.nepochs):
        optimizer.zero_grad()
        corrected_projections_k = correction_k * exp_projections.projections_tensor
        eDCCs_k = exp_projections.compute_edcc_vectorized_input_proj(em_slice=em_slice,projections_tensor=corrected_projections_k,del_mask=True,compute_var=True).mean()
        consistency_k = mse_loss(corrected_projections_k,exp_projections.projections_tensor)
        loss = eDCCs_k

        loss.backward()
        optimizer.step()

        print('iteration {} | loss: {} (edcc={} / mse={} )'.format(epoch,
                                                                   round(loss.item(), 4),
                                                                   round(eDCCs_k.item(), 4),
                                                                   (consistency_k.item())))


    corrected_projections_k_array = corrected_projections_k.detach().cpu().numpy()
    corrected_projections_k_itkimg = itk.image_from_array(corrected_projections_k_array)
    corrected_projections_k_itkimg.CopyInformation(exp_projections.projections_img)
    itk.imwrite(corrected_projections_k_itkimg, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--kregion")
    parser.add_argument("--attmap")
    parser.add_argument("--cf")
    parser.add_argument("--geom")
    parser.add_argument("--lr",type = float, default = 0.1)
    parser.add_argument("--nepochs",type = int, default = 10)
    parser.add_argument("--device", default = "cpu")
    parser.add_argument("--output")
    args = parser.parse_args()

    main()
