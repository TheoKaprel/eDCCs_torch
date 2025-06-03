#!/usr/bin/env python3

import argparse
import time

from itk import RTK as rtk
import itk
import numpy as np
import torch
import matplotlib.pyplot as plt

from projectors import ForwardProjection
from SPECTmodel import SPECT_system_torch

plt.rcParams.update({'font.size': 20})


class CNN(torch.nn.Module):
    def __init__(self, nc=8, ks = 3, nl = 6):
        super(CNN, self).__init__()
        sequence = []

        list_channels = [1]
        for _ in range(nl):
            list_channels.append(nc)

        p = (ks-1)//2

        for k in range(len(list_channels)-1):
            sequence.append(torch.nn.Conv3d(in_channels=list_channels[k], out_channels=list_channels[k+1],
                                           kernel_size=(ks,ks,ks),stride=(1,1,1),padding=p))
            sequence.append(torch.nn.BatchNorm3d(list_channels[k+1]))
            sequence.append(torch.nn.ReLU(inplace=True))

        sequence.append(torch.nn.Conv3d(in_channels=list_channels[-1], out_channels=1,
                                  kernel_size=(ks, ks, ks), stride=(1, 1, 1), padding=p))

        self.sequenceCNN = torch.nn.Sequential(*sequence)
        self.activation= torch.nn.ReLU(inplace=True)

    def forward(self,x):
        x = x[None,None,:,:,:]
        res = x
        y = self.sequenceCNN(x.clone())
        y = y + res
        return self.activation(y)[0,0,:,:,:]


def main():
    print(args)

    spect = SPECT_system_torch(projections_fn=args.projections,
                               like_fn=args.likeimg,
                               fbprojectors=args.fbprojectors,
                               nsubsets=args.nsubsets,
                               attmap_fn = args.attmap_fn)


    def forward_projection(input, spect = spect):
        return ForwardProjection.apply(input, spect)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # cnn = CNN(nl=2).to(device=device)
    # print(cnn)
    print(f"Device : {device}")
    print(f"Projections spacing: {spect.projection_spacing}")
    image_k_array = np.ones_like(spect.like_array)*(np.mean(spect.projection_array.sum((1,2)))/spect.projection_spacing[0])/np.size(spect.like_array)
    image_k_tensor = torch.tensor(image_k_array.astype(np.float32), device=device)
    image_k_tensor.requires_grad_(True)

    # bp_ones_tensor = torch.tensor(itk.array_from_image(itk.imread("bp_ones.mha")), device=device)

    # print("--"*30)
    # print("GRAD CHECK: ")
    # input = torch.nn.functional.relu(torch.randn(image_k_tensor.shape,dtype=torch.double,requires_grad=True) + 10)
    # test_grad = torch.autograd.gradcheck(forward_projection, input, eps = 1e-6, atol = 1e-4,fast_mode=True)
    # print(test_grad)
    # print("--" * 30)

    projections_tensor = torch.tensor(spect.projection_array, device=device)

    optimizer = torch.optim.Adam([image_k_tensor,], lr=args.lr)
    list_loss = []
    list_errors = []
    source_array = itk.array_from_image(itk.imread(args.source))

    for iteration in range(1,args.niter+1):
        for subset in range(spect.nsubsets):
            optimizer.zero_grad()
            spect.set_geometry(subset)
            image_k_tensor_positive = torch.nn.functional.relu(image_k_tensor) # just to make sure image counts are positive
            fp_image_k_tensor = forward_projection(image_k_tensor_positive, spect) # computes forward-projection of current estimate
            loss = (fp_image_k_tensor - projections_tensor[spect.subset_ids,:,:] * torch.log(fp_image_k_tensor+1e-8)).mean() # negativ poisson log likelihood loss
            loss.backward() # backpropagates gradients with respect to the input image
            # image_k_tensor.grad.data *= image_k_tensor_positive/bp_ones_tensor
            optimizer.step() # updates image voxels values


        list_loss.append(loss.item())
        list_errors.append(np.sqrt(np.mean((source_array - image_k_tensor_positive.detach().cpu().numpy())**2)))
        geometryWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
        geometryWriter.SetObject(spect.geometry)
        geometryWriter.SetFilename("virtual_geom.xml")
        geometryWriter.WriteFile()
        print('iteration {} | loss: {}'.format(iteration, round(loss.item(), 4)))

        if (args.output_every is not None) and (iteration%args.output_every==0):
            image_k_array = image_k_tensor_positive.detach().cpu().numpy()
            image_k_itkimg = itk.image_from_array(image_k_array)
            image_k_itkimg.CopyInformation(spect.like_itkimg)
            itk.imwrite(image_k_itkimg, args.iteration_file_name.replace("%d", str(iteration)))

        # device = image_tensor.device
        # image_array = image_tensor.cpu().numpy()
        # image_itkimg = itk.image_from_array(image_array)
        # image_itkimg.CopyInformation(spect_model.like_itkimg)

    image_k_array = image_k_tensor_positive.detach().cpu().numpy()
    image_k_itkimg = itk.image_from_array(image_k_array)
    image_k_itkimg.CopyInformation(spect.like_itkimg)
    itk.imwrite(image_k_itkimg, args.output)


    fig,ax = plt.subplots()
    ax.plot(np.arange(1,args.niter+1), list_loss)
    ax.set_ylabel("Negative Poisson Log Likelihood")
    ax.set_xlabel("Iterations")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, args.niter + 1), list_errors)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Iterations")
    plt.show()
    np.save(args.output[:-4]+"_lMSE.npy", list_errors)
    np.save(args.output[:-4]+"_lnPLL.npy", list_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--likeimg")
    parser.add_argument("--fbprojectors", choices=['Joseph', 'JosephAttenuated', 'Zeng', 'Cuda'])
    parser.add_argument("--attmap_fn")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--nsubsets", type=int)
    parser.add_argument("--output-every", type = int)
    parser.add_argument("--iteration-file-name", type = str)
    parser.add_argument("--niter", type = int, default=10)
    parser.add_argument("--lr", type = float, default = 0.1)
    args = parser.parse_args()

    main()
