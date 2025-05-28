#!/usr/bin/env python3

import argparse
import time

from itk import RTK as rtk
import itk
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(image_tensor, spect_model):
        """
        This function computes the forward projection of the an image "image_tensor" (given in input as a torch Tensor)
        according to the SPECT sytem contained in "spect_model"

        The forward projection is done with the RTK package.

        :param image_tensor: image to forward-project given as a torch Tensor
        :param spect_model:  spect model containing image/projection shapes, geometry, projectors
        :return: forward-projected image, also as a torch Tensor
        """
        device = image_tensor.device
        image_array = image_tensor.cpu().numpy()
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(spect_model.like_itkimg)
        spect_model.forward_projector.SetInput(1, image_itkimg)
        spect_model.set_zero_proj_to_forward_projector()
        spect_model.forward_projector.Update()

        forward_projected_itkimg = spect_model.forward_projector.GetOutput()
        return torch.tensor(itk.array_from_image(forward_projected_itkimg), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # keeps in memory the spect_model, which will be used in the "backward" to compute gradients
        _,spect_model = inputs
        ctx.spect_model = spect_model

    @staticmethod
    def backward(ctx, projections_tensor):
        """
        This function computes the gradient of "forward" function defined above.
        This gradient is the backprojection of the resulting projections, also computed with RTK

        :param ctx: context that was set during forward
        :param projections_tensor: projections to backproject
        :return: back-projected projections
        """
        spect_model = ctx.spect_model
        projections_array = projections_tensor.cpu().numpy()
        projections_itkimg = itk.image_from_array(projections_array)
        projections_itkimg.CopyInformation(spect_model.projection_itkimg)
        spect_model.set_zero_img_to_back_projector()
        spect_model.back_projector.SetInput(1, projections_itkimg)

        spect_model.back_projector.Update()

        image_itkimg = spect_model.back_projector.GetOutput()
        image_tensor = torch.tensor(itk.array_from_image(image_itkimg),device=projections_tensor.device)
        # We return gradients with respect to each input of the "forward" method
        # note: the returned "None" corresponds to the gradient wrt the second input ("spect_model")
        #       for which we don't need to track the gradient
        return image_tensor, None



class SPECT_system_torch(torch.nn.Module):
    def __init__(self, projections_fn, like_fn, fbprojectors,nsubsets):
        super().__init__()
        self.Dimension = 3
        # self.pixelType = itk.D
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.like_fn = like_fn

        self.init_images()

        self.nproj = 120
        self.nsubsets = nsubsets

        self.geometries = self.get_geometries()
        if fbprojectors=="Joseph":
            self.forward_projector = rtk.JosephForwardProjectionImageFilter[self.imageType,self.imageType].New()
            self.back_projector = rtk.JosephBackProjectionImageFilter[self.imageType,self.imageType].New()
            self.cuda_fb=False
        elif fbprojectors=="Zeng":
            self.forward_projector = rtk.ZengForwardProjectionImageFilter[self.imageType,self.imageType].New()
            self.forward_projector.SetAlpha(0.03235363042582603)
            self.forward_projector.SetSigmaZero(1.1684338873367237)

            self.back_projector = rtk.ZengBackProjectionImageFilter[self.imageType,self.imageType].New()
            self.back_projector.SetAlpha(0.03235363042582603)
            self.back_projector.SetSigmaZero(1.1684338873367237)
            self.cuda_fb = False
        elif fbprojectors=="Cuda":
            self.imageType = itk.CudaImage[self.pixelType,self.Dimension]
            self.forward_projector = rtk.CudaForwardProjectionImageFilter[self.imageType].New()
            self.back_projector = rtk.CudaBackProjectionImageFilter[self.imageType].New()
            self.cuda_fb = True

    def init_images(self):
        self.projection_itkimg = itk.imread(self.projection_fn,self.pixelType)
        self.like_itkimg = itk.imread(self.like_fn,self.pixelType)
        self.projection_spacing = list(self.projection_itkimg.GetSpacing())

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.like_array = itk.array_from_image(self.like_itkimg)

        self.zero_proj_array = np.zeros_like(self.projection_array)
        self.zero_img_array = np.zeros_like(self.like_array)


    def set_zero_proj_to_forward_projector(self):
        zero_proj_itkimg = itk.image_from_array(self.zero_proj_array[self.subset_ids,:,:])
        zero_proj_itkimg.CopyInformation(self.projection_itkimg)
        self.forward_projector.SetInput(0, zero_proj_itkimg)

    def set_zero_img_to_back_projector(self):
        zero_img_itkimg = itk.image_from_array(self.zero_img_array)
        zero_img_itkimg.CopyInformation(self.like_itkimg)
        self.back_projector.SetInput(0, zero_img_itkimg)


    def get_geometries(self):
        self.nprojs_per_subsets = self.nproj//self.nsubsets
        self.sid = 280
        list_angles = np.linspace(0,360,self.nproj+1)
        geometries = []
        for subset in range(self.nsubsets):
            geometry = rtk.ThreeDCircularProjectionGeometry.New()
            for i in range(self.nprojs_per_subsets):
                geometry.AddProjection(self.sid, 0, int(list_angles[subset + i*self.nsubsets]), 0,0)
            geometries.append(geometry)
        return geometries

    def set_geometry(self, geom_index):
        self.subset_ids = torch.tensor([int(geom_index + self.nsubsets * j) for j in range(self.nprojs_per_subsets)])
        self.geometry = self.geometries[geom_index]
        self.forward_projector.SetGeometry(self.geometry)
        self.back_projector.SetGeometry(self.geometry)


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
                               nsubsets=args.nsubsets)


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
    parser.add_argument("--fbprojectors", choices=['Joseph', 'Zeng', 'Cuda'])
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--nsubsets", type=int)
    parser.add_argument("--output-every", type = int)
    parser.add_argument("--iteration-file-name", type = str)
    parser.add_argument("--niter", type = int, default=10)
    parser.add_argument("--lr", type = float, default = 0.1)
    args = parser.parse_args()

    main()
