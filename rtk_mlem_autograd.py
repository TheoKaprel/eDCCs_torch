#!/usr/bin/env python3

import argparse
from itk import RTK as rtk
import itk
import numpy as np
import torch
import matplotlib.pyplot as plt

class forwardprojection(torch.autograd.Function):
    @staticmethod
    def forward(image_tensor, spect_model):
        """
        This function computes the forward projection of the an image "image_tensor" (given in input as a torch Tensor)
        according tothe SPECT sytem contained in "spect_model"

        The forward projection is done with the RTK package.

        :param image_tensor: image to forward-project given as a torch Tensor
        :param spect_model:  spect model containing image/projection shapes, geometry, projectors
        :return: forward-projected image, also as a torch Tensor
        """
        device = image_tensor.device
        image_array = image_tensor.cpu().numpy()
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(spect_model.like_itkimg)

        spect_model.joseph_forward_projector.SetInput(1, image_itkimg)
        spect_model.set_zero_proj_to_forward_projector()
        spect_model.joseph_forward_projector.Update()

        forward_projected_itkimg = spect_model.joseph_forward_projector.GetOutput()
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
        spect_model.joseph_back_projector.SetInput(1, projections_itkimg)
        spect_model.joseph_back_projector.Update()

        image_itkimg = spect_model.joseph_back_projector.GetOutput()
        image_tensor = torch.tensor(itk.array_from_image(image_itkimg),device=projections_tensor.device)

        # We return gradients with respect to each input of the "forward" method
        # note: the returned "None" corresponds to the gradient wrt the second input ("spect_model")
        #       for which we don't need to track the gradient
        return image_tensor, None



class SPECT_system_torch(torch.nn.Module):
    def __init__(self, projections_fn, like_fn, geom_fn):
        super().__init__()
        self.Dimension = 3
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.like_fn = like_fn

        self.init_images()

        xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        xmlReader.SetFilename(geom_fn)
        xmlReader.GenerateOutputInformation()
        self.geometry = xmlReader.GetOutputObject()
        self.joseph_forward_projector = rtk.JosephForwardProjectionImageFilter[self.imageType,self.imageType].New()
        self.joseph_forward_projector.SetGeometry(self.geometry)

        self.joseph_back_projector = rtk.JosephBackProjectionImageFilter[self.imageType,self.imageType].New()
        self.joseph_back_projector.SetGeometry(self.geometry)

    def init_images(self):
        self.projection_itkimg = itk.imread(self.projection_fn,self.pixelType)
        self.like_itkimg = itk.imread(self.like_fn,self.pixelType)

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.like_array = itk.array_from_image(self.like_itkimg)


    def set_zero_proj_to_forward_projector(self):
        zero_proj_array = np.zeros_like(self.projection_array)
        zero_proj_itkimg = itk.image_from_array(zero_proj_array)
        zero_proj_itkimg.CopyInformation(self.projection_itkimg)
        self.joseph_forward_projector.SetInput(0, zero_proj_itkimg)

    def set_zero_img_to_back_projector(self):
        zero_img_array = np.zeros_like(self.like_array)
        zero_img_itkimg = itk.image_from_array(zero_img_array)
        zero_img_itkimg.CopyInformation(self.like_itkimg)
        self.joseph_back_projector.SetInput(0, zero_img_itkimg)

def main():
    print(args)

    spect = SPECT_system_torch(projections_fn=args.projections,like_fn=args.likeimg,geom_fn=args.geom)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device : {device}")

    image_k_array = np.ones_like(spect.like_itkimg)
    image_k_tensor = torch.tensor(image_k_array, device=device)
    image_k_tensor.requires_grad_(True)

    projections_tensor = torch.tensor(spect.projection_array, device=device)

    optimizer = torch.optim.Adam([image_k_tensor,], lr=args.lr)
    list_loss = []

    for iteration in range(1,args.niter+1):
        optimizer.zero_grad()
        image_k_tensor_positive = torch.nn.functional.relu(image_k_tensor) # just to make sure image counts are positive
        fp_image_k_tensor = forwardprojection.apply(image_k_tensor_positive, spect) # computes forward-projection of current estimate
        loss = (fp_image_k_tensor - projections_tensor * torch.log(fp_image_k_tensor+1e-8)).mean() # negativ poisson log likelihood loss
        loss.backward() # backpropagates gradients with respect to the input image
        optimizer.step() # updates image voxels values

        list_loss.append(loss.item())
        print('iteration {} | loss: {}'.format(iteration, round(loss.item(), 4)))

        if (args.output_every is not None) and (iteration%args.output_every==0):
            image_k_array = image_k_tensor_positive.detach().cpu().numpy()
            image_k_itkimg = itk.image_from_array(image_k_array)
            image_k_itkimg.CopyInformation(spect.like_itkimg)
            itk.imwrite(image_k_itkimg, args.iteration_file_name.replace("%d", str(iteration)))

    fig,ax = plt.subplots()
    ax.plot(np.arange(1,args.niter+1), list_loss)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--likeimg")
    parser.add_argument("--geom")
    parser.add_argument("--output")
    parser.add_argument("--output-every", type = int)
    parser.add_argument("--iteration-file-name", type = str)
    parser.add_argument("--niter", type = int, default=10)
    parser.add_argument("--lr", type = float, default = 0.1)
    args = parser.parse_args()

    main()
