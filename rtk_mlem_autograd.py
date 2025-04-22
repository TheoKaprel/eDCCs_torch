#!/usr/bin/env python3

import argparse
from itk import RTK as rtk
import itk
import numpy as np
import torch

class SPECT_system_rtk:
    def __init__(self, projections_fn, attmap_fn, geom_fn):
        self.Dimension = 3
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.attmap_fn = attmap_fn

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
        self.attmap_itkimg = itk.imread(self.attmap_fn,self.pixelType)

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.attmap_array = itk.array_from_image(self.attmap_itkimg)

    def forward(self, image_array):
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(self.attmap_itkimg)

        self.joseph_forward_projector.SetInput(1, image_itkimg)
        zero_proj_array = np.zeros_like(self.projection_array)
        zero_proj_itkimg = itk.image_from_array(zero_proj_array)
        zero_proj_itkimg.CopyInformation(self.projection_itkimg)
        self.joseph_forward_projector.SetInput(0, zero_proj_itkimg)
        self.joseph_forward_projector.Update()
        forward_projected_itkimg = self.joseph_forward_projector.GetOutput()
        return itk.array_from_image(forward_projected_itkimg)

    def backward(self, projections_array):
        projections_itkimg = itk.image_from_array(projections_array)
        projections_itkimg.CopyInformation(self.projection_itkimg)
        zero_img_array = np.zeros_like(self.attmap_array)
        zero_img_itkimg = itk.image_from_array(zero_img_array)
        zero_img_itkimg.CopyInformation(self.attmap_itkimg)
        self.joseph_back_projector.SetInput(0, zero_img_itkimg)
        self.joseph_back_projector.SetInput(1, projections_itkimg)
        self.joseph_back_projector.Update()
        image_itkimg = self.joseph_back_projector.GetOutput()
        return itk.array_from_image(image_itkimg)

    def init_mlem(self):
        ones_projection_array = np.ones_like(self.projection_itkimg)
        ones_projection_itkimg = itk.image_from_array(ones_projection_array)
        ones_projection_itkimg.CopyInformation(self.projection_itkimg)

        zero_img_array = np.zeros_like(self.attmap_array)
        zero_img_itkimg = itk.image_from_array(zero_img_array)
        zero_img_itkimg.CopyInformation(self.attmap_itkimg)

        self.joseph_back_projector.SetInput(0, zero_img_itkimg)
        self.joseph_back_projector.SetInput(1, ones_projection_itkimg)
        self.joseph_back_projector.Update()
        bp_ones_itkimg = self.joseph_back_projector.GetOutput()
        self.bp_ones_array = itk.array_from_image(bp_ones_itkimg)
        self.bp_ones_array[self.bp_ones_array==0] = 1

    def mlem_iter(self, image_array):
        fp_image_array = self.forward(image_array)
        fp_image_array[fp_image_array==0] = 1
        return image_array  * self.backward(self.projection_array/fp_image_array) / self.bp_ones_array

class forwardprojection(torch.autograd.Function):
    @staticmethod
    def forward(image_tensor, spect_model):
        device = image_tensor.device
        image_array = image_tensor.cpu().numpy()
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(spect_model.attmap_itkimg)

        spect_model.joseph_forward_projector.SetInput(1, image_itkimg)
        zero_proj_array = np.zeros_like(spect_model.projection_array) #
        zero_proj_itkimg = itk.image_from_array(zero_proj_array) #
        zero_proj_itkimg.CopyInformation(spect_model.projection_itkimg) #
        spect_model.joseph_forward_projector.SetInput(0, zero_proj_itkimg) #
        spect_model.joseph_forward_projector.Update()
        forward_projected_itkimg = spect_model.joseph_forward_projector.GetOutput()
        return torch.tensor(itk.array_from_image(forward_projected_itkimg), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _,spect_model = inputs
        ctx.spect_model = spect_model

    @staticmethod
    def backward(ctx, projections_tensor):

        spect_model = ctx.spect_model

        projections_array = projections_tensor.cpu().numpy()
        projections_itkimg = itk.image_from_array(projections_array)
        projections_itkimg.CopyInformation(spect_model.projection_itkimg)
        zero_img_array = np.zeros_like(spect_model.attmap_array) #
        zero_img_itkimg = itk.image_from_array(zero_img_array) #
        zero_img_itkimg.CopyInformation(spect_model.attmap_itkimg) #
        spect_model.joseph_back_projector.SetInput(0, zero_img_itkimg) #
        spect_model.joseph_back_projector.SetInput(1, projections_itkimg)
        spect_model.joseph_back_projector.Update()
        image_itkimg = spect_model.joseph_back_projector.GetOutput()
        image_tensor = torch.tensor(itk.array_from_image(image_itkimg),device=projections_tensor.device)
        return image_tensor, None



class SPECT_system_torch(torch.nn.Module):
    def __init__(self, projections_fn, attmap_fn, geom_fn):
        super().__init__()

        self.Dimension = 3
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.attmap_fn = attmap_fn

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
        self.attmap_itkimg = itk.imread(self.attmap_fn,self.pixelType)

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.attmap_array = itk.array_from_image(self.attmap_itkimg)



def main():
    print(args)

    spect = SPECT_system_torch(projections_fn=args.projections,attmap_fn=args.attmap,geom_fn=args.geom)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device : {device}")

    image_k_array = np.ones_like(spect.attmap_itkimg)
    image_k_tensor = torch.tensor(image_k_array, device=device)
    image_k_tensor.requires_grad_(True)

    projections_tensor = torch.tensor(spect.projection_array, device=device)

    optimizer = torch.optim.Adam([image_k_tensor,], lr=args.lr)

    for iteration in range(1,args.niter+1):
        optimizer.zero_grad()
        print("iteration {}".format(iteration))
        image_k_tensor_positive = torch.nn.functional.relu(image_k_tensor)
        fp_image_k_tensor = forwardprojection.apply(image_k_tensor_positive, spect)
        loss = (fp_image_k_tensor - projections_tensor * torch.log(fp_image_k_tensor+1e-8)).mean() # poisson log likelihood loss
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        print('     loss: {}'.format(loss.item()))


        image_k_array = image_k_tensor.detach().cpu().numpy()
        image_k_itkimg = itk.image_from_array(image_k_array)
        image_k_itkimg.CopyInformation(spect.attmap_itkimg)
        itk.imwrite(image_k_itkimg, "test_autograd/rec_torch_{}.mha".format(iteration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--attmap")
    parser.add_argument("--geom")
    parser.add_argument("--output")
    parser.add_argument("--niter", type = int, default=10)
    parser.add_argument("--lr", type = float, default = 0.1)
    args = parser.parse_args()

    main()
