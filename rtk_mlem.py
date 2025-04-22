#!/usr/bin/env python3

import argparse
from itk import RTK as rtk
import itk
import numpy as np
import torch

class SPECT_system_rtk:
    def __init__(self, projections_fn, likeimg_fn, geom_fn):
        self.Dimension = 3
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.likeimg_fn = likeimg_fn

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
        self.likeimg_itkimg = itk.imread(self.likeimg_fn,self.pixelType)

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.likeimg_array = itk.array_from_image(self.likeimg_itkimg)

    def forward(self, image_array):
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(self.likeimg_itkimg)

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
        zero_img_array = np.zeros_like(self.likeimg_array)
        zero_img_itkimg = itk.image_from_array(zero_img_array)
        zero_img_itkimg.CopyInformation(self.likeimg_itkimg)
        self.joseph_back_projector.SetInput(0, zero_img_itkimg)
        self.joseph_back_projector.SetInput(1, projections_itkimg)
        self.joseph_back_projector.Update()
        image_itkimg = self.joseph_back_projector.GetOutput()
        return itk.array_from_image(image_itkimg)

    def init_mlem(self):
        ones_projection_array = np.ones_like(self.projection_itkimg)
        ones_projection_itkimg = itk.image_from_array(ones_projection_array)
        ones_projection_itkimg.CopyInformation(self.projection_itkimg)

        zero_img_array = np.zeros_like(self.likeimg_array)
        zero_img_itkimg = itk.image_from_array(zero_img_array)
        zero_img_itkimg.CopyInformation(self.likeimg_itkimg)

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



def main():
    print(args)

    spect = SPECT_system_rtk(projections_fn=args.projections,likeimg_fn = args.likeimg,geom_fn=args.geom)
    spect.init_mlem()
    image_k_array = np.ones_like(spect.likeimg_array)

    for iteration in range(1,args.niter+1):
        image_k_array = spect.mlem_iter(image_k_array)

    image_k_itkimg = itk.image_from_array(image_k_array)
    image_k_itkimg.CopyInformation(spect.likeimg_itkimg)
    itk.imwrite(image_k_itkimg, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--likeimg")
    parser.add_argument("--geom")
    parser.add_argument("--output")
    parser.add_argument("--niter", type = int, default=10)
    args = parser.parse_args()

    main()
