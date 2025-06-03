#!/usr/bin/env python3

import torch
import itk
from itk import RTK as rtk
import numpy as np

class SPECT_system_torch(torch.nn.Module):
    def __init__(self, projections_fn, like_fn, fbprojectors,nsubsets,attmap_fn=None):
        super().__init__()
        self.Dimension = 3
        # self.pixelType = itk.D
        self.pixelType = itk.F
        self.imageType = itk.Image[self.pixelType, self.Dimension]

        self.projection_fn = projections_fn
        self.like_fn = like_fn
        self.nproj = 120
        self.nsubsets = nsubsets

        self.geometries = self.get_geometries()
        self.init_images()

        self.fb_projectors_name = fbprojectors
        if fbprojectors=="Joseph":
            self.forward_projector = rtk.JosephForwardProjectionImageFilter[self.imageType,self.imageType].New()
            self.back_projector = rtk.JosephBackProjectionImageFilter[self.imageType,self.imageType].New()
            self.cuda_fb=False
        elif fbprojectors=="JosephAttenuated":
            self.forward_projector = rtk.JosephForwardAttenuatedProjectionImageFilter[self.imageType,self.imageType].New()
            self.back_projector = rtk.JosephBackAttenuatedProjectionImageFilter[self.imageType,self.imageType].New()
            self.attmap_itkimg = itk.imread(attmap_fn,self.pixelType)
            self.forward_projector.SetInput(2, self.attmap_itkimg)
            self.back_projector.SetInput(2, self.attmap_itkimg)
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

        self.set_geometry(0)
        self.projection_itkimg_subset = itk.image_from_array(self.projection_array[self.subset_ids,:,:])
        self.projection_itkimg_subset.SetSpacing(self.projection_spacing)
        size_subset = list(self.projection_array[self.subset_ids, :, :].shape)[::-1]
        origin_subset = [(-s*sp+sp)/2 for s,sp in zip(size_subset,self.projection_spacing)]
        self.projection_itkimg_subset.SetOrigin(origin_subset)

    def init_images(self):
        self.projection_itkimg = itk.imread(self.projection_fn,self.pixelType)
        self.like_itkimg = itk.imread(self.like_fn,self.pixelType)
        self.projection_spacing = list(self.projection_itkimg.GetSpacing())

        self.projection_array = itk.array_from_image(self.projection_itkimg)
        self.like_array = itk.array_from_image(self.like_itkimg)

        self.zero_proj_array = np.zeros_like(self.projection_array)
        self.zero_img_array = np.zeros_like(self.like_array)

    def set_zero_proj_to_forward_projector(self):
        # zero_proj_itkimg = itk.image_from_array(self.zero_proj_array[self.subset_ids,:,:])
        zero_proj_itkimg = itk.GetImageFromArray(self.zero_proj_array[self.subset_ids,:,:])
        zero_proj_itkimg.CopyInformation(self.projection_itkimg_subset)
        self.forward_projector.SetInput(0, zero_proj_itkimg)

        if self.fb_projectors_name=="JosephAttenuated":
            self.forward_projector.SetInput(2, self.attmap_itkimg)


    def set_zero_img_to_back_projector(self):
        # zero_img_itkimg = itk.image_from_array(self.zero_img_array)
        zero_img_itkimg = itk.GetImageFromArray(self.zero_img_array)
        zero_img_itkimg.CopyInformation(self.like_itkimg)
        self.back_projector.SetInput(0, zero_img_itkimg)
        if self.fb_projectors_name=="JosephAttenuated":
            self.back_projector.SetInput(2, self.attmap_itkimg)


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
        self.geom_index = geom_index
        self.subset_ids = torch.tensor([int(geom_index + self.nsubsets * j) for j in range(self.nprojs_per_subsets)])
        self.geometry = self.geometries[geom_index]
        self.forward_projector.SetGeometry(self.geometry)
        self.back_projector.SetGeometry(self.geometry)

    def get_bp_ones(self):
        self.set_zero_img_to_back_projector()
        ones_proj_array = np.ones_like(self.projection_array[self.subset_ids,:,:])
        ones_proj_img = itk.image_from_array(ones_proj_array)
        ones_proj_img.CopyInformation(self.projection_itkimg_subset)
        self.back_projector.SetInput(1, ones_proj_img)
        self.back_projector.Update()
        bp_ones_itkimg = self.back_projector.GetOutput()
        return itk.array_from_image(bp_ones_itkimg)