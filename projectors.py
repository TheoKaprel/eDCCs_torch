#!/usr/bin/env python3
import torch
import itk


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



class BackProjection(torch.autograd.Function):
    @staticmethod
    def forward(projections_tensor, spect_model):
        projections_array = projections_tensor.cpu().numpy()
        projections_itkimg = itk.image_from_array(projections_array)
        projections_itkimg.CopyInformation(spect_model.projection_itkimg)
        spect_model.set_zero_img_to_back_projector()
        spect_model.back_projector.SetInput(1, projections_itkimg)
        spect_model.back_projector.Update()
        image_itkimg = spect_model.back_projector.GetOutput()
        image_tensor = torch.tensor(itk.array_from_image(image_itkimg),device=projections_tensor.device)
        return image_tensor

    @staticmethod
    def setup_context(ctx, inputs, output):
        _,spect_model = inputs
        ctx.spect_model = spect_model

    @staticmethod
    def backward(ctx, image_tensor):
        device = image_tensor.device
        spect_model = ctx.spect_model
        image_array = image_tensor.cpu().numpy()
        image_itkimg = itk.image_from_array(image_array)
        image_itkimg.CopyInformation(spect_model.like_itkimg)
        spect_model.forward_projector.SetInput(1, image_itkimg)
        spect_model.set_zero_proj_to_forward_projector()
        spect_model.forward_projector.Update()

        forward_projected_itkimg = spect_model.forward_projector.GetOutput()
        return torch.tensor(itk.array_from_image(forward_projected_itkimg), device=device), None




