import numpy as np
import itk
from itk import RTK as rtk
import math
import gatetools as gt
import torch
import time
import torchvision

class ExponentialProjectionsTorch():
    def __init__(self,
                 projs_fn,
                 attmap_fn,
                 kregion_fn,
                 conversion_factor_fn,
                 geometry_fn,
                 device_name = None):

        self.device = get_device(device_name=device_name)
        self.read_input_images(projs_fn=projs_fn,attmap_fn=attmap_fn, kregion_fn=kregion_fn, conversion_factor_fn=conversion_factor_fn)
        self.read_geometry(geometry_fn=geometry_fn)
        self.compute_mu0()
        self.get_physical_coordinates()

        self.exponential_projections_array = self.projections_array * self.conversion_factor_array
        self.exponential_projections_tensor = torch.from_numpy(self.exponential_projections_array).to(device=self.device).to(torch.float64)

    def read_input_images(self, projs_fn,attmap_fn, kregion_fn, conversion_factor_fn):

        self.kregion_img = itk.imread(kregion_fn)
        self.attmap_img = itk.imread(attmap_fn)
        self.conversion_factor_img = itk.imread(conversion_factor_fn)
        self.projections_img = itk.imread(projs_fn)

        self.kregion_array = itk.array_from_image(self.kregion_img).astype(np.int32)
        self.attmap_array = itk.array_from_image(self.attmap_img)
        self.conversion_factor_array = itk.array_from_image(self.conversion_factor_img)
        self.projections_array = itk.array_from_image(self.projections_img)

        self.kregion_tensor = torch.from_numpy(self.kregion_array).to(self.device)
        self.attmap_tensor = torch.from_numpy(self.attmap_array).to(self.device)
        self.conversion_factor_tensor = torch.from_numpy(self.conversion_factor_array).to(self.device).to(torch.float64)
        self.projections_tensor = torch.from_numpy(self.projections_array).to(self.device).to(torch.float64)

    def re_init_projections_tensor(self):
        self.projections_tensor = torch.from_numpy(self.projections_array).to(self.device).to(torch.float64)
        self.exponential_projections_tensor = self.projections_tensor * self.conversion_factor_tensor

    def set_new_projections_tensor(self, projections_tensor):
        self.projections_tensor = projections_tensor
        self.exponential_projections_tensor = self.projections_tensor * self.conversion_factor_tensor

    def read_geometry(self, geometry_fn):
        geometryReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
        geometryReader.SetFilename(geometry_fn)
        geometryReader.GenerateOutputInformation()
        self.geometry = geometryReader.GetGeometry()
        self.angles_rad_array = 1. * np.asarray(self.geometry.GetGantryAngles())
        self.angles_rad_tensor = torch.from_numpy(self.angles_rad_array).to(device=self.device).to(torch.float64)
        self.n_angles = len(self.angles_rad_array)

    def compute_mu0(self):
        # self.mu0 = self.attmap_array[self.kregion_array==1].mean()

        mask = itk.binary_threshold_image_filter(self.kregion_img, lower_threshold=1e-10, outside_value=0, inside_value=1)
        mask = itk.cast_image_filter(mask, ttype=(itk.Image[itk.F, 3], itk.Image[itk.UC, 3]))
        stats = itk.LabelStatisticsImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
        stats.SetInput(self.attmap_img)
        stats.SetLabelInput(mask)
        stats.Update()
        self.mu0 = stats.GetMean(1)

    def get_physical_coordinates(self):
        self.projs_size = list(itk.size(self.projections_img))
        self.spacing = list(self.projections_img.GetSpacing())
        self.origin = list(self.projections_img.GetOrigin())
        self.direction = itk.GetArrayFromMatrix(self.projections_img.GetDirection())

        self.s = np.linspace(self.origin[0], self.origin[0] + (self.projs_size[0] - 1) * self.spacing[0], self.projs_size[0])
        self.s_tensor = torch.from_numpy(self.s).to(device=self.device)
        self.delta_s = (self.s_tensor[1] - self.s_tensor[0])

    def compute_edcc_sequential(self, em_slice):

        with torch.no_grad():
            t0 = time.time()
            ind_min, ind_max = extract_index(em_slice)
            if ind_max == -1:
                ind_max = self.projs_size[1]
            list_slice = torch.arange(ind_min, ind_max).to(device=self.device)
            self.edcc = torch.Tensor([]).to(device=self.device)
            # self.edcc_no_var = torch.Tensor([]).to(device=self.device)
            variance_pij = torch.Tensor([]).to(device=self.device)
            variance_pji = torch.Tensor([]).to(device=self.device)
            t1 = time.time()- t0
            t2_ = time.time()
            # list_double_angles = [(self.angles_rad_tensor[i], self.angles_rad_tensor[j]) for i in range(len(self.angles_rad_tensor)) for j in
            #                       range(len(self.angles_rad_tensor)) if (torch.abs(self.angles_rad_tensor[i] - self.angles_rad_tensor[j]) > 0.001)
            #                       and (torch.abs(torch.abs(self.angles_rad_tensor[i] - self.angles_rad_tensor[j]) - torch.pi) > torch.deg2rad(torch.tensor(1)))]
            list_double_indices = [(i, j) for i in range(len(self.angles_rad_tensor)) for j in
                                  range(len(self.angles_rad_tensor)) if (torch.abs(self.angles_rad_tensor[i] - self.angles_rad_tensor[j]) > 0.001)
                                  and (torch.abs(torch.abs(self.angles_rad_tensor[i] - self.angles_rad_tensor[j]) - torch.pi) > torch.deg2rad(torch.tensor(1)))]
            t2 = time.time() - t2_
            t3,t4 = 0,0
            for (ind_phi_i,ind_phi_j) in list_double_indices:
                t3_ = time.time()
                phi_i,phi_j = self.angles_rad_tensor[ind_phi_i], self.angles_rad_tensor[ind_phi_j]

                if torch.abs(phi_i- phi_j + torch.pi) < 0.0001 or torch.abs(phi_i - phi_j - torch.pi) < 0.0001 or torch.abs(
                        phi_i - phi_j) < 0.0001:
                    continue
                else:
                    sigma_ij = self.mu0 * torch.tan(0.5 * (phi_i - phi_j))
                    sigma_ji = -1 * sigma_ij

                    projection_i = self.exponential_projections_tensor[ind_phi_i][list_slice, :]
                    projection_j = self.exponential_projections_tensor[ind_phi_j][list_slice, :]
                    # P_ij = torch.sum(torch.multiply(projection_i, torch.exp(self.s_tensor * sigma_ij)) * (self.s_tensor[1] - self.s[0]), dim=1)
                    P_ij = torch.sum(projection_i * torch.exp(self.s_tensor * sigma_ij) * (self.s_tensor[1] - self.s_tensor[0]), dim=1)
                    # P_ji = torch.sum(torch.multiply(projection_j, torch.exp(self.s_tensor * sigma_ji)) * (self.s_tensor[1] - self.s[0]), dim=1)
                    P_ji = torch.sum(projection_j * torch.exp(self.s_tensor * sigma_ji) * (self.s_tensor[1] - self.s_tensor[0]), dim=1)
                    P_ij[torch.sum(projection_i, dim=1) < 10] = 0
                    P_ji[torch.sum(projection_j, dim=1) < 10] = 0
                    t3+=time.time()-t3_
                    t4_ = time.time()
                    if True:
                        self.edcc = torch.cat((self.edcc, torch.mean(torch.abs(torch.subtract(P_ij, P_ji)))[None])) if (len(self.edcc)!=0) else torch.mean(torch.abs(torch.subtract(P_ij, P_ji)))[None]
                        # self.edcc_no_var = torch.cat((self.edcc_no_var, torch.mean(torch.abs(torch.subtract(P_ij, P_ji)))[None])) if (len(self.edcc_no_var)!=0) else torch.mean(torch.abs(torch.subtract(P_ij, P_ji)))[None]

                        Var_ij = (self.s_tensor[1] - self.s_tensor[0]) ** 2 * torch.sum(
                            torch.multiply(self.conversion_factor_tensor[ind_phi_i, list_slice, :] ** 2,
                                        self.projections_tensor[ind_phi_i, list_slice, :]) * torch.exp(2 * sigma_ij * self.s_tensor), dim=1)
                        Var_ji = (self.s_tensor[1] - self.s_tensor[0]) ** 2 * torch.sum(
                            torch.multiply(self.conversion_factor_tensor[ind_phi_j, list_slice, :] ** 2,
                                        self.projections_tensor[
                                        ind_phi_j, list_slice, :]) * torch.exp(2 * sigma_ji * self.s_tensor), dim=1)
                        Var_ij[torch.sum(projection_i, dim=1) < 10] = 0
                        Var_ji[torch.sum(projection_j, dim=1) < 10] = 0
                        variance_pij = torch.cat((variance_pij,torch.sum(Var_ij)[None])) if (len(variance_pij)!=0) else torch.sum(Var_ij)[None]
                        variance_pji = torch.cat((variance_pji,torch.sum(Var_ji)[None])) if (len(variance_pji)!=0) else torch.sum(Var_ji)[None]
                    t4+=time.time()-t4_
            t5_ = time.time()
            if True:
                variance_pij = variance_pij / len(list_slice) ** 2
                variance_pji = variance_pji / len(list_slice) ** 2
                self.variance = torch.sqrt(variance_pij + variance_pji)
                # self.edcc[self.variance!=0] = torch.divide(self.edcc[self.variance!=0], self.variance[self.variance!=0] * len(list_slice)**0.5)
                self.edcc[self.variance!=0] = (self.edcc / (self.variance * len(list_slice)**0.5))[self.variance!=0]
            t5 = time.time()-t5_
            for i,t in enumerate([t1,t2,t3,t4,t5]):
                print('t{}:{}'.format(i+1,t))
            print(f"device: {projection_i.device}")
            return self.edcc

    def compute_edcc_vectorized(self, em_slice):
        with torch.no_grad():
            ind_min, ind_max = extract_index(em_slice)
            if ind_max == -1:
                ind_max = self.projs_size[1]
            N = ind_max-ind_min
            angle_indices = torch.arange(0,len(self.angles_rad_tensor)).to(device=self.device)
            i,j = torch.meshgrid(angle_indices,angle_indices, indexing="ij")
            phi_i,phi_j = torch.meshgrid(self.angles_rad_tensor,self.angles_rad_tensor, indexing="ij")

            sigma_ij = self.mu0 * torch.tan(0.5 * (phi_i - phi_j)) # 120,120
            sigma_ji = -1 * sigma_ij
            projection_i = self.exponential_projections_tensor[i,ind_min:ind_max, :] # 120,120,Ny,Nx
            print(f"projection_i.shape: {projection_i.shape}")
            # print(f"projection_i.dtype: {projection_i.dtype}")
            # mem = projection_i.element_size() * projection_i.nelement()
            # print(f"leading to {mem/2**30} GiB")
            projection_j = self.exponential_projections_tensor[j,ind_min:ind_max, :] # 120,120,Ny,Nx

            P_ij = torch.sum(projection_i * torch.exp(self.s_tensor[None,None,None,:] * sigma_ij[:,:,None,None]) * (self.s_tensor[1] - self.s_tensor[0]),
                dim=3) # 120, 120, Ny
            P_ji = torch.sum(projection_j * torch.exp(self.s_tensor[None,None,None,:] * sigma_ji[:,:,None,None]) * (self.s_tensor[1] - self.s_tensor[0]),
                dim=3)

            P_ij[torch.sum(projection_i, dim=3) < 10] = 0
            P_ji[torch.sum(projection_j, dim=3) < 10] = 0

            self.edcc_fast = torch.mean(torch.abs(torch.subtract(P_ij, P_ji)),dim=2)

            Var_ij = (self.s_tensor[1] - self.s_tensor[0]) ** 2 * torch.sum(
                torch.multiply(self.conversion_factor_tensor[i,ind_min:ind_max,:] ** 2,
                               self.projections_tensor[i,ind_min:ind_max,:]) * torch.exp(
                    2 * sigma_ij[:,:,None,None] * self.s_tensor[None,None,None,:]), dim=3)

            Var_ji = (self.s_tensor[1] - self.s_tensor[0]) ** 2 * torch.sum(
                torch.multiply(self.conversion_factor_tensor[j,ind_min:ind_max,:] ** 2,
                               self.projections_tensor[j,ind_min:ind_max,:]) * torch.exp(
                    2 * sigma_ji[:,:,None,None] * self.s_tensor[None,None,None,:]), dim=3)
            Var_ij[torch.sum(projection_i, dim=3) < 10] = 0
            Var_ji[torch.sum(projection_j, dim=3) < 10] = 0

            variance_pij = torch.sum(Var_ij,dim=2) / (N ** 2)
            variance_pji = torch.sum(Var_ji,dim=2) / (N ** 2)

            self.variance = torch.sqrt(N * (variance_pij + variance_pji))
            self.edcc_fast[self.variance > 1e-8] = (self.edcc_fast / self.variance)[self.variance > 1e-8]

            mask_angles = (torch.abs(torch.abs(phi_i - phi_j) - torch.pi) > 0.0001) & (torch.abs(phi_i - phi_j) > 0.0001)
            # self.edcc_fast = self.edcc_fast[mask_angles].ravel()
            # return self.edcc_fast

            self.edcc_fast[~mask_angles] = 0
            return self.edcc_fast


    def compute_edcc_vectorized_2(self, em_slice):
        ind_min, ind_max = extract_index(em_slice)
        if ind_max == -1:
            ind_max = self.projs_size[1]
        N = ind_max-ind_min
        phi_i,phi_j = torch.meshgrid(self.angles_rad_tensor,self.angles_rad_tensor, indexing="ij")
        mask_angles = (torch.abs(torch.abs(phi_i - phi_j) - torch.pi) > 0.0001) & (torch.abs(phi_i - phi_j) > 0.0001)
        # sigma_ij = self.mu0 * torch.tan(0.5 * (phi_i - phi_j)) # 120,120

        sigma_ij = torch.zeros_like(phi_i)
        sigma_ij[mask_angles] = self.mu0 * torch.tan(0.5 * (phi_i[mask_angles] - phi_j[mask_angles])) # 120,120

        exp_sigma_s_ij = torch.exp(sigma_ij[:,:,None]*self.s_tensor[None,None,:])
        P_ij = torch.einsum('ilk,ijk->ijl', self.exponential_projections_tensor[:,ind_min:ind_max,:], exp_sigma_s_ij*self.delta_s)
        P_ji = P_ij.transpose(0,1)

        # P_ij[torch.sum(projection_i, dim=3) < 10] = 0
        # P_ji[torch.sum(projection_j, dim=3) < 10] = 0

        self.edcc_fast = torch.mean(torch.abs(torch.subtract(P_ij, P_ji)),dim=2)

        Var_ij = torch.einsum('ilk,ijk->ij',
                              self.conversion_factor_tensor[:,ind_min:ind_max,:]**2 * self.projections_tensor[:,ind_min:ind_max,:],
                              torch.exp(2*sigma_ij[:,:,None]*self.s_tensor[None,None,:]))
        Var_ji = Var_ij.transpose(0,1)

        # Var_ij[torch.sum(projection_i, dim=3) < 10] = 0
        # Var_ji[torch.sum(projection_j, dim=3) < 10] = 0

        variance_pij = Var_ij * ((self.s_tensor[1] - self.s_tensor[0]) ** 2) / (N**2)
        variance_pji = Var_ji * ((self.s_tensor[1] - self.s_tensor[0]) ** 2) / (N**2)

        self.variance = torch.sqrt(N * (variance_pij + variance_pji))

        self.edcc_fast[self.variance > 1e-8] = self.edcc_fast.clone()[self.variance > 1e-8] / self.variance[self.variance > 1e-8]

        self.edcc_fast[~mask_angles] = 0
        return self.edcc_fast

    def compute_edcc_vectorized_input_proj(self, projections_tensor, em_slice,del_mask = False,compute_var = True):
        exponential_projections_tensor = self.conversion_factor_tensor * projections_tensor

        ind_min, ind_max = extract_index(em_slice)
        if ind_max == -1:
            ind_max = self.projs_size[1]
        N = ind_max-ind_min
        phi_i,phi_j = torch.meshgrid(self.angles_rad_tensor,self.angles_rad_tensor, indexing="ij")
        mask_angles = (torch.abs(torch.abs(phi_i - phi_j) - torch.pi) > 0.0001) & (torch.abs(phi_i - phi_j) > 0.0001)

        sigma_ij = torch.zeros_like(phi_i)
        sigma_ij[mask_angles] = self.mu0 * torch.tan(0.5 * (phi_i[mask_angles] - phi_j[mask_angles])) # 120,120

        exp_sigma_s_ij = torch.exp(sigma_ij[:,:,None]*self.s_tensor[None,None,:])
        P_ij = torch.einsum('ilk,ijk->ijl', exponential_projections_tensor[:,ind_min:ind_max,:], exp_sigma_s_ij*self.delta_s)
        P_ji = P_ij.transpose(0,1)

        self.edcc = torch.mean(torch.abs(torch.subtract(P_ij, P_ji)),dim=2)

        if compute_var:
            Var_ij = torch.einsum('ilk,ijk->ij',
                                  self.conversion_factor_tensor[:,ind_min:ind_max,:]**2 * projections_tensor[:,ind_min:ind_max,:],
                                  torch.exp(2*sigma_ij[:,:,None]*self.s_tensor[None,None,:]))
            Var_ji = Var_ij.transpose(0,1)

            variance_pij = Var_ij * ((self.s_tensor[1] - self.s_tensor[0]) ** 2) / (N**2)
            variance_pji = Var_ji * ((self.s_tensor[1] - self.s_tensor[0]) ** 2) / (N**2)

            self.variance = torch.sqrt(N * (variance_pij + variance_pji))

            self.edcc[self.variance > 1e-8] = self.edcc[self.variance > 1e-8] / self.variance[self.variance > 1e-8]

        if del_mask:
            return self.edcc[mask_angles].ravel()
        else:
            self.edcc[~mask_angles] = 0
            return self.edcc

    def apply_translation_torch(self, translation, index):
        translation_id_1 = (translation[0] * torch.cos(-self.angles_rad_tensor[index]) + translation[1] * math.sin(-self.angles_rad_tensor[index]))/self.spacing[0]
        translation_id_2 = translation[2]/self.spacing[1]
        translation_id = [-translation_id_1,-translation_id_2]
        return torchvision.transforms.functional.affine(self.projections_tensor[index:index+1,:,:],
                                                 translate=translation_id,
                                                 angle = 0, scale=1, shear = 0,
                                                 interpolation=2)
    def apply_translation_torch_2(self, translation, index):
        proj_index = self.projections_tensor[index:index+1,:,:][None,:,:,:]
        theta = torch.Tensor([[[1,
                                0,
                                2*(translation[0] * math.cos(-self.angles_rad_tensor[index]) +
                                translation[1] * math.sin(-self.angles_rad_tensor[index]))/(self.spacing[0]*proj_index.shape[2])],
                              [0, 1, 2*translation[2]/(self.spacing[1]*proj_index.shape[3])]]]).to(torch.float64)

        grid = torch.nn.functional.affine_grid(theta, proj_index.size())
        return torch.nn.functional.grid_sample(proj_index, grid)[0,:,:,:]

    def apply_translation_torch_m(self, translation, m):
        proj = self.projections_tensor[m:,None,:,:]
        # theta = torch.Tensor([[[1,0,2*(translation[0] * torch.cos(-self.angles_rad_tensor[k]) +
        #                         translation[1] * torch.sin(-self.angles_rad_tensor[k]))/(self.spacing[0]*proj.shape[2])],
        #                       [0, 1, 2*translation[2]/(self.spacing[1]*proj.shape[3])]] for k in range(m, len(self.angles_rad_tensor))]).to(torch.float64)
        theta = torch.eye(2,3,dtype=torch.float64, device=self.device)[None,:,:]
        theta = theta.repeat(self.n_angles-m,1,1)
        theta[:,0,2] = 2*(translation[0] * torch.cos(-self.angles_rad_tensor[m:]) +
                                translation[1] * torch.sin(-self.angles_rad_tensor[m:]))/(self.spacing[0]*proj.shape[2])
        theta[:,1,2] = 2*translation[2]/(self.spacing[1]*proj.shape[3])

        grid = torch.nn.functional.affine_grid(theta, proj.size())

        new_projections_tensor = self.projections_tensor.clone()
        new_projections_tensor[m:,:,:] = torch.nn.functional.grid_sample(proj, grid)[:,0,:,:]
        self.set_new_projections_tensor(new_projections_tensor)



    def apply_translation_itk(self, translation, index):
        '''
        Correct the rigid translation on a selected projection
        :param translation: A 3D vector representing the translation
        :type translation: list
        :param index: index of the projection
        :type index: int
        :return: itk image of the corrected projection
        :rtype itk.itkImagePython
        '''
        matrix = [[1, 0, 0.], [0, 1, 0], [0, 0, 1]]
        matrix[0][2] = translation[0] * math.cos(-self.angles_rad_array[index]) + translation[1] * math.sin(
            -self.angles_rad_array[index])
        matrix[1][2] = translation[2]
        matrixParameter = itk.matrix_from_array(matrix)
        itk_projection = itk.GetImageFromArray(self.projections_array[index])
        itk_projection.SetOrigin(self.origin[:2])
        itk_projection.SetSpacing(self.spacing[:2])
        itk_projection.SetDirection(self.direction[:2, :2])
        shifted_itk_projection = gt.applyTransformation(itk_projection, matrix=matrixParameter,
                                                        keep_original_canvas=True)
        return itk.array_from_image(shifted_itk_projection)

def extract_index(input_index):
    """
    Extract the index of start and stop from the input.

    :param input_index: Could be a list of two integers ([start, stop]), a string
        ("start:stop") or an integer
    :type input_index: Union[list, str, int]

    :returns: Index min and max extract from the input parameter
    :rtype: int
    """
    if isinstance(input_index, str):
        if ":" in input_index:
            input_index = [int(i) if i.isdigit() else i for i in input_index.split(":")]
            if isinstance(input_index[0], int):
                ind_min = input_index[0]
            else:
                ind_min = 0
            if isinstance(input_index[1], int):
                ind_max = input_index[1]
            else:
                ind_max = -1
        else:
            ind_min = int(input_index)
            ind_max = int(input_index) + 1
    elif isinstance(input_index, list):
        if isinstance(input_index[0], int):
            ind_min = input_index[0]
        else:
            ind_min = 0
        if isinstance(input_index[1], int):
            ind_max = input_index[1]
        else:
            ind_max = -1
    elif isinstance(input_index, int):
        ind_min = input_index
        ind_max = input_index + 1
    else:
        raise (TypeError("Wrong input type {} for {}".format(type(input_index), input_index)))

    return ind_min, ind_max

def get_device(device_name=None):
    if device_name is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_name == "cpu":
        device = torch.device('cpu')
    elif device_name in ["gpu", "cuda"]:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print(f'WARNING : device was set to cuda but cuda is not available, so cpu...')
    else:
        print("WARNING : device {} unrecognized.".format(device_name))
    print('Device is: {}'.format(device))
    return device



class Motion(torch.nn.Module):
    def __init__(self,projections_edcc, device, em_slice):
        super().__init__()
        self.translation = torch.nn.Parameter(
            torch.randn((3), device=device, requires_grad=True, dtype=torch.float32))

        self.m = torch.nn.Parameter(
            torch.rand((1), device=device, requires_grad=True, dtype=torch.float32))

        self.projections_edcc = projections_edcc
        self.em_slice = em_slice

    def forward(self):
        self.projections_edcc.re_init_projections_tensor()
        self.projections_edcc.apply_translation_torch_m(translation=self.translation, m = torch.round_(120*self.m).to(torch.int))
        eDCCs = self.projections_edcc.compute_edcc_vectorized_2(em_slice=self.em_slice).mean()
        return eDCCs