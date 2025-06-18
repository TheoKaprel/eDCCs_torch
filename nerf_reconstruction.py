#!/usr/bin/env python3

import os
import argparse
import itk
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from projectors import ForwardProjection
from SPECTmodel import SPECT_system_torch

plt.rcParams.update({'font.size': 20})


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class NERF(torch.nn.Module):
    def __init__(self,n_layers=3,n_features=128,archi="vanilla",
                 nfreq=10, layers_activation="relu", final_activation="relu"):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.archi = archi
        self.n_freq = nfreq

        self.final_activation = final_activation

        if self.archi=="siren":
            self.omega_0 = 30
            self.sequence_linear = []
            self.sequence_linear.append(SineLayer(in_features=3, out_features = self.n_features,
                                            bias = True, is_first = True, omega_0 = self.omega_0))
            for _ in range(self.n_layers):
                self.sequence_linear.append(SineLayer(in_features=self.n_features, out_features = self.n_features,
                                            bias = True, is_first = False, omega_0 = self.omega_0))

            final_linear = nn.Linear(self.n_features, 1)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / self.n_features) / self.omega_0,
                                             np.sqrt(6 / self.n_features) / self.omega_0)
            self.sequence_linear.append(final_linear)
            self.sequence_linear.append(self.get_activation(self.final_activation))
            self.sequence_linear = nn.Sequential(*self.sequence_linear)

        elif self.archi=="vanilla":
            self.activations = layers_activation
            sequence_linear = []
            sequence_linear.append(torch.nn.Linear(self.n_freq*2*3, self.n_features))
            sequence_linear.append(self.get_activation(self.activations))
            for _ in range(self.n_layers):
                sequence_linear.append(nn.Linear(self.n_features, self.n_features))
                sequence_linear.append(self.get_activation(self.activations))
            sequence_linear.append(nn.Linear(self.n_features, 1))
            sequence_linear.append(self.get_activation(self.final_activation))
            self.sequence_linear = nn.Sequential(*sequence_linear)

    def get_activation(self, name):
        if name=="relu":
            return nn.ReLU(inplace=False)
        elif name=="softplus":
            return nn.Softplus()

    def positional_encoding(self,x, num_frequencies=10):
        frequencies = torch.tensor([2 ** i for i in range(num_frequencies)])
        encoding = []
        for freq in frequencies:
            encoding.append(torch.sin(freq * x))
            encoding.append(torch.cos(freq * x))
        return torch.cat(encoding, dim=-1)

    def forward(self, positions):
        if self.archi=="vanilla":
            positions = self.positional_encoding(positions,num_frequencies=self.n_freq)
        return self.sequence_linear(positions)


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
    print(f"Device : {device}")
    print(f"Projections spacing: {spect.projection_spacing}")

    Nz,Ny,Nx = spect.like_array.shape
    spacing = 4.4952
    print(f"Nx: {Nx}")
    print(f"Ny: {Ny}")
    print(f"Nz: {Nz}")

    x = torch.linspace(-Nx * spacing/2+spacing, +(Nx * spacing)/2 - spacing, Nx)
    y = torch.linspace(-Ny * spacing/2+spacing, +(Ny * spacing)/2 - spacing, Ny)
    z = torch.linspace(-Nz * spacing/2+spacing, +(Nz * spacing)/2 - spacing, Nz)
    pos_X,pos_Y,pos_Z = torch.meshgrid([x,y,z], indexing="ij")
    pos_X = torch.reshape(pos_X,(-1,1))
    pos_Y = torch.reshape(pos_Y,(-1,1))
    pos_Z = torch.reshape(pos_Z,(-1,1))
    positions = torch.hstack((pos_X,pos_Y,pos_Z))
    positions_batched = positions.reshape((16,-1,3)) / positions.max()
    print(positions.shape)
    print(positions_batched.shape)

    nerf_model = NERF(n_layers= args.n_layers, n_features=args.n_features,archi = args.archi,nfreq=args.nfreqs,
                      layers_activation=args.layer_acti,final_activation = args.final_acti).to(device)

    print(nerf_model)
    nb_params = sum(p.numel() for p in nerf_model.parameters())
    print(f'NUMBER OF PARAMERS : {nb_params}')
    projections_tensor = torch.tensor(spect.projection_array)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=args.lr)
    list_loss = []
    list_errors = []
    source_array = itk.array_from_image(itk.imread(args.source))

    if args.archi=="vanilla":
        archi_infos = "vanilla_{}nfreq_{}_{}".format(args.nfreq,args.layer_acti,args.final_acti)
    else:
        archi_infos = "siren_{}".format(args.final_acti)

    output_ref = os.path.join(args.outputfolder, "nerf_{}iter_{}ss_{}lr_{}L_{}H_{}_%d.mha".format(args.niter,
                                                                                                  args.nsubsets,
                                                                                                  args.lr,
                                                                                                  args.n_layers,
                                                                                                  args.n_features,
                                                                                                  archi_infos))

    for iteration in range(1,args.niter+1):
        for subset in range(spect.nsubsets):
            optimizer.zero_grad()
            spect.set_geometry(subset)

            output = torch.zeros((positions_batched.shape[0],positions_batched.shape[1]), device=device)
            for b in range(16):
                batch = positions_batched[b,:,:].to(device)
                output[b,:]= nerf_model(batch)[:,0]
                torch.cuda.empty_cache()

            image_k_tensor = output.reshape((Nx,Ny,Nz))

            # image_k_tensor = nerf_model(positions.to(device=device)).reshape((Nx,Ny,Nz))
            fp_image_k_tensor = forward_projection(image_k_tensor, spect) # computes forward-projection of current estimate
            loss = (fp_image_k_tensor - projections_tensor[spect.subset_ids,:,:].to(device=device) * torch.log(fp_image_k_tensor+1e-8)).mean() # negativ poisson log likelihood loss

            loss.backward() # backpropagates gradients with respect to the input image

            optimizer.step() # updates image voxels values


        list_loss.append(loss.item())
        list_errors.append(np.sqrt(np.mean((source_array - image_k_tensor.detach().cpu().numpy())**2)))
        print('iteration {} | loss: {}'.format(iteration, round(loss.item(), 4)))

        if (args.output_every is not None) and (iteration%args.output_every==0):
            image_k_array = image_k_tensor.detach().cpu().numpy()
            image_k_itkimg = itk.image_from_array(image_k_array)
            image_k_itkimg.CopyInformation(spect.like_itkimg)
            itk.imwrite(image_k_itkimg, output_ref.replace("%d", str(iteration)))

    image_k_array = image_k_tensor.detach().cpu().numpy()
    image_k_itkimg = itk.image_from_array(image_k_array)
    image_k_itkimg.CopyInformation(spect.like_itkimg)
    itk.imwrite(image_k_itkimg, output_ref.replace("%d", str(iteration)))



    np.save(output_ref.replace("_%d.mha","_lMSE.npy"), list_errors)
    np.save(output_ref.replace("_%d.mha","_lnPLL.npy"), list_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections")
    parser.add_argument("--likeimg")
    parser.add_argument("--fbprojectors", choices=['Joseph', 'JosephAttenuated', 'Zeng', 'Cuda'])
    parser.add_argument("--attmap_fn")
    parser.add_argument("--source")
    parser.add_argument("--outputfolder")
    parser.add_argument("--nsubsets", type=int)
    parser.add_argument("--output-every", type = int)
    parser.add_argument("--niter", type = int, default=10)
    parser.add_argument("--lr", type = float, default = 0.1)
    parser.add_argument("--n_layers", type = int, default = 3)
    parser.add_argument("--n_features", type = int, default = 128)
    parser.add_argument("--archi", type = str, default = "vanilla")
    parser.add_argument("--nfreqs", type = int, default = 10)
    parser.add_argument("--layer_acti", type = str, default = "relu")
    parser.add_argument("--final_acti", type = str, default = "relu")
    args = parser.parse_args()

    main()
