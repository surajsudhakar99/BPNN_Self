from matplotlib import testing
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import utils

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_points(file_name):
    i = 0
    coord = torch.zeros(9,3)
    f = open(file_name, 'r')
    for line in f:
        coord[i] = torch.tensor([float(item) for item in line.strip().split(' ')]) 
        i += 1
    return coord

coords = get_points('input.dat')
#print(coords)
#sys.exit()

need_testing = False
nt = input('Do you want to test the model (T/F): ')
if nt == 'T': 
    need_testing = True

dft_energy = 5.234 #units
dft_charges = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0])

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111,projection='3d')
ax.scatter3D(coords[:,0],coords[:,1],coords[:,2], s=2**7)
fig.savefig('coords.png')

if torch.cuda.is_available(): 
    print(f'Cuda availability: {torch.cuda.is_available()}')
    print(f'Cuda device: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else: 
    sys.exit('Cuda is not available!')
    


#------------------------------Training Electrostatic Potential Network-----------------------------------#
bpnn_elec_net = utils.BPNN_Elec_Net(eta=0.1, R_cutoff=3.0, zeta=0.6, lamda=0.8, Rs=1.3)

loss_arr = []
coords_gpu = coords.cuda()
dft_charges_gpu = torch.tensor(dft_charges).float().cuda()

if torch.cuda.is_available():
    bpnn_elec_net.to(torch.device("cuda:0"))
    print('Running on GPU')

    optimizer = optim.Adam(bpnn_elec_net.parameters(), lr=0.001)

    EPOCHS_elec = 500
    bpnn_elec_net.train()

    for epochs in range(EPOCHS_elec):    
        bpnn_elec_net.zero_grad()
        output = bpnn_elec_net.forward(coords_gpu)
        loss = F.mse_loss(output, dft_charges_gpu)
        loss.backward()
        optimizer.step()
            
        print(f'"ElecNet Training" {epochs+1}: loss = {loss}')
        loss_arr.append(loss)
        
print("ElecNets Training Done!")
#----------------------------------------------------------------------------------------------------------#

qi_out = bpnn_elec_net(coords_gpu).clone().detach().cpu()
srp_energy = dft_energy-utils.electrostatic_energy(qi_out, coords)
print(f'Short Range Potential (SRP) Energy = {srp_energy}')

#------------------------------Training Short Range Potential Network-----------------------------------#
bpnn_short_net = utils.BPNN_Short_Net(eta=0.1, R_cutoff=3.0, zeta=0.6, lamda=0.8, Rs=1.3)

loss_arr = []
coords_gpu = coords.cuda()
srp_energy_gpu = torch.tensor(srp_energy).float().cuda()

if torch.cuda.is_available():
    bpnn_short_net.to(torch.device("cuda:0"))
    print('Running on GPU')

    optimizer = optim.Adam(bpnn_short_net.parameters(), lr=0.001)

    EPOCHS = 250
    bpnn_short_net.train()

    for epochs in range(EPOCHS):    
        bpnn_short_net.zero_grad()
        output = bpnn_short_net.forward(coords_gpu)
        loss = F.mse_loss(output, srp_energy_gpu)
        loss.backward()
        optimizer.step()
            
        print(f'"ShortRangePotential Training" {epochs+1}: loss = {loss}')
        loss_arr.append(loss)
        
print("Training Done!")
#-------------------------------------------------------------------------------------------------------#

#------------------------------Training Simple ANN Network for SRP-----------------------------------#
ann_net = utils.ANN_Net(N=coords.shape[0])

loss_arr = []

if torch.cuda.is_available():
    ann_net.to(torch.device("cuda:0"))
    print('Running on GPU')

    coords_gpu = coords.cuda()
    srp_energy_gpu = torch.tensor(srp_energy).float().cuda()
    
    optimizer = optim.Adam(ann_net.parameters(), lr=0.001)

    EPOCHS = 300
    ann_net.train()

    for epochs in range(EPOCHS):    
        ann_net.zero_grad()
        output = ann_net.forward(coords_gpu)
        loss = F.mse_loss(output, srp_energy_gpu)
        loss.backward()
        optimizer.step()
            
        print(f'{epochs+1}: loss = {loss}')
        loss_arr.append(loss)
        
print("Training Done!")
#----------------------------------------------------------------------------------------------------#

# Testing the trained models (Simple ANN v/s BPNN) in terms of symmetry in input coordinates
 

if need_testing == True:
    coords2 = torch.tensor([
                    [2.0, 2.0, 0.0], # interchanged coordinates
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0], # interchanged coordinates
                    [2.0, 0.0, 2.0],
                    [0.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0]
    ])

    coords3 = torch.tensor([
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0], # interchanged coordinates
                    [0.0, 0.0, 2.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [0.0, 2.0, 2.0],
                    [0.0, 2.0, 0.0], # interchanged coordinates
                    [1.0, 1.0, 1.0]
    ])

    print(f'coords2 = {coords2}')
    print(f'coords3 = {coords3}')
    coords2_gpu = coords2.float().cuda()
    coords3_gpu = coords3.float().cuda()

    m_ann2 = ann_net.forward(coords2_gpu)
    m_ann3 = ann_net.forward(coords3_gpu)
    print(f'The SRP Energy using simple ANN for coords2 = {m_ann2}')
    print(f'The SRP Energy using simple ANN for coords3 = {m_ann3}')

    m_bpnn2 = bpnn_short_net.forward(coords2_gpu)
    m_bpnn3 = bpnn_short_net.forward(coords3_gpu)
    print(f'The SRP Energy incorporating symmetry functions for coords2 = {m_bpnn2}')
    print(f'The SRP Energy incorporating symmetry functions for coords3 = {m_bpnn3}')
    print('-------------------------------------------------------------------------------------------------------------------------')
    print('Note that the simple ANN treats coords2 and coords3 differently! But the BP symmetry functions considers them to be same!')
    print('-------------------------------------------------------------------------------------------------------------------------')
