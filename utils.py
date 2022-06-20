import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ANN_Net(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        inp_size = int(3*self.N)
        self.fc1 = nn.Linear(inp_size, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1) 

    def af(self, x):
        return 1.0/(1.0+x**2)

    def forward(self, x):
        x = x.view(x.shape[0]*x.shape[1])
        x = self.af(self.fc1(x))
        x = self.af(self.fc2(x))
        x = self.fc3(x)
        return x


class ElemNet(nn.Module):
    def __init__(self, inp_num):
        super().__init__()
        self.inp_num = inp_num
        self.elemfc1 = nn.Linear(self.inp_num, 40)
        self.elemfc2 = nn.Linear(40, 40)
        self.elemfc3 = nn.Linear(40, 1)

    def af(self, x):
        return 1.0/(1.0+x**2)

    def forward(self, x):
        x = self.af(self.elemfc1(x))
        x = self.af(self.elemfc2(x))
        x = self.elemfc3(x)
        return x


class BPNN_Elec_Net(nn.Module):
    def __init__(self, eta, R_cutoff, zeta, lamda, Rs):
        super().__init__()
        self.eta = eta ; self.zeta = zeta ; self.lamda = lamda 
        self.R_cutoff = R_cutoff ; self.Rs = Rs 
        self.elem1 = ElemNet(inp_num = 2).to(torch.device("cuda:0"))
        self.elem2 = ElemNet(inp_num = 2).to(torch.device("cuda:0"))

    def get_parameters(self, atom_type, coord):
        atom1 = {
            'eta': 2.5,
            'zeta': 0.6,
            'R_s': 0.5,
            'lambda': 0.8,
        }
        atom2 = {
            'eta': 0.1,
            'zeta': 0.6,
            'R_s': 1.0,
            'lambda': 0.8,
        }

    def af(self, x):
        return 1.0/(1.0+x**2)

    def cutoff_function(self, Rij_mod):
        if Rij_mod<=self.R_cutoff:
            fc_ij = 0.5*(torch.cos((np.pi*Rij_mod)/self.R_cutoff)+1)
            return fc_ij
        else:
            return 0.0
        
    def radial_G(self, atom_coords, i):
        gi_arr = torch.tensor([0.0])
        for j in range(0, atom_coords.shape[0]):
            if i==j:
                continue
            Rij = atom_coords[i]-atom_coords[j]
            Rij_mod = torch.sqrt((Rij[0]**2)+(Rij[1]**2)+(Rij[2]**2))
            gi = torch.exp(-self.eta*((Rij_mod-self.Rs)**2))*self.cutoff_function(Rij_mod)
            gi_arr = torch.cat([gi_arr, torch.tensor([gi])])
        return torch.sum(gi_arr)

    def angular_G(self, atom_coords, i):
        gi_arr = torch.tensor([0.0])
        for j in range(0, atom_coords.shape[0]):
            for k in range(j+1, atom_coords.shape[0]):
                if ((i==j) or (j==k) or (i==k)):
                    continue
                Rij = atom_coords[i]-atom_coords[j]
                Rjk = atom_coords[j]-atom_coords[k]
                Rki = atom_coords[k]-atom_coords[i]
                Rij_mod = torch.sqrt((Rij[0]**2)+(Rij[1]**2)+(Rij[2]**2))
                Rjk_mod = torch.sqrt((Rjk[0]**2)+(Rjk[1]**2)+(Rjk[2]**2))
                Rki_mod = torch.sqrt((Rki[0]**2)+(Rki[1]**2)+(Rki[2]**2))
                cos_theta = (torch.dot(Rij, Rki))/(Rij_mod*Rki_mod)
                term1 = (1.0+self.lamda*cos_theta)**self.zeta
                term2 = torch.exp(-self.eta*(Rij_mod**2+Rjk_mod**2+Rki_mod**2))
                fc_ij = self.cutoff_function(Rij_mod)
                fc_jk = self.cutoff_function(Rjk_mod)
                fc_ki = self.cutoff_function(Rki_mod)
                term2 = term2*fc_ij*fc_jk*fc_ki
                gi = term1*term2
                gi_arr = torch.cat([gi_arr, torch.tensor([gi])])
        return (2.0**(1-self.zeta))*(torch.sum(gi_arr))
      
    def forward(self, x):
        qi = torch.tensor([0.0]).cuda()
        for i in range(0, x.shape[0]):
            ang = self.angular_G(x, i)
            rad = self.radial_G(x, i)
            Gi = torch.tensor([ang, rad]).float().cuda()
            if i!=8:
                qi_val = self.elem1.forward(Gi)
                qi = torch.cat([qi, qi_val])
            elif i==8:
                qi_val = self.elem2.forward(Gi)
                qi = torch.cat([qi, qi_val])
        return qi[1:]


class BPNN_Short_Net(nn.Module):
    def __init__(self, eta, R_cutoff, zeta, lamda, Rs):
        super().__init__()
        self.eta = eta ; self.zeta = zeta ; self.lamda = lamda 
        self.R_cutoff = R_cutoff ; self.Rs = Rs 
        self.elem1 = ElemNet(inp_num = 2).to(torch.device("cuda:0"))
        self.elem2 = ElemNet(inp_num = 2).to(torch.device("cuda:0"))

    def af(self, x):
        return 1.0/(1.0+x**2)

    def cutoff_function(self, Rij_mod):
        if Rij_mod<=self.R_cutoff:
            fc_ij = 0.5*(torch.cos((np.pi*Rij_mod)/self.R_cutoff)+1)
            return fc_ij
        else:
            return 0.0
        
    def radial_G(self, atom_coords, i):
        gi_arr = torch.tensor([0.0])
        for j in range(0, atom_coords.shape[0]):
            if i==j:
                continue
            Rij = atom_coords[i]-atom_coords[j]
            Rij_mod = torch.sqrt((Rij[0]**2)+(Rij[1]**2)+(Rij[2]**2))
            gi = torch.exp(-self.eta*((Rij_mod-self.Rs)**2))*self.cutoff_function(Rij_mod)
            gi_arr = torch.cat([gi_arr, torch.tensor([gi])])
        return torch.sum(gi_arr)

    def angular_G(self, atom_coords, i):
        gi_arr = torch.tensor([0.0])
        for j in range(0, atom_coords.shape[0]):
            for k in range(j+1, atom_coords.shape[0]):
                if ((i==j) or (j==k) or (i==k)):
                    continue
                Rij = atom_coords[i]-atom_coords[j]
                Rjk = atom_coords[j]-atom_coords[k]
                Rki = atom_coords[k]-atom_coords[i]
                Rij_mod = torch.sqrt((Rij[0]**2)+(Rij[1]**2)+(Rij[2]**2))
                Rjk_mod = torch.sqrt((Rjk[0]**2)+(Rjk[1]**2)+(Rjk[2]**2))
                Rki_mod = torch.sqrt((Rki[0]**2)+(Rki[1]**2)+(Rki[2]**2))
                cos_theta = (torch.dot(Rij, Rki))/(Rij_mod*Rki_mod)
                term1 = (1.0+self.lamda*cos_theta)**self.zeta
                term2 = torch.exp(-self.eta*(Rij_mod**2+Rjk_mod**2+Rki_mod**2))
                fc_ij = self.cutoff_function(Rij_mod)
                fc_jk = self.cutoff_function(Rjk_mod)
                fc_ki = self.cutoff_function(Rki_mod)
                term2 = term2*fc_ij*fc_jk*fc_ki
                gi = term1*term2
                gi_arr = torch.cat([gi_arr, torch.tensor([gi])])
        return (2.0**(1-self.zeta))*(torch.sum(gi_arr))
      
    def forward(self, x):
        Ei = torch.tensor([0.0]).cuda()
        for i in range(0, x.shape[0]):
            ang = self.angular_G(x, i)
            rad = self.radial_G(x, i)
            Gi = torch.tensor([ang, rad]).float().cuda()
            if i!=8:
                Ei_val = self.elem1.forward(Gi)
                Ei = torch.cat([Ei, Ei_val])
            elif i==8:
                Ei_val = self.elem2.forward(Gi)
                Ei = torch.cat([Ei, Ei_val])
        return torch.sum(Ei)


def electrostatic_energy(q_arr, atom_coords):
    elec_arr = torch.tensor([0.0])
    k = 0.00899
    for i in range(0, q_arr.shape[0]):
        for j in range(i+1, q_arr.shape[0]):
            Rij = atom_coords[i]-atom_coords[j]
            Rij_mod = torch.sqrt(Rij[0]**2 + Rij[1]**2 + Rij[2]**2)
            elec_energy = (k*q_arr[i]*q_arr[j])/Rij_mod
            elec_arr = torch.cat([elec_arr, torch.tensor([elec_energy])])
    return torch.sum(elec_arr)
