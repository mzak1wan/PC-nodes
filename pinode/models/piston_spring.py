import os
import time
import numpy as np

from torch import nn
import torch
import torch.nn.functional as F
from torch import optim

from util.util import format_elapsed_time

SAVE_NAME = os.path.join('data', "piston_spring_parameters_")

class PistonSpring(nn.Module):
    
    def __init__(self, nx, nu, h, parametrize_H_J=True, lr=0.1, verbose=4, try_to_load=True, name='default'):
        super().__init__()
        
        self.nx = nx
        self.nu = nu
        self.h = h
        self.parametrize_H_J = parametrize_H_J
        self.lr = lr
        self.verbose = verbose
        self.name = name
        
        if self.parametrize_H_J:
            self._initialize_H_J()
        else:
            self._initialize_parameters()
                
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"    
            
        self.times = []
        self.train_losses = []
        self.validation_losses = []
        
        self.loss = F.mse_loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        if try_to_load:
            self.load()
        
    def _initialize_H_J(self):
        
        self.K = nn.Parameter(torch.randn(self.nx, self.nx), True)
        self.b = nn.Parameter(torch.randn(self.nx, 1), True)
        self.G = nn.Parameter(torch.zeros(self.nx, self.nu), False)
        self.G[-1,0] = -1
        
        self.T = nn.Parameter(torch.randn(self.nx-2, ), True)
        self.gamma = nn.Sequential(nn.Linear(self.nx * 2, 1), nn.Sigmoid())
        
    def J(self, x):
        J = torch.zeros((x.shape[0], self.nx, self.nx))
        J[:,-1, 1:-1] = self.T.clone()
        J[:,1:-1, -1] = - self.T.clone()
        return J
    
    def dH_dx(self, x):
        K = torch.stack([self.K.clone() for _ in range(x.shape[0])], dim=0)
        KT = torch.stack([self.K.clone().permute(1, 0) for _ in range(x.shape[0])], dim=0)
        b = torch.stack([self.b.clone() for _ in range(x.shape[0])], dim=0)
        return torch.bmm(KT, torch.tanh(torch.bmm(K, x) + b))
    
    def R(self, x):
        # {S,H}_J = dH_dx[3]
        R = torch.zeros((x.shape[0], self.nx, self.nx))
        gamma = (self.gamma(torch.cat([x, self.dH_dx(x)], dim=1).squeeze(-1)) * self.dH_dx(x)[:,[3],0]).squeeze()
        R[:,-1,0] = - gamma.clone()
        R[:,0,-1] = gamma.clone()
        return R
    
    def RJ(self, x):
        return self.R(x).clone()+self.J(x).clone()
        
    def _initialize_parameters(self):
        
        self.m_system = nn.Parameter(torch.randn(1,1), True)
        self.A_piston = nn.Parameter(torch.randn(1,1), True)
        self.mu = nn.Parameter(torch.randn(1,1), True)
        self.n_mol = nn.Parameter(torch.randn(1,1), True)
        self.K_spring = nn.Parameter(torch.randn(1,1), True)
        self.mc_air = nn.Parameter(torch.randn(1,1), True)
        
        # Save their values
        self.params = [[float(torch.exp(x)) for x in [self.m_system, self.A_piston, self.mu, self.n_mol, self.K_spring, self.mc_air]]]
    
    def forward(self, x0, u_, init_=None):
        """
        x: temperatures of Room 1, 2, 3
        u: tensor with the external temperature, the solar irradiation, then 
            the power input of Room 1, 2, 3
        """
        
        x = x0.clone().permute(0,2,1)
        u = u_.clone()
        if not self.parametrize_H_J:
            S0 = init_[0]
            T0 = init_[1]
            V0 = init_[2]
        
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            u = u.unsqueeze(-1)
        elif len(u.shape) == 2:
            u = u.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        predictions = torch.empty((u.shape[0], u.shape[1], self.nx))
        predictions[:,0,:] = x.squeeze(-1).clone()
                    
        for t in range(u.shape[1]-1):
            
            # When we use NNs
            if self.parametrize_H_J:
                RJ = self.RJ(x[:,:,[0]]).clone()
                dH_dx = self.dH_dx(x[:,:,[0]]).clone()
                G = torch.stack([self.G.clone() for _ in range(u.shape[0])], dim=0)

                # Bulk of the work: x(t+1) = x(t) + h * [RJ * K^T*tanh(Kx+b) + G*u]
                x = x.clone() + self.h * (torch.bmm(RJ, dH_dx) + torch.bmm(G, u[:,[t],:].permute(0,2,1)))
            
            # Use the system equations
            else:
                # Define the speed, temperature, and pressure
                v = x[:,3,:].clone() / torch.exp(self.m_system.clone())
                T = T0 * torch.exp((x[:,0,:].clone()-S0) / torch.exp(self.mc_air.clone())) * torch.pow(x[:,1,:].clone() / V0, -torch.exp(self.n_mol.clone()) / torch.exp(self.mc_air.clone()))
                P = torch.exp(self.n_mol.clone()) * T / x[:,1,:].clone()         

                # Propagate in time with Euler discretization
                x[:, 0, :] = x[:, 0, :].clone() + self.h * (torch.exp(self.mu.clone()) / T * v * v) 
                x[:, 1, :] = x[:, 1, :].clone() + self.h * torch.exp(self.A_piston.clone()) * v
                x[:, 3, :] = x[:, 3, :].clone() + self.h * (P * torch.exp(self.A_piston.clone()).squeeze() - torch.exp(self.K_spring.clone()) * x[:, 2, :].clone() - torch.exp(self.mu.clone()) * v - u[:,[t],0].clone())
                x[:, 2, :] = x[:, 2, :].clone() + self.h * v
            
            # Store the step
            predictions[:, t+1, :] = x.squeeze(-1).clone()
            
        return predictions
            
    def fit(self, U, X, U_val, X_val, init=None, init_val=None, n_epochs=5000, print_each=100, only_pos=False):
        
        if not self.parametrize_H_J:
            assert init is not None, 'If H and J are explicitly defined, you need to provied initial conditions S0, T0, and V0 as `init`.'
            assert init_val is not None, 'If H and J are explicitly defined, you need to provied initial conditions S0, T0, and V0 as `init_val`.'
        
        if self.verbose > 0:
            print("\nTraining starts!\n")
        
        self.times.append(time.time())
            
        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf
        
        if self.verbose> 0:
            print('\t\tTraining loss\t\tValidation loss')
            
        if trained_epochs == 0:
            self.eval()
            predictions = self.forward(x0=X[:,0,:].reshape(X.shape[0],1,-1), u_=U, init_=init)
            if only_pos:
                train_loss = self.loss(predictions[:,:,2], X[:,:,2])
            else:
                train_loss = self.loss(predictions, X)

            predictions = self.forward(x0=X_val[:,0,:].reshape(X_val.shape[0],1,-1), u_=U_val, init_=init_val)
            if only_pos:
                val_loss = self.loss(predictions[:,:,2], X_val[:,:,2])
            else:
                val_loss = self.loss(predictions, X_val)

            self.train_losses.append(float(train_loss))
            self.validation_losses.append(float(val_loss))

            if self.verbose > 0:
                print(f"Epoch 0:\t  {float(train_loss):.2E}\t\t   {float(val_loss):.2E}")

        for epoch in range(trained_epochs, trained_epochs + n_epochs):
            
            # Training step
            self.train()
            train_losses = []

            self.optimizer.zero_grad()

            # Compute the loss of the batch and store it
            predictions = self.forward(x0=X[:,0,:].reshape(X.shape[0],1,-1), u_=U, init_=init)
            if only_pos:
                loss = self.loss(predictions[:,:,2], X[:,:,2])
            else:
                loss = self.loss(predictions, X)
            
            # Compute the gradients and take one step using the optimizer
            loss.backward()
            self.optimizer.step()

            self.train_losses.append(float(loss))
            
            # Validation step
            self.eval()
            predictions = self.forward(x0=X_val[:,0,:].reshape(X_val.shape[0],1,-1), u_=U_val, init_=init_val)
            if only_pos:
                loss = self.loss(predictions[:,:,2], X_val[:,:,2])
            else:
                loss = self.loss(predictions, X_val)

            self.validation_losses.append(float(loss))
            
            if not self.parametrize_H_J:
                self.params.append([float(torch.exp(x)) for x in [self.m_system, self.A_piston, self.mu, self.n_mol, self.K_spring, self.mc_air]])
            
            if np.isnan(float(loss)):
                print('\nExplosion! Restarting training\n')
                self.__init__(self.nx, self.nu, self.h, self.lr)
                self.fit(U, X, U_val, X_val, init, init_val, n_epochs)
                break
                
            # Compute the average loss of the training epoch and print it
            if epoch % print_each == print_each-1:
                print(f"Epoch {epoch + 1}:\t  {self.train_losses[-1]:.2E}\t\t   {float(loss):.2E}")
                
            if float(loss) < best_loss:
                self.save(name_to_add="best", verbose=0)
                best_loss = float(loss)
            
            self.times.append(time.time())

        if self.verbose > 0:
            print('\nTraining finished!')
            print(f"Average time elapsed by epoch:\t{np.mean(np.diff(self.times)): .2f}''")
            print(f"Total training time:\t\t{format_elapsed_time(self.times[0], self.times[-1])}")
            
        self.load()
            
    def save(self, name_to_add: str = None, verbose: int = 0):
        
        if verbose > 0:
            print(f"\nSaving the {name_to_add} model...")

        torch.save(
            {"model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "validation_losses": self.validation_losses,
                "times": self.times,
                },
            SAVE_NAME+self.name+'.pt',
        )

    def load(self):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(SAVE_NAME+self.name+'.pt'), f"The file {SAVE_NAME+self.name+'.pt'} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(SAVE_NAME+self.name+'.pt', map_location=lambda storage, loc: storage)

            # Put it into the model
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            self.times = checkpoint["times"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print("No existing model was found!")
            
            
            
class PistonSpringNN(PistonSpring):
    
    def __init__(self, sizes, nx, nu, h, parametrize_H_J=True, lr=0.1, verbose=4, try_to_load=True, name='default'):
        super().__init__(nx, nu, h, parametrize_H_J, lr, verbose, try_to_load=False, name=name)
        
        sizes = [nx+nu] + sizes
        self.net = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()) for i in range(len(sizes)-1)])
        self.out = nn.Linear(sizes[-1], nx)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        if try_to_load:
            self.load()
    
    def forward(self, x0, u_, init_=None):
        """
        x: temperatures of Room 1, 2, 3
        u: tensor with the external temperature, the solar irradiation, then 
            the power input of Room 1, 2, 3
        """
        
        x = x0.clone().permute(0,2,1)
        u = u_.clone()
        
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            u = u.unsqueeze(-1)
        elif len(u.shape) == 2:
            u = u.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        predictions = torch.empty((u.shape[0], u.shape[1], self.nx))
        predictions[:,0,:] = x.squeeze(-1).clone()
                    
        for t in range(u.shape[1]-1):
            
            temp = torch.cat([x, u[:,[t],:]], axis=1).squeeze()
            for layer in self.net:
                temp = layer(temp)
            x = x.clone() + self.h * self.out(temp).unsqueeze(-1)
            
            # Store the step
            predictions[:, t+1, :] = x.squeeze(-1).clone()
            
        return predictions
    