import numpy as np
from copy import deepcopy
import torch

def force(t):
    return ((int(t) % 6 <= 2) * 1)*2 if int(t) % 4 <= 1 else ((int(t) % 6 <= 2) * 2 - 0.5) * 1

def piston_spring(x, t, m_system, A_piston, mu, n_mol, K_spring, mc_air, S0, V0, T0):
    """Dynamics of a piston-spring system with an external force
    """
    
    # Unpack the state (Entropy, Volume, Position, Momentum)
    S, V, z, p = x
    
    # Define the speed and the derivative of position and volume
    v = p / m_system
    dzdt = v
    dVdt = A_piston * v
            
    # Define the temperature and pressure
    T = T0 * np.exp((S-S0) / mc_air) * np.power(V/V0, -n_mol/mc_air)
    P = n_mol * T / V
    
    # Derivative of the entropy and momentum
    dSdt = 1/T * mu * v * v
    dpdt = -K_spring * z + P * A_piston - mu * v - force(t) 
                
    return [dSdt, dVdt, dzdt, dpdt]

def piston_spring_args():
    """Arguments of the pistong-spring system used in this work
    """
    
    # Initial state
    S = 0.
    V = 0.01
    z = 0.3
    p = 0.
    x0 = [S, V, z, p]
    
    # Initial conditions
    S0 = S
    V0 = V
    T0 = 290
    
    # system parameters
    m_system = 5              # Mass of the system
    A_piston = x0[1] / x0[2]  # Area of the piston
    mu = 1                    # Friction coefficient
    n_mol = 8.31441 * 0.001   # nR in pV = nRT 
    K_spring = 10             # Spring constant
    mc_air = 500              # mc in the thermal energy E = mcT
    
    # Simulation setup
    h = 0.01    # Step size
    N = 10000   # Number of steps
    
    return m_system, A_piston, mu, n_mol, K_spring, mc_air, x0, S0, V0, T0, h, N

def prepare_data(t, sol, sol_noisy, cut=250):
    
    # Train data
    U = torch.stack([torch.Tensor([force(x) for x in t])], dim=-1).unsqueeze(0)
    X = torch.Tensor(sol_noisy).unsqueeze(0)

    # Validation data
    U_val = deepcopy(U)
    X_val = torch.Tensor(sol).unsqueeze(0)
    
    # Save this data for the arx model
    U_arx = U.squeeze().detach().numpy().copy()
    X_arx = X.squeeze().detach().numpy().copy()
    U_val_arx = U_val.squeeze().detach().numpy().copy()
    X_val_arx = X_val.squeeze().detach().numpy().copy()
    
    # Cut the data in sequences of the wanted length
    U = torch.stack([U[0, cut*i: cut*(i+1), :] for i in range(int(sol.shape[0]/cut))], dim=0)
    X = torch.stack([X[0, cut*i: cut*(i+1), :] for i in range(int(sol.shape[0]/cut))], dim=0)
    U_val = torch.stack([U_val[0, cut*i: cut*(i+1), :] for i in range(int(sol.shape[0]/cut))], dim=0)
    X_val = torch.stack([X_val[0, cut*i: cut*(i+1), :] for i in range(int(sol.shape[0]/cut))], dim=0)
    
    return U, X, U_val, X_val, U_arx, X_arx, U_val_arx, X_val_arx