
"""

@author: sshto
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splev
from scipy.integrate import cumtrapz

np.random.seed(42)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def b_spline_basis(U, degree=3, num_knots=8):

    knots = np.concatenate(([0] * degree, np.linspace(0, 1, num_knots - degree + 2), [1] * degree))
    
    basis = []
    for i in range(len(knots) - degree - 1):
        coeffs = np.zeros(len(knots) - degree - 1)
        coeffs[i] = 1  
        spline = BSpline(knots, coeffs, degree)
        basis.append(splev(U, spline))
    
    return np.array(basis)


def i_spline_basis(U, degree=3, num_knots=8):
    b_splines = b_spline_basis(U, degree, num_knots)
    

    assert len(U) > 1, "U must have more than one element for cumtrapz to work."
    assert b_splines.shape[1] == len(U), "b_splines must have the same number of columns as U's length."
    
    i_splines = np.array([cumtrapz(m, U, initial=0) for m in b_splines])
    

    return i_splines / i_splines[:, -1][:, np.newaxis]



def g(U, b_params, degree=3, num_knots=8):
    U = np.atleast_1d(U) 
    basis_functions = i_spline_basis(U, degree, num_knots)
    
    
    basis_functions = basis_functions[:len(b_params), :]
    
    return -2 * np.sum(b_params[:, np.newaxis] * basis_functions, axis=0) + 1

def f(X, a1, a2, a3, a4):
    return (1 - 2*X + a1 * (1 - (2*X - 1)**2)) * (a2 + a3 * (2*X - 1)**2 + a4 * (2*X - 1)**6)

def delta(X, U, a1, a2, a3, a4, b_params, k):
    return -1 + 2 / (1 + np.exp(-k * (f(X, a1, a2, a3, a4) + g(U, b_params))))

def D(Baseline, X, U, Delta, a1, a2, a3, a4, b_params, k, lamm):
    d = delta(X, U, a1, a2, a3, a4, b_params, k)
    return Baseline + d * Delta * ((1 / (1 + np.exp(-lamm * d))) * (1 - Baseline) + 
                                    (1 / (1 + np.exp(lamm * d))) * Baseline)

iteration_counter = 0
def log_likelihood(params, U, Baseline, D_obs, Delta):
    global iteration_counter
    iteration_counter += 1

    C, a1, a2, a3, a4, k, lamm = params[:7]
    b_params = params[7:]


    X_init = 0.5
    n_samples = len(Baseline)
    n_steps = 10
    X = np.full(n_samples, X_init)
    X[0] = X_init
    D_pred = np.zeros(n_samples)
    dt = 0.1
    
    for i in range(1, n_samples):
        for _ in range(n_steps):
            X[i] += X_init + dt * (D_obs[i] - Baseline[i]) / C 
        X[i] = np.clip(X[i], 0, 1) 
        X_init = X[i]

    D_pred = D(Baseline, X, U, Delta, a1, a2, a3, a4, b_params, k, lamm)
    likelihood = -np.sum((D_obs - D_pred) ** 2)  
    penalty_weight = 100  
    constraint_violation = abs(np.sum(b_params) - 1) + abs(a2 + a3 + a4 - 1) 
    

    return -likelihood + penalty_weight * constraint_violation - 50*(a3 + 3*a4) 

def callback(params):
    print(f"Step Update: Params = {params}")

def estimate_parameters(U, Baseline, D_obs, Delta, num_initializations=20):
    best_params = None
    best_likelihood = np.inf  

    bounds = [
        (1, 10),   # C
        (-0.5, 0.5), # a1
        (0, 1),      # a2
        (-1, 1),     # a3
        (-1, 1),     # a4
        (1, 10),     # k
        (400, 500)      # lamm
    ] + [(0.01, None)] * 7  # Unbounded b_params (handled separately)

    

    initial_guess = None  

    initial_params = np.zeros((num_initializations,14))
    initial_params[:,0] = np.random.uniform(1, 10, num_initializations)  # C
    initial_params[:, 1] = np.random.uniform(-0.5, 0.5, num_initializations)  # a1
    random_numbers = np.random.rand(num_initializations, 3)
    initial_params[:,2:5] = random_numbers / np.sum(random_numbers, axis=1, keepdims=True)  # Ensure a2 + a3 + a4 = 1
    initial_params[:,5] = np.random.uniform(1, 10, num_initializations)  # k
    initial_params[:,6] = np.random.uniform(400, 500, num_initializations)  # lamm
    random_numbers = np.random.rand(num_initializations, 7)
    initial_params[:, 7:14] = random_numbers / np.sum(random_numbers,axis=1, keepdims=True)  # Normalize b_params
    
    
        


    for i in range(num_initializations):
        

        if initial_guess is None:
            initial_guess = initial_params[i]  

        print(f"\nStarting Optimization {i+1}/{num_initializations} with Initial Params: {initial_params[i]}")

        result = opt.minimize(log_likelihood, initial_params[i], args=(U, Baseline, D_obs, Delta),
                              method='Nelder-Mead', bounds=bounds, 
                              tol=1e-5,              
                              options={
                                  'maxiter': 10000,   
                                  'maxfev': 10000,     
                                  'disp': False        
                                  })# SLSQP # Nelder-Mead #L-BFGS-B #COBYLA


        print("###############################################################")
        print("###############################################################")

        print(f"Initialization {i+1}/{num_initializations}: Success = {result.success}, Final Likelihood = {result.fun:.6f}")
        print("Result message:", result.message)
        print(f"Estimated Parameters {i+1}/{num_initializations}: {result.x}")
        
        if result.success and result.fun < best_likelihood:
            best_likelihood = result.fun
            best_params = result.x
            
    
    
    print("Best cost function:", best_likelihood)
    return best_params if best_params is not None else initial_guess  



def hourly_average(arr, group_size=12):
    n = len(arr) - (len(arr) % group_size)
    arr = arr[:n]
    return arr.reshape(-1, group_size).mean(axis=1)

def load_data(csv_file, split_ratio=0.7):
    data = pd.read_csv(csv_file)
    
    Baseline_raw = data.iloc[:, 7].values#7
    D_obs_raw = data.iloc[:, 6].values#6
    U_raw = data.iloc[:, 1].values

    Baseline_hourly = hourly_average(Baseline_raw)
    D_obs_hourly = hourly_average(D_obs_raw)
    U_hourly = hourly_average(U_raw)

    Baseline = normalize(Baseline_hourly)
    D_obs = normalize(D_obs_hourly)
    U = normalize(U_hourly)
    
    split_index = int(len(U) * split_ratio)
    
    U_id, U_val = U[:split_index], U[split_index:]
    Baseline_id, Baseline_val = Baseline[:split_index], Baseline[split_index:]
    D_obs_id, D_obs_val = D_obs[:split_index], D_obs[split_index:]
    
    return U_id, Baseline_id, D_obs_id, U_val, Baseline_val, D_obs_val


csv_file = 'FF_data_hp.csv' 
U_id, Baseline_id, D_obs_id, U_val, Baseline_val, D_obs_val = load_data(csv_file)
Delta = 1

print("###############################################################")
print("###############################################################")
print("###############################################################")
print("###############################################################")
print("###############################################################")
print("###############################################################")

estimated_params = estimate_parameters(U_id, Baseline_id, D_obs_id, Delta)
print("Final Estimated Parameters:", estimated_params)

param_names = ["C", "a1", "a2", "a3", "a4", "k", "lamm"] + [f"b{i+1}" for i in range(7)]
param_table = pd.DataFrame({"Parameter": param_names, "Value": estimated_params})
print(param_table)


U_values = np.linspace(0, 1, 100)
X_values = np.linspace(0, 1, 100)
f_values = f(X_values, *estimated_params[1:5])
g_values = g(U_values, estimated_params[7:])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(X_values, f_values, label='f(X)')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('Function f(X)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(U_values, g_values, label='g(U)', color='r')
plt.xlabel('U')
plt.ylabel('g(U)')
plt.title('Function g(U)')
plt.legend()

plt.show()

########################################################################
############ Validation ################################################
########################################################################
from scipy.interpolate import interp1d


interp_g = interp1d(U_values, g_values, kind='linear', fill_value="extrapolate")

g_at_0_25 = interp_g(0.25)

X_init = 0.5
n_samples = len(Baseline_val)
n_steps = 10
X = np.full(n_samples, X_init)
Xs = np.full(n_steps+1, X_init)
delt = np.full(n_samples, X_init)
D_pred = np.zeros(n_samples)
d = np.zeros(n_samples)
X[0] = X_init
dt = 0.1



print("NaN in Baseline_val:", np.any(np.isnan(Baseline_val)))
print("NaN in D_obs_val:", np.any(np.isnan(D_obs_val)))
print("NaN in U_val:", np.any(np.isnan(U_val)))


b_params = np.array(estimated_params[7:])  
for i in range(n_samples):
    Xs[0] = X[i]
    d[i] = -1 + 2 / (1 + np.exp(-estimated_params[5] * (f(Xs[0], estimated_params[1], estimated_params[2], estimated_params[3], estimated_params[4]) + interp_g(U_val[i]))))
    D_pred[i] = Baseline_val[i] + d[i] * Delta * ((1 / (1 + np.exp(-estimated_params[6] * d[i]))) * (1 - Baseline_val[i]) + 
                                (1 / (1 + np.exp(estimated_params[6] * d[i]))) * Baseline_val[i])
    for ii in range(n_steps):        
        Xs[ii+1] = Xs[ii] + dt * (D_pred[i] - Baseline_val[i]) / estimated_params[0]        
        
    
    X[i] = np.clip(Xs[ii+1], 0, 1)
    

        
        
        


plt.figure(figsize=(12, 5))
plt.plot(np.linspace(0,n_samples*0.1, n_samples), X, label='X over time', color='g')
plt.xlabel('Time')
plt.ylabel('X')
plt.title('Evolution of X')
plt.legend()

plt.tight_layout()
plt.show()



mean_D_obs_val = np.mean(D_obs_val)
ss_total = np.sum((D_obs_val - mean_D_obs_val) ** 2)
ss_residual = np.sum((D_obs_val - D_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
rmse = np.sqrt(ss_residual / len(D_obs_val))

print(f'R^2 Score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

plt.figure(figsize=(8, 5))
plt.plot(D_obs_val, label='D_obs_val', linestyle='solid', color='blue')
plt.plot(D_pred, label='D_pred', linestyle='solid', color='red')
plt.plot(Baseline_val, label='Baseline_val', linestyle='solid', color='green')
plt.plot(U_val, label='U_val', linestyle='dashed', color='black')
plt.plot(d, label='delta', linestyle='dashed', color='purple')
plt.plot(X, label='X', linestyle='dashed', color='pink')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Value')
plt.legend()
plt.title('Validation Data vs Predicted Data')
plt.show()


