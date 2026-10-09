import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs as sparse_eigs
from sklearn.preprocessing import StandardScaler
import antropy as ant
from matplotlib.backends.backend_pdf import PdfPages
import time
import warnings

# =============================================================================
# 1. 설정
# =============================================================================
BEST_PARAMS = {
    'N_r': 1000,
    'average_degree': 113,
    'spectral_radius': 1.5,
    'leakage_rate': 0.01,
    'regularization': 0.00183011,
    'input_scaling': 5.0,
    'param_scales': [0.1, 0.1],
    'param_biases': [-1.6317999432230639, 0.0517568554782466]
}

# User requested 7 points: 0.97 ~ 0.997
TRAIN_K_POINTS = [0.97, 0.975, 0.98, 0.985, 0.99]
N_ENSEMBLE = 1000
AMP_FACTOR = 1.0        
RAMP_STEPS = 0
WARMUP_STEPS = 500
TOTAL_GEN_STEPS = 6000  # Increased for robust sampling  

# =============================================================================
# 2. 시스템 및 헬퍼
# =============================================================================
def food_chain_system(t, state, K):
    R, C, P = state
    R = max(0, R); C = max(0, C); P = max(0, P)
    
    params = {"xc": 0.4, "yc": 2.009, "xp": 0.08, "yp": 2.876, "R0": 0.16129, "C0": 0.5}
    dRdt = R * (1 - R / K) - (params["xc"] * params["yc"] * C * R) / (R + params["R0"])
    dCdt = params["xc"] * C * ((params["yc"] * R) / (R + params["R0"]) - 1) - (params["xp"] * params["yp"] * P * C) / (C + params["C0"])
    dPdt = params["xp"] * P * ((params["yp"] * C) / (C + params["C0"]) - 1)
    return [dRdt, dCdt, dPdt]

def generate_robust_data(K, n_steps, dt=1.0, n_discard=2000):
    t_span = (0, (n_steps + n_discard) * dt)
    t_eval = np.arange(t_span[0], t_span[1] + 0.1, 0.1) 
    args = (K,)
    
    max_retries = 30 
    for i in range(max_retries):
        init = [0.4*np.random.rand()+0.6, 0.4*np.random.rand()+0.15, 0.1*np.random.rand()+0.6]
        try:
            sol = solve_ivp(food_chain_system, t_span, init, args=args, method='RK45', t_eval=t_eval)
            data = sol.y.T[::10] 
            if len(data) >= n_discard + n_steps:
                valid_data = data[n_discard : n_discard + n_steps]
                # Relaxed threshold to allow near-collapse trajectories
                # This is crucial for RC to learn the boundary of extinction
                if np.min(valid_data[:, 2]) > 0.01 and valid_data[-1, 2] > 0.01:
                    return valid_data
        except: continue
    return np.zeros((n_steps, 3))

def get_stats(series):
    if len(series) < 10: return np.zeros(2)
    s = np.std(series)
    try: e = ant.perm_entropy(series, normalize=True)
    except: e = 0.0
    return np.array([s, e])

# =============================================================================
# 3. StatRC
# =============================================================================
class StatRC:
    def __init__(self, params):
        self.p = params
        np.random.seed(None) 
        self.W_in = (np.random.rand(self.p['N_r'], 3)*2-1) * self.p['input_scaling']
        self.W_stat = (np.random.rand(self.p['N_r'], 2)*2-1)
        
        dens = self.p['average_degree'] / self.p['N_r']
        if dens > 1.0: dens = 1.0
        A = sparse_random(self.p['N_r'], self.p['N_r'], density=dens, format='csr', data_rvs=np.random.randn)
        A = (A + A.T) / 2
        try:
            ev = sparse_eigs(A, k=1, which='LM', return_eigenvectors=False)
            self.A = A * (self.p['spectral_radius'] / np.abs(ev[0]))
        except: self.A = A * self.p['spectral_radius']
        
        self.s_vec = np.array(self.p['param_scales']).reshape(2,1)
        self.b_vec = np.array(self.p['param_biases']).reshape(2,1)
        self.W_out = None

    def _get_aug(self, r):
        r_aug = r.copy(); r_aug[1::2] = r_aug[1::2] ** 2
        return r_aug

    def train(self, u_train, s_train, steps):
        R, Y = [], []
        for u, s in zip(u_train, s_train):
            if np.all(u==0): continue
            bias = self.W_stat @ (self.s_vec * s.reshape(2,1) + self.b_vec)
            r = np.zeros((self.p['N_r'], 1))
            r_list, y_list = [], []
            for t in range(min(len(u)-1, steps)):
                inp = u[t].reshape(3,1)
                r = (1 - self.p['leakage_rate']) * r + self.p['leakage_rate'] * np.tanh(self.A @ r + self.W_in @ inp + bias)
                if t > 100:
                    r_aug = self._get_aug(r)
                    r_list.append(r_aug[:,0]); y_list.append(u[t+1])
            if len(r_list) > 0: R.append(np.array(r_list).T); Y.append(np.array(y_list).T)
        
        if not R: return False
        R_all = np.hstack(R); Y_all = np.hstack(Y)
        I = np.eye(self.p['N_r'])
        try:
            self.W_out = Y_all @ R_all.T @ np.linalg.inv(R_all @ R_all.T + self.p['regularization'] * I)
            return True
        except: return False

    def predict_with_ramp(self, u_warmup, s_warmup, s_target, steps, ramp_steps=0):
        bias_start = self.W_stat @ (self.s_vec * s_warmup.reshape(2,1) + self.b_vec)
        r = np.zeros((self.p['N_r'], 1))
        for t in range(len(u_warmup)):
            inp = u_warmup[t].reshape(3,1)
            r = (1 - self.p['leakage_rate']) * r + self.p['leakage_rate'] * np.tanh(self.A @ r + self.W_in @ inp + bias_start)
            
        preds = []
        curr = u_warmup[-1].reshape(3,1)
        bias_end = self.W_stat @ (self.s_vec * s_target.reshape(2,1) + self.b_vec)
        
        for t in range(steps):
            if t < ramp_steps:
                alpha = t / ramp_steps
                current_bias = (1 - alpha) * bias_start + alpha * bias_end
            else:
                current_bias = bias_end
            
            r = (1 - self.p['leakage_rate']) * r + self.p['leakage_rate'] * np.tanh(self.A @ r + self.W_in @ curr + current_bias)
            r_aug = self._get_aug(r)
            out = self.W_out @ r_aug
            out[out<0] = 0
            
            preds.append(out.flatten())
            curr = out.reshape(3,1)
            
            if out[2] < 0.01: 
                remaining = steps - len(preds)
                preds.extend([np.zeros(3)] * remaining)
                break
        return np.array(preds)

# =============================================================================
# 4. 메인 실험: Survival Rate Analysis
# =============================================================================
