"""
Voltage Collapse Model
4D ODE system modeling power grid voltage collapse
Based on Kong et al. 2021 paper
"""

import numpy as np
from scipy.integrate import solve_ivp


class VoltageCollapseModel:
    """
    4D Voltage Collapse Model for power grid dynamics
    
    State variables:
        x[0] = deltam: generator angle
        x[1] = omega: frequency deviation
        x[2] = delta: load angle
        x[3] = v: voltage magnitude
    
    Bifurcation parameter:
        Q1: reactive power demand
    """
    
    def __init__(self, Q1=2.9897):
        """
        Initialize voltage collapse model with physical parameters
        
        Parameters:
        -----------
        Q1 : float
            Reactive power demand (bifurcation parameter)
        """
        self.Q1 = Q1
        
        # Physical parameters (from MATLAB code)
        self.Kpw = 0.4
        self.Kpv = 0.3
        self.Kqw = -0.03
        self.Kqv = -2.8
        self.Kqv2 = 2.1
        
        self.T = 8.5
        self.Pm = 1.0
        self.dm = 0.05
        self.M = 0.01464
        self.Em = 1.05
        
        self.Y0 = 3.33
        self.Ym = 5.0
        
        self.P0 = 0.6
        self.Q0 = 1.3
        self.P1 = 0.0
        
        # Additional parameters for power calculations
        self.C = 3.5
        self.E0 = 1.0
        self.theta0 = 0.0
        self.thetam = 0.0
        
        # Calculate derived parameters
        theta0p = self.theta0 + np.arctan(
            self.C * np.sin(self.theta0) / self.Y0 / 
            (1 - self.C * np.cos(self.theta0) / self.Y0)
        )
        self.Y0p = self.Y0 * np.sqrt(
            1 + self.C**2 / self.Y0**2 - 
            2 * self.C / self.Y0 * np.cos(self.theta0)
        )
        self.E0p = self.E0 / np.sqrt(
            1 + self.C**2 / self.Y0**2 - 
            2 * self.C / self.Y0 * np.cos(self.theta0)
        )
    
    def _calculate_power(self, x):
        """
        Calculate active (P) and reactive (Q) power
        
        Parameters:
        -----------
        x : array-like
            State vector [deltam, omega, delta, v]
            
        Returns:
        --------
        P, Q : float
            Active and reactive power
        """
        deltam, omega, delta, v = x
        
        # Power equations (from MATLAB code)
        P = (-self.E0p * v * self.Y0p * np.sin(delta) + 
             self.Em * v * self.Ym * np.sin(deltam - delta))
        
        Q = (self.E0p * v * self.Y0p * np.cos(delta) + 
             self.Em * v * self.Ym * np.cos(deltam - delta) - 
             (self.Y0p + self.Ym) * v**2)
        
        return P, Q
    
    def voltage_collapse_ode(self, t, x):
        """
        4D ODE system for voltage collapse
        
        dx[0]/dt = omega
        dx[1]/dt = (-dm*omega + Pm - Em*v*Ym*sin(deltam-delta)) / M
        dx[2]/dt = (-Kqv2*v^2 - Kqv*v + Q - Q0 - Q1) / Kqw
        dx[3]/dt = (Kpw*Kqv2*v^2 + (Kpw*Kqv-Kqw*Kpv)*v + 
                    Kqw*(P-P0-P1) - Kpw*(Q-Q0-Q1)) / (T*Kqw*Kpv)
        
        Parameters:
        -----------
        t : float
            Time (not used, autonomous system)
        x : array-like
            State vector [deltam, omega, delta, v]
            
        Returns:
        --------
        dxdt : ndarray
            Time derivatives
        """
        deltam, omega, delta, v = x
        
        # Calculate power
        P, Q = self._calculate_power(x)
        
        # ODE equations
        dxdt = np.zeros(4)
        
        dxdt[0] = omega
        
        dxdt[1] = ((-self.dm * omega + self.Pm - 
                    self.Em * v * self.Ym * np.sin(deltam - delta)) / self.M)
        
        dxdt[2] = ((-self.Kqv2 * v**2 - self.Kqv * v + Q - self.Q0 - self.Q1) / 
                   self.Kqw)
        
        dxdt[3] = ((self.Kpw * self.Kqv2 * v**2 + 
                    (self.Kpw * self.Kqv - self.Kqw * self.Kpv) * v + 
                    self.Kqw * (P - self.P0 - self.P1) - 
                    self.Kpw * (Q - self.Q0 - self.Q1)) / 
                   (self.T * self.Kqw * self.Kpv))
        
        return dxdt
    
    def simulate(self, x0, t_span, dt=0.01, method='LSODA', verbose=False, 
                 max_step=np.inf, rtol=1e-6, atol=1e-9):
        """
        Simulate voltage collapse system
        
        Parameters:
        -----------
        x0 : array-like
            Initial condition [deltam, omega, delta, v]
        t_span : tuple
            Time span (t_start, t_end)
        dt : float
            Time step for output
        method : str
            Integration method ('LSODA', 'BDF', 'RK45', etc.)
            Default 'LSODA' is good for potentially stiff systems
        verbose : bool
            Print progress information
        max_step : float
            Maximum allowed step size (to prevent hanging)
        rtol, atol : float
            Relative and absolute tolerances
            
        Returns:
        --------
        t : ndarray
            Time points
        x : ndarray
            State trajectory (n_points, 4)
        """
        if verbose:
            print(f"  Starting simulation: t={t_span[0]} to {t_span[1]}, dt={dt}")
            print(f"  Initial condition: {x0}")
        
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        # Suppress all warnings during integration
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                self.voltage_collapse_ode,
                t_span,
                x0,
                method=method,
                t_eval=t_eval,
                dense_output=False,  # Faster without dense output
                max_step=max_step,  # Prevent hanging on difficult regions
                rtol=rtol,
                atol=atol
            )
        
        if verbose:
            print(f"  Simulation complete: {sol.nfev} function evaluations")
            print(f"  Success: {sol.success}, Message: {sol.message}")
        
        if not sol.success:
            print(f"Warning: Integration failed - {sol.message}")
        
        return sol.t, sol.y.T
    
    def get_initial_condition(self, seed=None):
        """
        Generate random initial condition based on MATLAB code
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        x0 : ndarray
            Initial condition [deltam, omega, delta, v]
        """
        if seed is not None:
            np.random.seed(seed)
        
        # From MATLAB: x0 = [0.13*rand+0.17; 0.1*rand; 0.1*rand+0.05; 0.05*rand+0.83]
        x0 = np.array([
            0.13 * np.random.rand() + 0.17,  # deltam in [0.17, 0.30]
            0.1 * np.random.rand(),           # omega in [0, 0.1]
            0.1 * np.random.rand() + 0.05,    # delta in [0.05, 0.15]
            0.05 * np.random.rand() + 0.83    # v in [0.83, 0.88]
        ])
        
        return x0


def test_voltage_model():
    """Test voltage collapse model with different Q1 values"""
    import matplotlib.pyplot as plt
    
    # Test parameters from MATLAB code
    Q1_train = [2.98968, 2.98973, 2.98978]
    Q1_test = 2.98983
    
    t_span = (0, 100)
    dt = 0.05
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Voltage Collapse System Dynamics', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'red']
    Q1_values = Q1_train + [Q1_test]
    labels = [f'Q1={q:.5f} (train)' for q in Q1_train] + \
             [f'Q1={Q1_test:.5f} (test)']
    
    for idx, (Q1, color, label) in enumerate(zip(Q1_values, colors, labels)):
        model = VoltageCollapseModel(Q1=Q1)
        x0 = model.get_initial_condition(seed=42)
        
        print(f"\nSimulating {label}...")
        t, x = model.simulate(x0, t_span, dt=dt, verbose=True)
        
        # Plot state variables
        axes[0, 0].plot(t, x[:, 0], color=color, label=label, alpha=0.7)
        axes[0, 1].plot(t, x[:, 1], color=color, label=label, alpha=0.7)
        axes[1, 0].plot(t, x[:, 2], color=color, label=label, alpha=0.7)
        axes[1, 1].plot(t, x[:, 3], color=color, label=label, alpha=0.7)
    
    axes[0, 0].set_ylabel('deltam (rad)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('omega (rad/s)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_ylabel('delta (rad)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_ylabel('v (voltage)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVoltage Collapse Model Test Complete")
    print(f"Initial condition: {x0}")
    print(f"Final state (Q1={Q1_test:.5f}): {x[-1]}")


