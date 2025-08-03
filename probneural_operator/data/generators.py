"""Synthetic data generators for PDE datasets."""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class SyntheticPDEGenerator:
    """Generator for synthetic PDE solutions.
    
    This class provides methods to generate synthetic solutions for various
    PDEs commonly used in neural operator benchmarks.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
    
    def generate_navier_stokes(self,
                             n_samples: int,
                             resolution: int = 64,
                             time_steps: int = 50,
                             viscosity: float = 1e-3,
                             dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2D Navier-Stokes solutions.
        
        Args:
            n_samples: Number of samples to generate
            resolution: Spatial resolution (NxN grid)
            time_steps: Number of time steps
            viscosity: Fluid viscosity
            dt: Time step size
            
        Returns:
            Tuple of (initial_conditions, solutions)
        """
        # Create spatial grid
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        xx, yy = np.meshgrid(x, y)
        
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            # Generate random initial vorticity field
            initial_vorticity = self._generate_random_field_2d(resolution, length_scale=0.1)
            
            # Solve Navier-Stokes (simplified using vorticity-streamfunction formulation)
            solution = self._solve_navier_stokes_2d(
                initial_vorticity, 
                viscosity=viscosity,
                time_steps=time_steps,
                dt=dt
            )
            
            inputs.append(initial_vorticity)
            outputs.append(solution)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} Navier-Stokes samples")
        
        return np.array(inputs), np.array(outputs)
    
    def generate_burgers(self,
                        n_samples: int,
                        resolution: int = 256,
                        time_steps: int = 100,
                        viscosity: float = 0.01,
                        dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 1D Burgers equation solutions.
        
        Args:
            n_samples: Number of samples
            resolution: Spatial resolution
            time_steps: Number of time steps
            viscosity: Viscosity parameter
            dt: Time step size
            
        Returns:
            Tuple of (initial_conditions, solutions)
        """
        x = np.linspace(0, 1, resolution)
        
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            # Generate random initial condition
            initial_condition = self._generate_random_field_1d(resolution, length_scale=0.1)
            
            # Solve Burgers equation
            solution = self._solve_burgers_1d(
                initial_condition,
                viscosity=viscosity,
                time_steps=time_steps,
                dt=dt
            )
            
            inputs.append(initial_condition)
            outputs.append(solution)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} Burgers samples")
        
        return np.array(inputs), np.array(outputs)
    
    def generate_darcy_flow(self,
                           n_samples: int,
                           resolution: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Darcy flow solutions.
        
        Args:
            n_samples: Number of samples
            resolution: Spatial resolution
            
        Returns:
            Tuple of (permeability_fields, pressure_solutions)
        """
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            # Generate random permeability field (log-normal)
            log_perm = self._generate_random_field_2d(resolution, length_scale=0.2)
            permeability = np.exp(log_perm)
            
            # Solve Darcy flow equation
            pressure = self._solve_darcy_flow_2d(permeability, resolution)
            
            inputs.append(permeability)
            outputs.append(pressure)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} Darcy flow samples")
        
        return np.array(inputs), np.array(outputs)
    
    def generate_heat_equation(self,
                              n_samples: int,
                              resolution: int = 64,
                              time_steps: int = 50,
                              diffusivity: float = 0.01,
                              dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate heat equation solutions.
        
        Args:
            n_samples: Number of samples
            resolution: Spatial resolution
            time_steps: Number of time steps
            diffusivity: Thermal diffusivity
            dt: Time step size
            
        Returns:
            Tuple of (initial_conditions, solutions)
        """
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            # Generate random initial temperature field
            initial_temp = self._generate_random_field_2d(resolution, length_scale=0.15)
            
            # Solve heat equation
            solution = self._solve_heat_equation_2d(
                initial_temp,
                diffusivity=diffusivity,
                time_steps=time_steps,
                dt=dt
            )
            
            inputs.append(initial_temp)
            outputs.append(solution)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} heat equation samples")
        
        return np.array(inputs), np.array(outputs)
    
    def _generate_random_field_1d(self, resolution: int, length_scale: float = 0.1) -> np.ndarray:
        """Generate 1D Gaussian random field."""
        x = np.linspace(0, 1, resolution)
        
        # Generate smooth random field using Fourier modes
        n_modes = min(20, resolution // 4)
        field = np.zeros(resolution)
        
        for k in range(1, n_modes + 1):
            amp = np.random.normal(0, 1) * np.exp(-k * length_scale)
            phase = np.random.uniform(0, 2 * np.pi)
            field += amp * np.sin(2 * np.pi * k * x + phase)
        
        return field
    
    def _generate_random_field_2d(self, resolution: int, length_scale: float = 0.1) -> np.ndarray:
        """Generate 2D Gaussian random field."""
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Generate smooth random field using Fourier modes
        n_modes = min(10, resolution // 8)
        field = np.zeros((resolution, resolution))
        
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                k_norm = np.sqrt(kx**2 + ky**2)
                amp = np.random.normal(0, 1) * np.exp(-k_norm * length_scale)
                phase = np.random.uniform(0, 2 * np.pi)
                field += amp * np.sin(2 * np.pi * kx * xx + 2 * np.pi * ky * yy + phase)
        
        return field
    
    def _solve_burgers_1d(self,
                         initial_condition: np.ndarray,
                         viscosity: float,
                         time_steps: int,
                         dt: float) -> np.ndarray:
        """Solve 1D Burgers equation using finite differences."""
        resolution = len(initial_condition)
        dx = 1.0 / (resolution - 1)
        
        # Initialize solution array
        solution = np.zeros((time_steps, resolution))
        u = initial_condition.copy()
        solution[0] = u
        
        for t in range(1, time_steps):
            u_new = u.copy()
            
            # Interior points (upwind scheme for advection + central for diffusion)
            for i in range(1, resolution - 1):
                # Advection term (upwind)
                if u[i] >= 0:
                    advection = u[i] * (u[i] - u[i-1]) / dx
                else:
                    advection = u[i] * (u[i+1] - u[i]) / dx
                
                # Diffusion term (central difference)
                diffusion = viscosity * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
                
                u_new[i] = u[i] - dt * advection + dt * diffusion
            
            # Boundary conditions (periodic)
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            
            u = u_new
            solution[t] = u
        
        return solution
    
    def _solve_navier_stokes_2d(self,
                               initial_vorticity: np.ndarray,
                               viscosity: float,
                               time_steps: int,
                               dt: float) -> np.ndarray:
        """Solve 2D Navier-Stokes using simplified vorticity-streamfunction method."""
        resolution = initial_vorticity.shape[0]
        dx = 1.0 / (resolution - 1)
        
        # Initialize arrays
        solution = np.zeros((time_steps, resolution, resolution))
        vorticity = initial_vorticity.copy()
        solution[0] = vorticity
        
        # Simple forward Euler evolution (simplified)
        for t in range(1, time_steps):
            # Diffusion term (simplified Laplacian)
            vort_new = vorticity.copy()
            
            for i in range(1, resolution - 1):
                for j in range(1, resolution - 1):
                    laplacian = (vorticity[i+1, j] + vorticity[i-1, j] + 
                               vorticity[i, j+1] + vorticity[i, j-1] - 4*vorticity[i, j]) / (dx**2)
                    vort_new[i, j] = vorticity[i, j] + dt * viscosity * laplacian
            
            # Apply boundary conditions (no-slip)
            vort_new[0, :] = 0
            vort_new[-1, :] = 0
            vort_new[:, 0] = 0
            vort_new[:, -1] = 0
            
            vorticity = vort_new
            solution[t] = vorticity
        
        return solution
    
    def _solve_darcy_flow_2d(self, permeability: np.ndarray, resolution: int) -> np.ndarray:
        """Solve 2D Darcy flow using finite differences."""
        dx = 1.0 / (resolution - 1)
        
        # Create finite difference matrix
        n = resolution * resolution
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Generate random source term
        source = self._generate_random_field_2d(resolution, length_scale=0.3)
        
        for i in range(resolution):
            for j in range(resolution):
                idx = i * resolution + j
                
                if i == 0 or i == resolution-1 or j == 0 or j == resolution-1:
                    # Boundary conditions (Dirichlet: p = 0)
                    A[idx, idx] = 1.0
                    b[idx] = 0.0
                else:
                    # Interior points: -∇·(κ∇p) = f
                    k_center = permeability[i, j]
                    k_left = permeability[i, j-1] if j > 0 else k_center
                    k_right = permeability[i, j+1] if j < resolution-1 else k_center
                    k_up = permeability[i-1, j] if i > 0 else k_center
                    k_down = permeability[i+1, j] if i < resolution-1 else k_center
                    
                    # Coefficient for center point
                    A[idx, idx] = -(k_left + k_right + k_up + k_down) / (dx**2)
                    
                    # Coefficients for neighbors
                    if j > 0:
                        A[idx, idx-1] = k_left / (dx**2)
                    if j < resolution-1:
                        A[idx, idx+1] = k_right / (dx**2)
                    if i > 0:
                        A[idx, idx-resolution] = k_up / (dx**2)
                    if i < resolution-1:
                        A[idx, idx+resolution] = k_down / (dx**2)
                    
                    b[idx] = source[i, j]
        
        # Solve linear system
        pressure_flat = np.linalg.solve(A, b)
        pressure = pressure_flat.reshape((resolution, resolution))
        
        return pressure
    
    def _solve_heat_equation_2d(self,
                               initial_temp: np.ndarray,
                               diffusivity: float,
                               time_steps: int,
                               dt: float) -> np.ndarray:
        """Solve 2D heat equation using explicit finite differences."""
        resolution = initial_temp.shape[0]
        dx = 1.0 / (resolution - 1)
        
        # Initialize solution array
        solution = np.zeros((time_steps, resolution, resolution))
        temp = initial_temp.copy()
        solution[0] = temp
        
        # Stability condition for explicit scheme
        alpha = diffusivity * dt / (dx**2)
        if alpha > 0.25:
            print(f"Warning: alpha = {alpha} > 0.25, solution may be unstable")
        
        for t in range(1, time_steps):
            temp_new = temp.copy()
            
            # Interior points
            for i in range(1, resolution - 1):
                for j in range(1, resolution - 1):
                    laplacian = (temp[i+1, j] + temp[i-1, j] + 
                               temp[i, j+1] + temp[i, j-1] - 4*temp[i, j]) / (dx**2)
                    temp_new[i, j] = temp[i, j] + dt * diffusivity * laplacian
            
            # Boundary conditions (Dirichlet: T = 0)
            temp_new[0, :] = 0
            temp_new[-1, :] = 0
            temp_new[:, 0] = 0
            temp_new[:, -1] = 0
            
            temp = temp_new
            solution[t] = temp
        
        return solution


class RandomFieldGenerator:
    """Generator for random fields used as initial conditions or coefficients."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def gaussian_random_field(self,
                             size: Tuple[int, ...],
                             length_scale: float = 0.1,
                             variance: float = 1.0) -> np.ndarray:
        """Generate Gaussian random field.
        
        Args:
            size: Shape of the field
            length_scale: Correlation length scale
            variance: Field variance
            
        Returns:
            Random field array
        """
        if len(size) == 1:
            return self._grf_1d(size[0], length_scale, variance)
        elif len(size) == 2:
            return self._grf_2d(size, length_scale, variance)
        else:
            raise ValueError("Only 1D and 2D fields supported")
    
    def _grf_1d(self, n: int, length_scale: float, variance: float) -> np.ndarray:
        """Generate 1D Gaussian random field."""
        x = np.linspace(0, 1, n)
        field = np.zeros(n)
        
        n_modes = min(50, n // 4)
        for k in range(1, n_modes + 1):
            amp = np.random.normal(0, 1) * np.exp(-k * length_scale) * np.sqrt(variance)
            phase = np.random.uniform(0, 2 * np.pi)
            field += amp * np.sin(2 * np.pi * k * x + phase)
        
        return field
    
    def _grf_2d(self, size: Tuple[int, int], length_scale: float, variance: float) -> np.ndarray:
        """Generate 2D Gaussian random field."""
        nx, ny = size
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        
        field = np.zeros((nx, ny))
        n_modes = min(20, min(nx, ny) // 8)
        
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                k_norm = np.sqrt(kx**2 + ky**2)
                amp = np.random.normal(0, 1) * np.exp(-k_norm * length_scale) * np.sqrt(variance)
                phase = np.random.uniform(0, 2 * np.pi)
                field += amp * np.sin(2 * np.pi * kx * xx + 2 * np.pi * ky * yy + phase)
        
        return field