import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# Constants
N_CITIES = 10
P_TOTAL = 150
ALPHA = 0.01
BETA = 1.0

base_amenities = np.array([100, 98, 95, 93, 92, 90, 88, 87, 85, 83])

def solve_equilibrium_scipy(A):
    """
    Solves for equilibrium using Scipy Constrained Optimization.
    We MINIMIZE the negative potential energy.
    """
    
    # 1. The Objective Function (Negative Potential Energy)
    def objective(P):
        # Integral of U = A - alpha*P^beta
        # Int(U) = A*P - (alpha / beta+1) * P^(beta+1)
        term1 = np.dot(A, P)
        term2 = (ALPHA / (BETA + 1)) * np.sum(P**(BETA + 1))
        return -(term1 - term2)  # Negate because we want to MAXIMIZE utility

    # 2. Constraints: Sum of P must equal P_TOTAL
    constraints = ({
        'type': 'eq', 
        'fun': lambda P: np.sum(P) - P_TOTAL
    })

    # 3. Bounds: Population cannot be negative (0 to P_TOTAL)
    bounds = [(0, P_TOTAL) for _ in range(N_CITIES)]

    # 4. Initial Guess (Equal distribution)
    P0 = np.ones(N_CITIES) * (P_TOTAL / N_CITIES)

    # 5. Run the Solver (SLSQP is excellent for constrained problems)
    result = minimize(
        objective, 
        P0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        tol=1e-6
    )

    return result.x

# --- Data Generation Loop ---
NUM_SAMPLES = 1200
amenities_list = []
populations_list = []

print("Generating data with Scipy...")
for _ in tqdm(range(NUM_SAMPLES)):
    # Random ±20% amenity perturbation
    A = base_amenities * np.random.uniform(0.8, 1.2, size=N_CITIES)
    
    # Solve exactly
    P_eq = solve_equilibrium_scipy(A)
    
    amenities_list.append(A)
    populations_list.append(P_eq)

# Save to CSV (Same as before)
amenities = np.array(amenities_list)
populations = np.array(populations_list)

columns = [f"amenity_{i+1}" for i in range(N_CITIES)] + [f"pop_{i+1}" for i in range(N_CITIES)]
df = pd.DataFrame(np.hstack([amenities, populations]), columns=columns)
df.to_csv("urban_migration_equilibrium_usingScipy2.csv", index=False)
print("Done.")