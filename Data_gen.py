#----------------------------------------------------------------------------------------------

#Imports

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Problem constants

N_CITIES = 10
P_TOTAL = 150e6       # 150 million
ALPHA = 0.01
BETA = 1.0
EPSILON = 0.01

#Defining Base amneities

base_amenities = np.array([
    100, 98, 95, 93, 92,
    90, 88, 87, 85, 83
])

#----------------------------------------------------------------------------------------------

#Utility function : Utility = Amenity - alpha*(Population)^beta

def compute_utility(A, P):
    return A - ALPHA * (P ** BETA)

# Creating target values (equilibrium populations considering utility which is affected by ameneties and population crowding)

def solve_equilibrium(A, max_iter=5000,lr=0.01):
  P= np.ones(N_CITIES) * (P_TOTAL/N_CITIES)

  for i in range(max_iter):
    U=compute_utility(A, P)
    U_mean = U.mean()

    #Gradient like adjustment (Projected Gradient ascent)
    P = P + lr *(U-U_mean)

    #Ensure positivity, i.e, no negative population.
    P=np.clip(P,1e5,None)

    #Rescales the population to ensure population conservation.
    P= P * (P_TOTAL/P.sum())

    if np.max(np.abs(U- U_mean)) <EPSILON:
      break

  return P



#---------------------------------------------------------------------------------------------
NUM_SAMPLES = 1200
amenities_list = []
populations_list = []

#Generating a dataset (1200 x 20) where 1200 x 10 is basic amenities that is preturbed by +=20%

for _ in tqdm(range(NUM_SAMPLES)):

    # Random ±20% amenity perturbation
    A = base_amenities * np.random.uniform(0.8, 1.2, size=N_CITIES)
    P_eq = solve_equilibrium(A)

    amenities_list.append(A)
    populations_list.append(P_eq)

amenities = np.array(amenities_list)
populations = np.array(populations_list)


#---------------------------------------------------------------------------------------------

#Naming the columns as (amenity_1,2,3....10 & pop_1,2,3.....10) for ceating DataFrame to further save it as a csv file

columns = (
    [f"amenity_{i+1}" for i in range(N_CITIES)] +
    [f"pop_{i+1}" for i in range(N_CITIES)]
)

# Converting to DataFrame
data = np.hstack([amenities, populations])
df = pd.DataFrame(data, columns=columns)
df.to_csv("urban_migration_equilibrium.csv", index=False)




