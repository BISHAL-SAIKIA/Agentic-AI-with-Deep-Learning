#----------------------------------------------------------------------------------------------

#Imports

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from dotenv import load_dotenv
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 1. Load the variables from .env
load_dotenv()

# Problem constants
N_CITIES = 10
P_TOTAL = 150e6       # 150 million
# ALPHA = 0.01
ALPHA = 100 / (150e6)
BETA = 1.0
EPSILON = 0.01

base_amenities = np.array([
    100, 98, 95, 93, 92,
    90, 88, 87, 85, 83
])

#----------------------------------------------------------------------------------------------

# Using Dataset class of Pytorch to: 
# 1. Normalize the values for efficient training.
# 2. Converting the Dataframe to tensor.
AMENITY_SCALE = 120.0
class MigrationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        self.A_raw = df.iloc[:, :N_CITIES].values.astype(np.float32)
        self.P_raw = df.iloc[:, N_CITIES:].values.astype(np.float32)

        # Normalize Amenities (Divide by max roughly 100 to get 0-1 range)
        self.A_max = 120.0 # slightly higher than max base amenity
        self.A_norm = self.A_raw / self.A_max

        # Normalize Population (Convert to fractions summing to 1.0)
        self.P_norm = self.P_raw / P_TOTAL

        # Convert to Tensor
        self.A_tensor = torch.tensor(self.A_norm, dtype=torch.float32)
        self.P_tensor = torch.tensor(self.P_norm, dtype=torch.float32)

    def __len__(self):
            return len(self.A_tensor)

    def __getitem__(self, idx):
            return self.A_tensor[idx], self.P_tensor[idx]

# Using Pytorch dataloader to split the data into batche of 32 for training.

dataset= MigrationDataset("urban_migration_equilibrium_usingScipy.csv")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


#----------------------------------------------------------------------------------------------

# Defining the model architecture.

class MigrationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(N_CITIES, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, N_CITIES)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        # Softmax ensures outputs sum to 1.0 (Population Fractions)
        return F.softmax(x, dim=1)
    
#----------------------------------------------------------------------------------------------

# Defining the physics informed losses.
# 1. Data Loss - Simple MSE Loss calculated by considering pedicted population and true population(target)
# 2. Equilibirum Loss- Which penalises the model if the 

class PILoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=10.0,lambda3=50):
        super().__init__()
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3
        self.mse = nn.MSELoss()

    def forward(self, A_norm, P_pred_fraction, P_true_fraction):

        # --- 1. Data Loss ---
        # Compare predicted fractions to true fractions
        data_loss = self.mse(P_pred_fraction, P_true_fraction)

        # --- Physics Recovery (Denormalization) ---
        # Recover Real World Units
        A_real = A_norm * AMENITY_SCALE
        P_real = P_pred_fraction * P_TOTAL

        # --- 2. Equilibrium Loss ---
        U = A_real - ALPHA * (P_real ** BETA)
        eq_loss = U.var(dim=1).mean()

        # --- 3. Conservation Loss ---
        cons_loss = ((P_pred_fraction.sum(dim=1) - 1.0) ** 2).mean()


        # --- Total ---
        total_loss = (self.l1 * data_loss) + (self.l2 * eq_loss) + (self.l3 * cons_loss)
        
        return total_loss, data_loss, eq_loss, cons_loss


model = MigrationNet().to(device)
criterion = PILoss(lambda1=1.0, lambda2=10.0, lambda3=50.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#WANDB Initialization
wandb.init(
    project="urban-migration-pinn",
    config={
        "lambda_data": 1.0,
        "lambda_eq": 10.0,
        "lambda_cons": 50.0,
        "lr": 0.001,
        "epochs": 120,
        "batch_size": 32
    }
)

#----------------------------------------------------------------------------------------------

#Training Loop

epochs=120
for epoch in range(epochs):
    model.train()
    
    # Accumulators (Reset every epoch)
    epoch_loss = 0
    epoch_data = 0
    epoch_eq = 0
    epoch_cons = 0
    
    for A, P_true in loader:
        A, P_true = A.to(device), P_true.to(device)

        # Forward
        P_pred = model(A)
        
        # Loss Calculation
        loss, dl, el, cl = criterion(A, P_pred, P_true)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        epoch_loss += loss.item()
        epoch_data += dl.item()
        epoch_eq   += el.item()
        epoch_cons += cl.item()

    # Calculate Averages
    avg_loss = epoch_loss / len(loader)
    avg_data = epoch_data / len(loader)
    avg_eq   = epoch_eq   / len(loader)
    avg_cons = epoch_cons / len(loader)

    # Log to WandB (Once per epoch)
    wandb.log({
        "epoch": epoch + 1,
        "total_loss": avg_loss,
        "data_loss": avg_data,
        "equilibrium_loss": avg_eq,
        "conservation_loss": avg_cons
    })
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Data: {avg_data:.6f} | Eq: {avg_eq:.6f} | Cons: {avg_cons:.2e}")


# Finish the run 
wandb.finish()
print("Training complete")



#----------------------------------------------------------------------------------------------

# Testing model on randomly generated amernities and checking the mean variance of the utilities to check how well the model is working.
def predict_equilibrium(amenities):
    # 1. Normalize Input (Same as training: divide by 120)
    amenities = np.array(amenities, dtype=np.float32)
    A_norm = amenities / AMENITY_SCALE
    
    # Convert to tensor [1, 10]
    A_tensor = torch.tensor(A_norm, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # Model outputs fractions (Softmax)
        P_fraction = model(A_tensor)

    # 2. Denormalize Output (Fraction * Total Population)
    P_real = P_fraction.cpu().numpy().squeeze() * P_TOTAL
    
    return P_real


# --- Running Single Test ---
print("\n--- Single Prediction Test ---")
test_amenities = base_amenities * np.random.uniform(0.90, 1.1, size=10)
predicted_pops = predict_equilibrium(test_amenities)

print("Predicted populations (millions):")

print("Predicted populations (millions):")
for i, pop in enumerate(predicted_pops):
    print(f"City {i+1:2d} : {pop/1e6 :.6f} M")

print(f"Total population: {predicted_pops.sum()/1e6:.6f} M (Target: 150.0)")


# --- Validation Function ---
def check_equilibrium_variance(amenities, populations):
    # Calculate Utility: U = A - alpha * P^beta
    utilities = amenities - ALPHA * (populations ** BETA)
    
    # Variance should be close to 0
    var = np.var(utilities)
    print(f"Utility Variance: {var:.6f}")
    return var


# Generating 5 random ameneties vector and testing
variances = []

for i in range(5):
    # 1. Generate new random amenities
    current_amenities = base_amenities * np.random.uniform(0.95, 1.1, size=10)
    
    # 2. Predict population for THIS specific scenario
    current_pops = predict_equilibrium(current_amenities)
    
    # 3. Check if Utilities are equal (Variance ~ 0)
    print(f"Test {i+1}: ", end="")
    var = check_equilibrium_variance(current_amenities, current_pops)
    variances.append(var)

print(f"\nMean Variance over 5 runs: {np.mean(variances):.6f}")

model_path = "migration_model.pt"

torch.save(model.state_dict(),model_path)
print(f"Model saved to {model_path}")