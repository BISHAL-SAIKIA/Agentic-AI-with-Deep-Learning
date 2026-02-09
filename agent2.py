from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_config import CITY_TO_IDX, CITY_NAMES, BASE_AMENITIES
N_CITIES = 10

#Model Architecture
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

# ------------------------------------------------------------------
# loading API Keys
load_dotenv()

# ------------------------------------------------------------------
# LLM Parser
parser_prompt = PromptTemplate.from_template("""
You are a city migration analysis assistant.

Extract structured information from the query.

Return JSON with:
- affected_cities: list of city names
- amenity_changes: dict {{city: percentage_change}}
- scenario: short description

Query:
{query}

Return ONLY valid JSON.
""")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser_chain = parser_prompt | llm

# ------------------------------------------------------------------
# Load trained neural model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MigrationNet().to(device)
model.load_state_dict(
    torch.load("migration_model.pt", map_location=device)
)
model.eval()

# Constants (MUST match training)
AMENITY_SCALE = 120.0
P_TOTAL = 150e6

# ------------------------------------------------------------------
def apply_amenity_change(parsed):
    amenities = np.array(BASE_AMENITIES, dtype=np.float32)

    for city, percent in parsed["amenity_changes"].items():
        city_key = city.lower()
        if city_key not in CITY_TO_IDX:
            raise ValueError(f"Unknown city: {city}")

        idx = CITY_TO_IDX[city_key]
        amenities[idx] *= (1 + percent / 100.0)

    return amenities

# ------------------------------------------------------------------
def run_equilibrium(amenities):
    A_norm = torch.tensor(
        amenities / AMENITY_SCALE,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        P_fraction = model(A_norm)

    return P_fraction.cpu().numpy().squeeze() * P_TOTAL

# ------------------------------------------------------------------
def format_response(populations):
    total = populations.sum()
    lines = []

    for i, pop in enumerate(populations):
        lines.append(
            f"{CITY_NAMES[i]}: {pop/1e6:.2f} M "
            f"({(pop/total)*100:.1f}%)"
        )

    return "\n".join(lines)

# ------------------------------------------------------------------
def migration_agent(query,verbose=True):
    if verbose :
        print("\nStep 1.User query:\n")
        print(query)
        print("\n----------------------------------------------------------\n")

    # Parse query
    llm_response = parser_chain.invoke({"query": query})
    parsed = json.loads(llm_response.content)

    if verbose:
        print("\nStep 2.Parsed output (LLM --> JSON)\n")
        print(json.dumps(parsed, indent=2))
        print("\n----------------------------------------------------------\n")

    # Modify amenities
    amenities = apply_amenity_change(parsed)

    # Neural equilibrium inference
    populations = run_equilibrium(amenities)

    if verbose:
        print("\nStep 3.Migration Result using the trained model\n")
    # Generate response
    return format_response(populations)

# ------------------------------------------------------------------
if __name__ == "__main__":
    query = "If Beijing loses 10% amenity due to pollution, where do people migrate?"
    print(migration_agent(query))
