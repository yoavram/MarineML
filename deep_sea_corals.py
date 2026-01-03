import pandas as pd
import numpy as np

# @title 🌊 Generate "Deep Sea Coral Survey" Dataset
# Based on real niche profiles of common Red Sea & Deep Sea corals.

np.random.seed(42)
n_samples = 1500

# --- 1. Define Species Niches (Real Biology) ---
# Species A: Stylophora pistillata (Shallow, Warm, High Light)
# Species B: Desmophyllum pertusum (Deep, Cold, Low Light - formerly Lophelia)
# Species C: Dendrophyllia (Mesophotic, Medium Depth)

species = []
depths = []
temps = []
salinities = []
oxygens = []

for i in range(n_samples):
    # Pick a "True" species for this site
    sp = np.random.choice(['Stylophora_pistillata', 'Desmophyllum_pertusum', 'Dendrophyllia_sp'])
    
    if sp == 'Stylophora_pistillata':
        # Shallow Reef: 5-50m, 24-28C
        d = np.random.normal(20, 10)
        t = np.random.normal(26, 1.5)
        o2 = np.random.normal(5.5, 0.5) # High oxygen (waves)
    
    elif sp == 'Dendrophyllia_sp':
        # Mesophotic/Twilight Zone: 50-150m, 20-24C
        d = np.random.normal(100, 25)
        t = np.random.normal(22, 1.2)
        o2 = np.random.normal(4.5, 0.4)
        
    elif sp == 'Desmophyllum_pertusum':
        # Deep Sea: 200m+, 12-16C (Red Sea is warm deep, unlike Atlantic!)
        d = np.random.normal(400, 50)
        t = np.random.normal(21, 0.5) # Red Sea deep water is weirdly warm (~21C)
        o2 = np.random.normal(2.0, 0.3) # Lower oxygen

    # Add environmental noise (Real sensors are noisy)
    d = abs(d + np.random.normal(0, 5))
    t = t + np.random.normal(0, 0.5)
    s = np.random.normal(40.5, 0.2) # Red Sea is salty!
    o2 = abs(o2 + np.random.normal(0, 0.2))

    species.append(sp)
    depths.append(round(d, 1))
    temps.append(round(t, 2))
    salinities.append(round(s, 3))
    oxygens.append(round(o2, 2))

df_coral = pd.DataFrame({
    'Species_ID': species,
    'Depth_m': depths,
    'Temperature_C': temps,
    'Salinity_PPS': salinities,
    'Dissolved_Oxygen_mlL': oxygens
})

# Save
df_coral.to_csv('deep_sea_corals.csv', index=False)
print("✅ Created 'deep_sea_corals.csv' with 1500 survey points.")