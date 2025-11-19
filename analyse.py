import pandas as pd
import numpy as np

# Constants (taken from your notebook)
R = 0.04        # wheel radius (m)
L = 0.125       # distance from wheel to robot center (m)
alpha = np.deg2rad(0)     # robot orientation angle (rad) – change if needed
theta = np.deg2rad(30)    # wheel angle (rad)

# Velocity transformation matrix
M_vel = np.array([
    [(-2/3)*np.cos(alpha - theta), (2/3)*np.sin(alpha),       (2/3)*np.cos(alpha + theta)],
    [(-2/3)*np.sin(alpha - theta), (-2/3)*np.cos(alpha),      (2/3)*np.sin(alpha + theta)],
    [1/(3*L),                   1/(3*L),                1/(3*L)]
])

# Current projection matrix (same structure as notebook)
M_current = np.array([
    [(-2/3)*np.cos(alpha - theta), (2/3)*np.sin(alpha),       (2/3)*np.cos(alpha + theta)],
    [(-2/3)*np.sin(alpha - theta), (-2/3)*np.cos(alpha),      (2/3)*np.sin(alpha + theta)],
    [1/3,                          1/3,                       1/3]
])

def compute_velocities(row):
    omega = np.array([row['V1real'], row['V2real'], row['V3real']])
    res = R * (M_vel @ omega)
    return pd.Series({'Vx': res[0], 'Vy': res[1], 'Omega': res[2]})

def compute_projected_currents(row):
    currents = np.array([row['I1'], row['I2'], row['I3']])
    res = M_current.dot(currents)
    return pd.Series({'Ix': res[0], 'Iy': res[1], 'Iphi': res[2]})

def compute_torques(df):
    epsilon = 1e-6  # to avoid division by zero
    df['Tx'] = df['Vx'] / (df['Ix'] + epsilon)
    df['Ty'] = df['Vy'] / (df['Iy'] + epsilon)
    df['Tphi'] = df['Omega'] / (df['Iphi'] + epsilon)
    df['Tz'] = df['gz'] / (df['Iphi'] + epsilon)

def main():
    # Load C dataset
    df = pd.read_excel("data/raw/Data_Set_C.xlsx")

    # Compute velocities
    df[['Vx', 'Vy', 'Omega']] = df.apply(compute_velocities, axis=1)

    # Compute projected currents
    df[['Ix', 'Iy', 'Iphi']] = df.apply(compute_projected_currents, axis=1)

    # Compute torques
    compute_torques(df)

    # Save
    df.to_excel("data/raw/processed_C_data.xlsx", index=False, engine="openpyxl")
    print("Saved: C_dataset_processed.xlsx")

if __name__ == "__main__":
    main()
