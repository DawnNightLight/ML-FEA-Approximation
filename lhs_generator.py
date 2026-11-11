import pandas as pd
import numpy as np
from scipy.stats import qmc

# SETTINGS

N = 300
SEED = 42
BATCH_SIZE = 1000

# Parameter: (minimum, maximum)
parameters = {
    "load": (0, 50),
    "baseSize": (150, 350),
    "externalBeamWidth": (30, 100),
    "internalBeamSpacing": (20, 35),
    "internalBeamSize": (20, 35)
}

# CONSTRAINTS

def is_valid(row):

    # Example constraint:
    # If thickness < 250, length must be > 50
    if row["baseSize"] < 250 and row["externalBeamWidth"] >= 50:
        print("Rejected, poor external beam width")
        return False

    # Example constraint:
    # 5 * (A + B) must be greater than 500
    if 12 * (row["internalBeamSpacing"] + row["internalBeamSize"]) <= 600:
        print("Rejected, poor internal beam height:", 12 * (row["internalBeamSpacing"] + row["internalBeamSize"]))
        return False

    # Add further constraints here:
    #
    # if row["force"] > row["length"]:
    #     return False
    #
    # if row["parameter_A"] + row["parameter_B"] > 10:
    #     return False

    print("Succesful geneartion!")
    return True



def generate_candidates(n):

    sampler = qmc.LatinHypercube(
        d=len(parameters),
        seed=None
    )

    sample = sampler.random(n=n)

    parameter_names = list(parameters.keys())

    data = {}

    for i, name in enumerate(parameter_names):

        minimum, maximum = parameters[name]

        data[name] = np.floor(
            minimum
            + sample[:, i]
            * (maximum - minimum + 1)
        ).astype(int)

        data[name] = np.clip(
            data[name],
            minimum,
            maximum
        )

    return pd.DataFrame(data)

rng = np.random.default_rng(SEED)

valid_rows = []

attempts = 0

while len(valid_rows) < N:

    attempts += 1

    # Generate a batch
    candidates = generate_candidates(BATCH_SIZE)

    # Check constraints
    for _, row in candidates.iterrows():

        if is_valid(row):

            valid_rows.append(row)

            if len(valid_rows) >= N:
                break

df = pd.DataFrame(valid_rows)

df = df.reset_index(drop=True)

df.to_csv(
    "lhs_simulations.csv",
    index=False
)

print("Generated simulations:", len(df))
print("LHS candidates generated:", attempts * BATCH_SIZE)

print("\nFirst 10 simulations:")
print(df.head(10))

print("\nParameter ranges:")
print(df.describe().loc[["min", "max"]])

print("\nSaved to:")
print("lhs_simulations.csv")
