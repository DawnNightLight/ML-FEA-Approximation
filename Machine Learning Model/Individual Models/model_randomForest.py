import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("../../../Misc/Run Results/FinrayTestRun.csv")

print("\nNumber of simulations:", len(data))

# ============================================================
# 2. DEFINE INPUTS AND OUTPUT
# ============================================================


input_columns = [
    "baseSize",
    "externalBeamWidth",
    "internalBeamSpacing",
    "internalBeamSize",
    "load"
]

output_column = "max_Displacement"

X = data[input_columns]
y = data[output_column]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training simulations:", len(X_train))
print("Testing simulations:", len(X_test))


# ============================================================
# 4. CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 6. TEST
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 7. CALCULATE PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n============================")
print("RANDOM FOREST PERFORMANCE")
print("============================")

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")


# ============================================================
# 8. DISPLAY EXAMPLE PREDICTIONS
# ============================================================

print("\nExample predictions:")

for actual, predicted in zip(
    y_test.values[:10],
    predictions[:10]
):

    print(
        f"FEA: {actual:.4f} "
        f"| Random Forest: {predicted:.4f}"
    )


# ============================================================
# 9. PLOT FEA VS RANDOM FOREST
# ============================================================

plt.scatter(
    y_test,
    predictions
)

minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("FEA Tip Displacement")
plt.ylabel("Random Forest Prediction")

plt.title("FEA vs Random Forest")

plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

print("\nFeature importance:")

for feature, value in zip(
    input_columns,
    importance
):

    print(
        f"{feature}: {value:.4f}"
    )