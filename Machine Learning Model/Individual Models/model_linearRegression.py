import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# 4. CREATE MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining Linear Regression...")

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
print("LINEAR REGRESSION PERFORMANCE")
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
        f"| Linear Regression: {predicted:.4f}"
    )


# ============================================================
# 9. PLOT FEA VS LINEAR REGRESSION
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
plt.ylabel("Linear Regression Prediction")

plt.title("FEA vs Linear Regression")

plt.show()


# ============================================================
# 10. DISPLAY MODEL COEFFICIENTS
# ============================================================

print("\nLinear regression coefficients:")

for feature, coefficient in zip(
    input_columns,
    model.coef_
):

    print(
        f"{feature}: {coefficient:.6f}"
    )

print(
    f"\nIntercept: {model.intercept_:.6f}"
)