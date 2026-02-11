# flower_classification.py

# 📚 Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1️⃣ Load Dataset
df = pd.read_excel("data/flower_ml_strong_pattern.xlsx")
print("Dataset Loaded:")
print(df.head())

# 2️⃣ Separate Features and Target
X = df[["Length (cm)", "Width (cm)", "Size (cm)"]]
y = df["Species"]

# 3️⃣ Optional: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 5️⃣ Train Logistic Regression
# Using solver lbfgs (multinomial automatically)
model = LogisticRegression(max_iter=200, solver="lbfgs")
model.fit(x_train, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# 7️⃣ Predict New Flower
new_flower = [[7.0, 3.2, 7.0*3.2]]  # Length, Width, Size
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print("Predicted Species for new flower:", prediction[0])

# 8️⃣ Save Predictions to CSV
df_test = pd.DataFrame(x_test, columns=["Length", "Width", "Size"])
df_test["Actual"] = y_test.values
df_test["Predicted"] = y_pred
df_test.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
