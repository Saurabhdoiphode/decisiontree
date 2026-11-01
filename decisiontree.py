import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
df = pd.read_csv("customer-churn.csv")

print(df.head())
print("Dataset Shape:", df.shape)

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df = df.dropna()

# Drop unnecessary column
df = df.drop("customerID", axis=1)

# Convert categorical data into numerical data
df = pd.get_dummies(df, drop_first=True)

print("Dataset Shape after preprocessing:", df.shape)

# Separate input and output
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(criterion="gini", random_state=42)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("\nF1 Score:", f1)

# Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree - Customer Churn Prediction")
plt.show()

# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()