import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())
print("\nChurn distribution:\n", df["churn"].value_counts())

# Convert target to 0/1
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

# Remove identifier
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

numeric_features = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]
categorical_features = ["gender", "contract", "payment_method", "internet_service", "tech_support"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred), 3))
print("\nClassification Report:\n", classification_report(y_test, pred, zero_division=0))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred),
    display_labels=["No Churn", "Churn"]
).plot()
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Example prediction
new_customer = pd.DataFrame([{
    "gender": "Female",
    "senior_citizen": 0,
    "tenure": 3,
    "monthly_charges": 85.0,
    "total_charges": 255.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "tech_support": "No"
}])

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

print("\nExample customer prediction:")
print("Prediction:", "Likely to Churn" if prediction == 1 else "Likely to Stay")
print("Churn probability:", round(probability * 100, 2), "%")
