import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=== Machine Learning Predictive Analytics Project ===")

# ==============================================================================
# Step 1: डेटा तैयार करना | Data Preparation & Synthetic Dataset Creation
# ==============================================================================
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame(
    {
        # Tenure_Months: ग्राहक कितने महीने से जुड़ा है | Customer tenure in months
        "Tenure_Months": np.random.randint(1, 72, n_samples),
        # Monthly_Charges: मासिक शुल्क | Monthly subscription charges
        "Monthly_Charges": np.random.uniform(20.0, 120.0, n_samples),
        # Total_Charges: कुल जमा राशि | Cumulative total charges paid
        "Total_Charges": np.random.uniform(100.0, 8000.0, n_samples),
        # Support_Calls: कस्टमर केयर कॉल की संख्या | Number of customer support calls
        "Support_Calls": np.random.randint(0, 10, n_samples),
        # Contract_Type: अनुबंध का प्रकार (0: Monthly, 1: 1-Year, 2: 2-Year) | Contract duration type
        "Contract_Type": np.random.choice([0, 1, 2], n_samples),
        # Churn: ग्राहक सेवा छोड़ रहा है या नहीं (Target Variable) | Customer churn status (0: Retained, 1: Churned)
        "Churn": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    }
)

print(f"Dataset Shape (डेटा का आकार): {data.shape}")

# ==============================================================================
# Step 2: फीचर्स और टारगेट को अलग करना | Feature & Target Separation
# ==============================================================================
# X: स्वतंत्र विशेषताएं (Input Features)
X = data.drop("Churn", axis=1)
# y: लक्षित मान (Target Label to predict)
y = data["Churn"]

# ==============================================================================
# Step 3: डेटा को ट्रेन और टेस्ट में विभाजित करना | Train-Test Split (80% Train, 20% Test)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# Step 4: डेटा स्केलिंग (मानकीकरण) | Feature Scaling using StandardScaler
# ==============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# Step 5: मॉडल ट्रेनिंग | Model Training using Random Forest Classifier
# ==============================================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# Step 6: मॉडल का मूल्यांकन और परिणाम | Model Evaluation & Performance Output
# ==============================================================================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("-" * 50)
print(f"Model Training Completed Successfully! (मॉडल ट्रेनिंग पूरी हुई)")
print(f"Test Accuracy Score (सटीकता स्कोर): {acc * 100:.2f}%")
print("-" * 50)
print("Classification Report (वर्गीकरण रिपोर्ट):\n", classification_report(y_test, y_pred))
