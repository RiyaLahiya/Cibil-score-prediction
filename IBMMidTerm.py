import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("credit_risk_dataset.csv")

import pandas as pd

df = pd.read_csv("credit_risk_dataset.csv")

df.head()

X = df.drop(columns=["cb_person_default_on_file"])
y = df["cb_person_default_on_file"]

X = pd.get_dummies(X, drop_first=True)

print(df.isnull().sum())

df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)


X = df.drop(columns=["cb_person_default_on_file"])
y = df["cb_person_default_on_file"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Before splitting into train and test
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Then split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Now train
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, y_pred))

importances = rf_model.feature_importances_

feature_names = X.columns  # Save column names before transformation
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.show()

import joblib



joblib.dump(rf_model, 'random_forest_credit_risk_model.pkl')

# Save the feature column names
joblib.dump(X.columns.tolist(), "model_features.pkl")


svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)


imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

y_pred = svm_model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\n--- K-Nearest Neighbors (k=5) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))


