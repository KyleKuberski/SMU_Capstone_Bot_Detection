# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.sparse import vstack, hstack
from collections import Counter
from datetime import datetime
import pytz
import scipy  # Import scipy here

# Step 2: Load and Clean Data
df = pd.read_csv("C:/Users/kuber/Desktop/Capstone/test train set/train.csv")

# Drop unnecessary columns if they exist
columns_to_drop = ['user__entities__description__urls__url', 'user__entities__|__urls__url',
                   'user__entities__url__urls__indices__001', 'user__profile_sidebar_border_color',
                   'user__entities__|', 'user__entities__|__urls__indices__001']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

# Fill missing values and drop duplicates
df.fillna({"Location": "Unknown", "Bio": "Unknown"}, inplace=True)
df.drop_duplicates(inplace=True)

# Ensure date columns are datetime format
df['AccountCreateDate'] = pd.to_datetime(df['AccountCreateDate'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')

# Step 3: Feature Engineering
today = datetime.now(pytz.utc)
df['AccountCreateYear'] = df['AccountCreateDate'].dt.year
df['AccountCreateMonth'] = df['AccountCreateDate'].dt.month
df['AccountCreateDayOfWeek'] = df['AccountCreateDate'].dt.dayofweek
df['AccountCreateHour'] = df['AccountCreateDate'].dt.hour
df['AccountAgeDays'] = (today - df['AccountCreateDate']).dt.days
df['NameLength'] = df['Name'].str.len()
df['HandleContainsNumber'] = df['Handle'].apply(lambda x: int(any(char.isdigit() for char in str(x))))

# Encode categorical features
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'].astype(str))

# Step 4: Text Feature Extraction
text_data = df['TweetText'].astype(str) + " " + df['Bio'].astype(str)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
text_features = tfidf_vectorizer.fit_transform(text_data)

# Step 5: Impute and Scale Numerical Features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col != 'TweetText']  # Exclude non-numeric columns

numerical_imputer = SimpleImputer(strategy='median')

# Impute missing values on each numerical column individually
for col in numerical_cols:
    if df[col].isnull().sum() > 0:  # Check if there are missing values in the column
        df[[col]] = numerical_imputer.fit_transform(df[[col]])

# Scale the numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 6: Prepare Data for Model Training
X = hstack([scipy.sparse.csr_matrix(df[numerical_cols].values), text_features])
y = df['is_bot'].astype(int)  # Ensure y contains integer labels

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Step 8: Balance Classes with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 9: Model Training with Ensemble
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
logistic_model = LogisticRegression(C=10, penalty='l1', solver='liblinear', max_iter=2000, random_state=42)
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('lr', logistic_model),
    ('rf', rf_model),
    ('xgb', xgb_model)
], voting='soft')
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Step 10: Model Evaluation
y_test_pred = ensemble_model.predict(X_test)
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Step 11: Check Class Distribution in Training Data
print("Class distribution in training data:", Counter(y_train))

# Step 12: Display Feature Importance (for RandomForest)
if hasattr(rf_model, "feature_importances_"):
    feature_importances = rf_model.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    print("Top 10 Feature Importances:")
    for i in range(10):  # Display top 10 features
        print(f"Feature {sorted_indices[i]}: Importance {feature_importances[sorted_indices[i]]}")
