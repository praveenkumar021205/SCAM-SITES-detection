
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re

# Sample dataset (You can replace with a real one with more URLs)
data = {
    'url': [
        'http://example.com/login',
        'https://secure.paypal.com/us/cgi-bin/webscr?cmd=_login-run',
        'http://freegift.ru/scam',
        'https://accounts.google.com',
        'http://malicious-login.info/steal-data'
    ],
    'label': [0, 0, 1, 0, 1]  # 0 = Legit, 1 = Scam
}
df = pd.DataFrame(data)

# URL Feature Extractor
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['has_https'] = int('https' in url)
    features['num_digits'] = sum(char.isdigit() for char in url)
    features['has_login'] = int('login' in url.lower())
    features['has_free'] = int('free' in url.lower())
    features['num_special_char'] = len(re.findall(r'[^\w\s]', url))
    return features

# Extract features from all URLs
feature_list = [extract_features(url) for url in df['url']]
features_df = pd.DataFrame(feature_list)

# Labels
labels = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Test on a new URL
def predict_url(url):
    features = pd.DataFrame([extract_features(url)])
    prediction = model.predict(features)[0]
    print(f"\nURL: {url}")
    print("Prediction:", "SCAM" if prediction == 1 else "Legit")

# Try a new URL
predict_url("http://paypal-login-freegift.com")
