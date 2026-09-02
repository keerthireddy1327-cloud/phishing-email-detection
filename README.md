# phishing-email-detection
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

data = pd.read_csv("phishing_emails.csv")

print("Dataset loaded successfully!")
print(data.head())


# ---------------------------------------
# 2. Clean Email Text
# ---------------------------------------

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["email_text"] = data["email_text"].apply(clean_text)


# ---------------------------------------
# 3. Convert Labels
# ---------------------------------------

data["label"] = data["label"].map({
    "Safe": 0,
    "Phishing": 1
})


# Remove rows with invalid labels
data = data.dropna(subset=["label"])


# ---------------------------------------
# 4. Separate Features and Labels
# ---------------------------------------

X = data["email_text"]
y = data["label"]


# ---------------------------------------
# 5. Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# 6. TF-IDF Feature Extraction
# ---------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ---------------------------------------
# 7. Train Machine Learning Model
# ---------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ---------------------------------------
# 8. Test Model
# ---------------------------------------

y_pred = model.predict(X_test_tfidf)


# ---------------------------------------
# 9. Accuracy
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ---------------------------------------
# 10. Classification Report
# ---------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Phishing"]
    )
)


# ---------------------------------------
# 11. Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Phishing Email Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Safe", "Phishing"])
plt.yticks([0, 1], ["Safe", "Phishing"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.colorbar()
plt.show()


# ---------------------------------------
# 12. Test New Emails
# ---------------------------------------

test_emails = [
    "Congratulations! You won $5000. Click http://free-prize.com to claim your reward.",
    "Hi Keerthi, your class is scheduled for tomorrow at 10 AM.",
    "URGENT! Your bank account has been suspended. Verify your password immediately.",
    "Your order has been shipped and will arrive tomorrow."
]


test_features = vectorizer.transform(test_emails)

predictions = model.predict(test_features)


# ---------------------------------------
# 13. Display Predictions
# ---------------------------------------

print("\nNew Email Predictions:")

for email, prediction in zip(test_emails, predictions):

    if prediction == 1:
        result = "PHISHING"
    else:
        result = "SAFE"

    print("\nEmail:")
    print(email)

    print("Prediction:", result)
