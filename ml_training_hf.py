from datasets import load_dataset, DownloadConfig
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset from Hugging Face
download_config = DownloadConfig(
    max_retries=5,
)

dataset = load_dataset("deepset/prompt-injections", download_config=download_config)


train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

print("Train size:", len(train_df))
print("Test size:", len(test_df))

#Separate Features and Labels
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

#Convert Text to Numbers (TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams (important upgrade)
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train the AI model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


#Evaluate Model
predictions = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

#Save Model
pickle.dump(model, open("risk_model_hf.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer_hf.pkl", "wb"))

print("\nModel trained and saved successfully!")