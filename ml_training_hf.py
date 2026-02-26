from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# =========================================
# 1️⃣ Load Deepset Dataset
# =========================================

ds1 = load_dataset("deepset/prompt-injections")
df1 = ds1["train"].to_pandas()
df1 = df1[["text", "label"]]

print("Deepset dataset size:", df1.shape)


# =========================================
# 2️⃣ Load AiActivity Jailbreak Dataset
# =========================================

df2_part1 = pd.read_parquet("data/0000.parquet")
df2_part2 = pd.read_parquet("data/0001.parquet")

df2 = pd.concat([df2_part1, df2_part2], ignore_index=True)

df2 = df2[["prompt", "type"]]
df2.rename(columns={"prompt": "text"}, inplace=True)

df2["label"] = df2["type"].apply(lambda x: 1 if x == "jailbreak" else 0)
df2 = df2[["text", "label"]]

print("AiActivity dataset size:", df2.shape)

# =========================================
# Safe-Guard-prompt-injection
# =========================================
ds3 = load_dataset("xTRam1/safe-guard-prompt-injection")
df3 = ds3["train"].to_pandas()
df3 = df3[["text", "label"]]

print("Safe Guard dataset size:", df3.shape)

# =========================================
# Mosscap-Prompt-injection
# =========================================

ds4 = load_dataset("Lakera/mosscap_prompt_injection", split="train")
df4 = ds4.to_pandas()

df4 = df4[["prompt"]]
df4["label"] = 1
df4.rename(columns={"prompt": "text"}, inplace=True)

print("Mosscap dataset size:", df4.shape)

# =========================================
# 3️⃣ Merge Datasets
# =========================================

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Remove duplicates
df.drop_duplicates(subset=["text"], inplace=True)

print("Merged dataset size:", df.shape)
print("Label distribution:\n", df["label"].value_counts())


# =========================================
# 4️⃣ Train/Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# =========================================
# 5️⃣ TF-IDF Vectorization
# =========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================================
# 6️⃣ Train Logistic Regression
# =========================================

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_vec, y_train)


# =========================================
# 7️⃣ Evaluate Model
# =========================================

predictions = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# =========================================
# 8️⃣ Save Model
# =========================================

pickle.dump(model, open("risk_model_hf.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer_hf.pkl", "wb"))

print("\nMerged model trained and saved successfully!")