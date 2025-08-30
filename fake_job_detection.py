#=============================
#           DAY 1
#=============================

import os
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline 
from wordcloud import WordCloud
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print("Current working directory: ", os.getcwd())

filename = "fake_job_posting.csv"

# Check if file exists
if not os.path.exists(filename):
    # Try to look for common alternatives
    possible_files = [f for f in os.listdir() if "fake" in f.lower() and f.endswith(".csv")]
    if possible_files:
        filename = possible_files[0]
        print(f"Using found file: {filename}")
    else:
        raise FileNotFoundError(f"File {filename} not found in {os.getcwd()}")

DATA = pd.read_csv(filename)
print("Dataset loaded successfully...")       
print("Shape(rows,columns): ", DATA.shape)
print("Columns: ", DATA.columns.tolist())
DATA.head()

DATA.dropna(subset=['title','description','requirements'], inplace=True)                     #Remove missing values

DATA.reset_index(drop=True, inplace=True)                                                    #resets the index from{0,1,3,4} ---> {0,1,2,3}

DATA.drop_duplicates(inplace=True)                                                          #Removes duplicates from the the data

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))                            #Removes the punctuation marks from the text
    text = ' '.join([word for word in text.split() if word not in stop_words])                #Removes stopwords from the text and then rebuilds the text without them.
    return text

DATA['clean_description'] = DATA['description'].apply(clean_text)                             # Apply clean_text on 'description' → lowercase, remove punctuation & stopwords → save as 'clean_description'

DATA.to_csv("clean_fake_job_posting.csv", index=False)
print("Cleaned dataset saved as 'clean_fake_job_posting.csv'")

#=============================
#           DAY 2
#=============================

DATA = pd.read_csv(filename)

print("Class Distribution (0=Real , 1=Fake):")
print(DATA['fraudulent'].value_counts())                                                   #Prints the number of Real and Fake jobs
print("\nMissing values in each column:")                                           
print(DATA.isnull().sum())                                                                 #Checks missing values in columns

categorical_cols = ['title','location','department','company_profile','industry','function']
for col in categorical_cols:
    if col in DATA.columns:
        DATA[col].fillna("unknown", inplace=True)                                           #Replaces none/NaN values with "unknown"

numerical_cols = ["salary_range"]
for col in numerical_cols:
    if col in DATA.columns:
        # Convert to numeric first
        DATA[col] = pd.to_numeric(DATA[col], errors='coerce')
        DATA[col].fillna(DATA[col].median(), inplace=True)                                  #Replaces the missing value in column with the middle value


print("\nMissing values after handling:")
print(DATA.isnull().sum())                                                                 #checks if there is any missing value left or not


for col in categorical_cols:
    if col in DATA.columns:
        print(f"\nUnique values in {col}:")
        print(DATA[col].nunique())                                                         #Prints the number of unique values in that particular column
        print(DATA[col].unique()[:10])                                                     #Prints atleast 10 unique values

DATA['fraudulent'].value_counts().plot(kind='bar')
plt.title("Class Distribution: Real vs Fake Jobs")
plt.xlabel("Class(0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()

print("\nCorrelation matrix (numerical features):")
print(DATA.corr(numeric_only=True))   

#=============================
#           DAY 3
#=============================

DATA['description'] = DATA['description'].fillna("")                                      #Replaces none values with empty string

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')                           #Creates a vectorizer object to keep 5000 imp words and ignore stop_words
X = tfidf.fit_transform(DATA['description'])                                            # Convert the 'description' column into numerical TF-IDF vectors (ignores English stop words)

y = DATA['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                               random_state=42,
                                               stratify=y)                             # split data 80/20 with balanced classes
print("Data prepared successfully!")
print("Training samples: ", X_train.shape[0])
print("Testing samples: ", X_test.shape[0])

#=============================
#           DAY 4
#=============================

log_reg = LogisticRegression(max_iter=1000)                                             #Creates a logistic regression model that can train itself upto 1000 learning steps  
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)                                                        # Use the trained model to predict outputs (y) for new input data (X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))                                      #Overall correctness/accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))                #A detailed report about how well your model performed for each class
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))                          #Table showing correct vs wrong prediction

#=============================
#           DAY 5
#=============================

cm = confusion_matrix(y_test, y_pred)                                                      #Table showing correct vs wrong prediction

plt.figure(figsize=(6,4))                                                               #Plots a blank canvas of size 6 x 4
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",                 
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])                                          #It makes the confusion matrix easy to read visually.
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

#tests the model on a new job ad to see if it’s fake or real
sample_job = ["Work from home,$5000 per week, no skills required!"]                       # A new job description we want to test on the trained model
sample_vector = tfidf.transform(sample_job)                                               # Convert the new job description into numerical features using the same TF-IDF vectorizer
sample_pred = log_reg.predict(sample_vector)                                              # Use the trained Logistic Regression model to predict (0 = Not Fraudulent, 1 = Fraudulent)

print("\nMini Project Demo Prediction: ", 
      "Fraudulent" if sample_pred[0] == 1 else "Not Fraudulent")                        # Print out the prediction in human-readable form

#=============================
#           DAY 6
#=============================

joblib.dump(log_reg, "log_reg_model.pkl")                                               # Save the trained Logistic Regression model to a file
joblib.dump(tfidf, "tfidf_vectorizer.pkl")                                              # Save the trained TF-IDF vectorizer to a file
print("Model and vectorizer saved successfully!")

log_reg = joblib.load("log_reg_model.pkl")                                               # Load the previously saved Logistic Regression model from the file
tfidf = joblib.load("tfidf_vectorizer.pkl")                                              # Load the previously saved TF-IDF vectorizer from the file
print("Model and vectorizer loaded successfully!")

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('model', log_reg)
])                                                                                    # Combines TF-IDF and model: input text → vectorized → prediction in one step

joblib.dump(pipeline, "pipeline_model.pkl")                                           # Save the entire pipeline (TF-IDF + model) to a file for later use
print("Pipeline model saved successfully!")   

def predict_job(description):
    vector = tfidf.transform([description])                                             # Turns text into numbers so the model can understand and make predictions.
    prediction = log_reg.predict(vector)                                                # Takes the numeric features and outputs the model’s prediction.
    return "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

print("\n=== Mini Project Interactive Demo ===")
print("Type exit to quit.")

while True:
    user_input = input("\nEnter a job description to check: ")
    if user_input.lower() == 'exit':
        print("Exiting interactive demo.")
        break
    result = predict_job(user_input)
    print(f"Prediction: {result}")

try:
    batch_file = "new_jobs.csv"
    new_jobs = pd.read_csv(batch_file)
    if 'description' in new_jobs.columns:
        new_jobs['Prediction'] = new_jobs['description'].apply(predict_job)
        new_jobs.to_csv("predicted_jobs.csv", index=False)
        print(f"\nBatch predictions saved to 'predicted_jobs.csv'")
    else:
        print(f"\nColumn 'description' not found in {batch_file}")
except FileNotFoundError:
    print("\nNo batch file found. Skipping batch predictions")

#=============================
#           DAY 7
#=============================

# Ensure all clean_description values are strings
DATA['clean_description'] = DATA['clean_description'].astype(str)

# 1️⃣ Visualize most frequent words in fake vs real job descriptions
fake_text = " ".join(DATA[DATA['fraudulent']==1]['clean_description'])
real_text = " ".join(DATA[DATA['fraudulent']==0]['clean_description'])

# WordCloud for Fake jobs
wc_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in FAKE Jobs")
plt.show()

# WordCloud for Real jobs
wc_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.figure(figsize=(10,5))
plt.imshow(wc_real, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in REAL Jobs")
plt.show()

# 2️⃣ Enhanced confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=["Not Fraud","Fraud"],
            yticklabels=["Not Fraud","Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap (Final)")
plt.show()

# 3️⃣ Mini interactive demo for portfolio / resume
print("\n=== Final Interactive Demo ===")
print("Type 'exit' to quit.")
while True:
    user_input = input("\nEnter a job description to check: ")
    if user_input.lower() == 'exit':
        print("Exiting demo. Thank you!")
        break
    prediction = pipeline.predict([user_input])[0]
    print("Prediction:", "Fraudulent ❌" if prediction==1 else "Not Fraudulent ✅")

# 4️⃣ Optional: Save demo predictions for a few sample jobs
sample_jobs = [
    "Work from home, $5000 per week, no skills required!",
    "Software Engineer at leading tech company, 3+ years experience required",
    "Earn $1000 daily from home, no investment needed"
]

demo_results = [(job, "Fraudulent ❌" if pipeline.predict([job])[0]==1 else "Not Fraudulent ✅") 
                for job in sample_jobs]

print("\nSample Job Predictions:")
for job, pred in demo_results:
    print(f"- {job[:60]}... => {pred}")
