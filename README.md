# Review Sentiment Analysis using Logistic Regression

This project performs binary sentiment analysis on customer reviews using NLP preprocessing, TF-IDF vectorization, SMOTE oversampling, and Logistic Regression. It classifies reviews as Positive (1) or Negative (0) based on their text content.

## 📊 Dataset
- Input: Reviews.csv  
- Important columns: Score, Text  
- Neutral 3-star reviews are removed.  
- Sentiment labels created:  
  - 1 → Positive (Score > 3)  
  - 0 → Negative (Score < 3)  

## 🧹 Text Preprocessing
Each review is cleaned by:
1. Removing HTML tags  
2. Removing punctuation and numbers  
3. Lowercasing  
4. Removing stopwords  
5. Lemmatizing words  

Cleaned text is stored in a new column: cleaned_text.

## 🧩 Feature Extraction (TF-IDF)
TF-IDF is applied using:  
TfidfVectorizer(max_features=5000)

This converts text into numerical vectors that represent word importance.

## ⚖️ Handling Class Imbalance (SMOTE)
SMOTE is applied on the training data to balance positive and negative samples:
SMOTE(random_state=42)

## 🧠 Model (Logistic Regression)
The classifier used:
LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

It is trained on the SMOTE-enhanced TF-IDF features.

## 📈 Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

A heatmap of the confusion matrix is plotted for visualization.

## 🛠️ Technologies Used
- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- NLTK  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Google Colab / Jupyter Notebook  

## 📂 Project Structure
review-sentiment-analysis-logreg/
│── ReviewAnalysis.ipynb  
│── Reviews.csv (optional, if included)  
└── README.md  

## 📥 Dataset Download

The Amazon Reviews dataset used in this project is too large for GitHub 
(Reviews.csv = 284 MB, Reviews.zip = 114 MB).

Download the dataset from Google Drive:

🔗 https://drive.google.com/file/d/1Ake6cZ-zbp-b2sYa2FmiDnw4FtsrkyWB/view?usp=sharing

Extract `Reviews.zip` and place `Reviews.csv` in the project folder 
before running the notebook.

## ▶️ How to Run
1. Clone the repo:
   git clone https://github.com/<your-username>/review-sentiment-analysis-logreg.git
   cd review-sentiment-analysis-logreg

2. Install requirements:
   pip install pandas matplotlib seaborn nltk scikit-learn imbalanced-learn

3. Download NLTK data:
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')

4. Run the notebook:
   jupyter notebook ReviewAnalysis.ipynb  
   or upload to Google Colab.

5. Execute all cells to preprocess text, apply TF-IDF, apply SMOTE, train the model, and evaluate results.

## ⭐ Future Improvements
- Try SVM, Random Forest, or XGBoost  
- Add deep learning models (LSTM, BERT)  
- Use GridSearchCV for hyperparameter tuning  
- Deploy using Streamlit  

## 🤝 Contributions
Contributions and improvements are welcome. Feel free to submit an issue or pull request.
