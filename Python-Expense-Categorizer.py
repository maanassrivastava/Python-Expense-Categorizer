import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def python_expense_categorizer():
    try:
        df = pd.read_csv('training_data.csv')
        df['description'] = df['description'].astype(str).str.lower().str.strip()
        df['category'] = df['category'].str.strip()
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), sublinear_tf=True)
        X = vectorizer.fit_transform(df['description'])
        y = df['category']
        base_svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
        model = CalibratedClassifierCV(base_svm, cv=3)
        model.fit(X, y)
        joblib.dump(model, 'finance_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
        print("\n" + "="*26)
        print("PYTHON EXPENSE CATEGORIZER")
        print("="*26)
        print("Type your expense or 'STOP' to exit.\n")
        while True:
            user_input = input("Enter expense: ").lower().strip()            
            if user_input == 'stop':
                print("\nThank you for using. Session terminated.")
                break
            if not user_input:
                continue
            input_vec = vectorizer.transform([user_input])
            probs = model.predict_proba(input_vec)[0]
            confidence = max(probs) * 100
            if confidence < 25.0:
                prediction = "Miscellaneous / Uncategorized"
                note = " (Low confidence: No strong match found)"
            else:
                prediction = model.predict(input_vec)[0]
                note = ""
            print(f"   >> [Category]: {prediction.upper()}")
            print(f"   >> [Confidence]: {confidence:.2f}%{note}\n")
    except FileNotFoundError:
        print("ERROR: 'training_data.csv' not found.")
    except Exception as e:
        print(f"SYSTEM ERROR: {e}")

if __name__ == "__main__":
    python_expense_categorizer()
