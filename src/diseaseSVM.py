import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import data
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_disease_data(test_size=0.2, random_state=42):
    disease_data = pd.read_csv('src/improved_disease_dataset.csv')
    X = disease_data.iloc[:, :-1]   # all columns except last
    y = disease_data.iloc[:, -1]    # last column as target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/diseaseSVM.py disease_data")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    
    # Load data based on the dataset name
    if dataset_name == 'disease_data':
        X_train, X_test, y_train, y_test = load_and_preprocess_disease_data()
    else:
        print(f"Dataset '{dataset_name}' is not recognized.")
        sys.exit(1)
    
    # Initialize and train the SVM classifier
    print(f"Training SVM on {dataset_name} data...")
    clf = SVC(gamma='scale')  # using default parameters; can be tuned
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on {dataset_name} test set: {accuracy * 100:.2f}%")

    # Print classification report for the labels present in this test set only
    labels = np.unique(y_test)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # Return normally so callers (including interactive shells) can decide how to exit
    return 0

if __name__ == "__main__":
    # Call main and exit the interpreter process with its return code
    sys.exit(main())