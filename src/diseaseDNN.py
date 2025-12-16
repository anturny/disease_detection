import sys
import numpy as np
import pandas as pd
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import data

''' In order to run this script, you need open a terminal and use the command: 
    
    python src/diseaseDNN.py disease_data
    
    Also, make sure to set the environment variable in your terminal:
    For Linux/Mac: export TF_ENABLE_ONEDNN_OPTS=0

    For Windows (PowerShell): $env:TF_ENABLE_ONEDNN_OPTS=0 

in order to setup TensorFlow properly'''

def load_and_preprocess_disease_data(test_size=0.2, random_state=42):
    disease_data = pd.read_csv('src/improved_disease_dataset.csv')
    X = disease_data.iloc[:, :-1]   # all columns except last
    y = disease_data.iloc[:, -1]    # last column as target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes, layers_count):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # Add hidden layers
    for _ in range(layers_count):
        model.add(layers.Dense(128, activation='relu'))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def main():
    # Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python src/diseaseDNN.py disease_data <number_of_layers> <epochs>")
        sys.exit(1)

    dataset_name = sys.argv[1].lower()
    layers_count = int(sys.argv[2])
    epochs = int(sys.argv[3])

    # Load dataset
    if dataset_name == 'disease_data':
        X_train, X_test, y_train, y_test = load_and_preprocess_disease_data()
    else:
        print("Unrecognized dataset")
        sys.exit(1)

    # Reshape if needed
    if X_train.ndim > 2:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Encode string labels to integers
    label_encoder = LabelEncoder()

    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_enc, num_classes)

    input_shape = (X_train.shape[1],)

    # Build and compile model
    model = build_model(input_shape, num_classes, layers_count)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    print(f"\nTraining a {layers_count}-layer deep neural network on {dataset_name} for {epochs} epochs...")
    model.fit(X_train, y_train_cat, epochs=epochs, batch_size=32, verbose=2)

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")

    # Generate predictions for training data
    y_train_pred_probs = model.predict(X_train)
    y_train_pred = np.argmax(model.predict(X_train), axis=1)

    # Generate predictions for test data
    y_test_pred_probs = model.predict(X_test)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)

    # Labels for classification report
    labels = np.unique(y_train)
    target_names = [str(i) for i in labels]

    # Classification report for training data
    print("\nClassification Report on Training Data:\n")
    print(classification_report(y_train_enc, y_train_pred, target_names=label_encoder.classes_, zero_division=0))

    # Classification report for test data
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_, zero_division=0))

    # Clear session
    try:
        keras.backend.clear_session()
    except:
        pass

    return 0

if __name__ == "__main__":
    main()