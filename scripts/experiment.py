import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Step 1: Load and preprocess the data
df = pd.read_csv('data/data.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Save the processed data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Step 4: Version the data with DVC
os.system('dvc add data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv')
os.system('dvc push')

# Step 5: Start an MLFlow run and train the model
with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Step 6: Predict and evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Step 7: Log parameters, metrics, and model
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    # Step 8: Save and version the model
    joblib.dump(model, 'models/model.pkl')
    os.system('dvc add models/model.pkl')
    os.system('dvc push')

    # Step 9: Evaluate the model
    conf_matrix = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    
    # Step 10: Log additional artifacts
    mlflow.log_artifact('confusion_matrix.png')
