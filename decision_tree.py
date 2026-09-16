# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset
pima = pd.read_csv("diabetes.csv")

# Display first 5 rows
print(pima.head())

# Split dataset into features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age',
                'Glucose', 'BloodPressure',
                'DiabetesPedigreeFunction']

X = pima[feature_cols]
y = pima['Outcome']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=1
)

# Create Decision Tree classifier
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3
)

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Display Decision Tree
from sklearn import tree
import matplotlib.pyplot as plt

feature_cols = X.columns.tolist()

plt.figure(figsize=(20, 10))

tree.plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)

plt.show()