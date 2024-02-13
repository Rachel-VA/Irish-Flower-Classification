""" 
install libs:
pip install tabulate
pip install scikit-learn
pip install seaborn
"""

# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate 

# Step 2: Load the Iris dataset
iris = load_iris()

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Step 4: Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test)

# Step 6: Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
#print(f"Accuracy: {accuracy:.2f}")
# Define the confusion matrix here
cm = confusion_matrix(y_test, predictions)

#step 7: Print out the accuracy using tabulate for a nicer format
table_data = [
    ["Accuracy", f"{accuracy:.2f}"],
    ["Number of Setosa correctly classified", cm[0, 0]],
    ["Number of Versicolor correctly classified", cm[1, 1]],
    ["Number of Virginica correctly classified", cm[2, 2]],
]

# Print the table
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))

#Display a confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
