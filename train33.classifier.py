import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

# Calculate accuracy
train_score = accuracy_score(y_train_predict, y_train)
test_score = accuracy_score(y_test_predict, y_test)
print('{}% of training samples were classified correctly!'.format(train_score * 100))
print('{}% of testing samples were classified correctly!'.format(test_score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Confusion matrix for training data
plot_confusion_matrix(y_train, y_train_predict, 'Confusion Matrix (Training Data)')

# Confusion matrix for testing data
plot_confusion_matrix(y_test, y_test_predict, 'Confusion Matrix (Testing Data)')

# Classification report for training data
print('Classification Report (Training Data):')
print(classification_report(y_train, y_train_predict))

# Classification report for testing data
print('Classification Report (Testing Data):')
print(classification_report(y_test, y_test_predict))

# Feature importance
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]), feature_importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), indices)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

def visualize_predictions(x_data, y_true, y_pred, title, num_samples=5):
    incorrect_predictions = np.where(y_pred != y_true)[0]
    correct_predictions = np.where(y_pred == y_true)[0]
    
    print(f'{title} - Correct predictions: {len(correct_predictions)}')
    print(f'{title} - Incorrect predictions: {len(incorrect_predictions)}')
    
    # Display some incorrect predictions
    for index in incorrect_predictions[:num_samples]:  # Display the first 'num_samples' incorrect predictions
        plt.imshow(x_data[index].reshape(28, 28), cmap='gray')  # Adjust the reshape dimensions as per your data
        plt.title(f"True: {y_true[index]}, Predicted: {y_pred[index]}")
        plt.show()

# Visualize incorrect predictions for training data
visualize_predictions(x_train, y_train, y_train_predict, 'Training Data')

# Visualize incorrect predictions for testing data
visualize_predictions(x_test, y_test, y_test_predict, 'Testing Data')

# Distribution of Correct and Incorrect Predictions
def plot_prediction_distribution(y_true, y_pred, title):
    correct_predictions = np.where(y_pred == y_true)[0]
    incorrect_predictions = np.where(y_pred != y_true)[0]

    plt.figure(figsize=(10, 6))
    plt.hist([y_pred[correct_predictions], y_pred[incorrect_predictions]], 
             bins=np.arange(len(set(labels)) + 1) - 0.5, 
             stacked=True, 
             label=['Correct', 'Incorrect'], 
             color=['green', 'red'])
    plt.legend()
    plt.xlabel('Predicted Labels')
    plt.ylabel('Number of Samples')
    plt.title(f'Distribution of Correct and Incorrect Predictions ({title})')
    plt.xticks(np.arange(len(set(labels))))
    plt.show()

# Plot prediction distribution for training data
plot_prediction_distribution(y_train, y_train_predict, 'Training Data')

# Plot prediction distribution for testing data
plot_prediction_distribution(y_test, y_test_predict, 'Testing Data')
