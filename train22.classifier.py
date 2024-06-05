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
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Error Analysis
# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_predict))

# Precision, Recall, F-score, Support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict, average=None)
for i, (p, r, f, s) in enumerate(zip(precision, recall, fscore, support)):
    print(f"Class {i}: Precision={p:.2f}, Recall={r:.2f}, F1-Score={f:.2f}, Support={s}")

# Visualize incorrect predictions
incorrect_predictions = np.where(y_predict != y_test)[0]

# Assuming your data are images, modify this if your data are different
for index in incorrect_predictions[:5]:  # Display the first 5 incorrect predictions
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')  # Adjust the reshape dimensions as per your data
    plt.title(f"True: {y_test[index]}, Predicted: {y_predict[index]}")
    plt.show()

# Result Analysis
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

# Distribution of Correct and Incorrect Predictions
correct_predictions = np.where(y_predict == y_test)[0]
incorrect_predictions = np.where(y_predict != y_test)[0]

print(f'Correct predictions: {len(correct_predictions)}')
print(f'Incorrect predictions: {len(incorrect_predictions)}')

# Distribution of predictions
plt.figure(figsize=(10, 6))
plt.hist([y_predict[correct_predictions], y_predict[incorrect_predictions]], bins=np.arange(len(set(labels)) + 1) - 0.5, stacked=True, label=['Correct', 'Incorrect'], color=['green', 'red'])
plt.legend()
plt.xlabel('Predicted Labels')
plt.ylabel('Number of Samples')
plt.title('Distribution of Correct and Incorrect Predictions')
plt.xticks(np.arange(len(set(labels))))
plt.show()
