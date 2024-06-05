import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Custom function to track accuracy and log loss over training epochs
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=10):
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Train the model
        model.fit(x_train, y_train)

        # Predictions
        y_train_predict = model.predict(x_train)
        y_test_predict = model.predict(x_test)
        
        # Probabilities for log loss calculation
        y_train_proba = model.predict_proba(x_train)
        y_test_proba = model.predict_proba(x_test)
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, y_train_predict)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        
        # Calculate log loss
        train_loss = log_loss(y_train, y_train_proba)
        test_loss = log_loss(y_test, y_test_proba)
        
        # Store the metrics
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_accuracies, test_accuracies, train_losses, test_losses

# Initialize the model
model = RandomForestClassifier()

# Train and evaluate the model
train_accuracies, test_accuracies, train_losses, test_losses = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=10)

# Plot accuracy over epochs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 11), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# Plot loss over epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), train_losses, label='Train Loss')
plt.plot(range(1, 11), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
