import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from sklearn.metrics import confusion_matrix, classification_report

def load_and_preprocess_data():
    """
    Load the MNIST dataset and normalize the images.
    
    Returns:
        (x_train, y_train): Training data and labels.
        (x_test, y_test): Test data and labels.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model(input_shape=(28, 28), num_classes=10):
    """
    Build and compile a simple neural network model.
    
    Args:
        input_shape: Shape of the input images.
        num_classes: Number of output classes.
        
    Returns:
        model: A compiled Keras model.
    """
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),  # Flatten images
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, output_dir="plots"):
    """
    Plot and save training and validation accuracy and loss.
    
    Args:
        history: History object returned by model.fit.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    plt.show()

def plot_confusion_matrix(cm, classes, output_dir="plots", normalize=True,
                          title='Normalized Confusion Matrix', cmap=plt.cm.Blues,
                          filename="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    
    Args:
        cm: Confusion matrix.
        classes: List of class labels.
        output_dir: Directory to save plots.
        normalize: Boolean; if True, normalize the confusion matrix.
        title: Title of the plot.
        cmap: Colormap for the plot.
        filename: Filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, filename)
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    plt.show()

def plot_sample_predictions(x_test, y_test, predictions, output_dir="plots", filename="sample_predictions.png"):
    """
    Plot and save a grid of sample predictions.
    
    Args:
        x_test: Test images.
        y_test: True labels.
        predictions: Model predictions.
        output_dir: Directory to save plots.
        filename: Filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    num_rows, num_cols = 5, 5
    num_images = num_rows * num_cols
    indices = np.random.choice(len(x_test), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x_test[idx], cmap=plt.cm.binary)
        pred_label = np.argmax(predictions[idx])
        plt.title(f"True: {y_test[idx]}\nPred: {pred_label}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    sample_plot_path = os.path.join(output_dir, filename)
    plt.savefig(sample_plot_path)
    print(f"Sample predictions plot saved to {sample_plot_path}")
    plt.show()

def plot_misclassified_images(x_test, y_test, predictions, output_dir="plots", filename="misclassified.png"):
    """
    Plot and save a grid of misclassified images.
    
    Args:
        x_test: Test images.
        y_test: True labels.
        predictions: Model predictions.
        output_dir: Directory to save plots.
        filename: Filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    misclassified = np.where(np.argmax(predictions, axis=1) != y_test)[0]
    if len(misclassified) == 0:
        print("No misclassified images found!")
        return
    
    num_images = min(25, len(misclassified))
    indices = np.random.choice(misclassified, num_images, replace=False)
    
    plt.figure(figsize=(10, 10))
    num_rows, num_cols = 5, 5
    for i, idx in enumerate(indices):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x_test[idx], cmap=plt.cm.binary)
        pred_label = np.argmax(predictions[idx])
        plt.title(f"T:{y_test[idx]}\nP:{pred_label}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    misclassified_plot_path = os.path.join(output_dir, filename)
    plt.savefig(misclassified_plot_path)
    print(f"Misclassified images plot saved to {misclassified_plot_path}")
    plt.show()

def plot_probability_distribution(x_test, y_test, predictions, index=None, output_dir="plots", filename="probability_distribution.png"):
    """
    Plot and save the softmax probability distribution for a sample image.
    
    Args:
        x_test: Test images.
        y_test: True labels.
        predictions: Model predictions.
        index: Index of the sample image to plot. If None, a random index is chosen.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if index is None:
        index = np.random.randint(0, len(x_test))
    
    sample_pred = predictions[index]
    true_label = y_test[index]
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(10), sample_pred)
    plt.xticks(np.arange(10))
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title(f"Probability Distribution for Sample Index {index} (True label: {true_label})")
    prob_plot_path = os.path.join(output_dir, filename)
    plt.savefig(prob_plot_path)
    print(f"Probability distribution plot saved to {prob_plot_path}")
    plt.show()

def save_model_architecture(model, output_dir="plots", filename="model_architecture.png"):
    """
    Save a plot of the model architecture.
    
    Args:
        model: Keras model instance.
        output_dir: Directory to save the plot.
        filename: Filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(output_dir, filename), show_shapes=True, show_layer_names=True)
        print(f"Model architecture plot saved to {os.path.join(output_dir, filename)}")
    except Exception as e:
        print("Could not save model architecture plot. Please ensure that 'pydot' and Graphviz are installed.")
        print(str(e))

def print_classification_report(y_test, predictions):
    """
    Print the classification report showing precision, recall, and F1-score.
    
    Args:
        y_test: True labels.
        predictions: Model predictions.
    """
    y_pred = np.argmax(predictions, axis=1)
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
    print("Classification Report:")
    print(report)

def main():
    # Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build and summarize the model
    model = build_model(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()
    
    # Save the model architecture visualization
    save_model_architecture(model)
    
    # Define callbacks: EarlyStopping and ModelCheckpoint
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint_path = "best_model.h5"
    model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                                 save_best_only=True, verbose=1)
    
    # Train the model with callbacks
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=32,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stop, model_checkpoint])
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest Accuracy: {test_acc:.4f}')
    
    # Generate and save training history plots
    plot_history(history)
    
    # Make predictions on the test data
    predictions = model.predict(x_test)
    
    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
    class_labels = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, classes=class_labels, normalize=True)
    
    # Plot and save sample predictions
    plot_sample_predictions(x_test, y_test, predictions)
    
    # Print classification report
    print_classification_report(y_test, predictions)
    
    # Plot and save misclassified images
    plot_misclassified_images(x_test, y_test, predictions)
    
    # Plot and save the probability distribution for a sample image
    plot_probability_distribution(x_test, y_test, predictions)

if __name__ == '__main__':
    main()
