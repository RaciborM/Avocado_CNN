import os
import tensorflow as tf
from tensorflow.keras import layers, models
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# dataset path
data_dir = 'C:\\Users\\misie\\studia\\sem2\\awokado\\baza_danych\\baza_danych\\luminacja\\fazy1_3_5'

# parameters
batch_size = 32
img_height = 200
img_width = 200
epochs = 50

def extract_class_from_filename(filename):
    match = re.search(r'[ab]_(\d+)\.jpg$', filename)
    if match:
        class_num = int(match.group(1))
        if class_num == 1:
            return 0
        elif class_num == 3:
            return 1
        elif class_num == 5:
            return 2
        else:
            raise ValueError(f"Nieznana klasa w nazwie pliku: {filename}")
    else:
        raise ValueError(f"Nie udało się wyciągnąć klasy z nazwy pliku: {filename}")

def load_images_from_subfolders_with_labels(data_dir, img_height, img_width, batch_size):
    image_paths = []
    image_labels = []
    
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(subdir, file)
                image_paths.append(full_path)
                image_labels.append(extract_class_from_filename(file))
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    
    # preprocess and augmentation
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_height, img_width])
        image = image / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        angles = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=angles)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        return image, label
    
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)
    return dataset

dataset = load_images_from_subfolders_with_labels(data_dir, img_height, img_width, batch_size)

# train. and valid.
val_size = int(0.25 * len(dataset))
train_dataset = dataset.skip(val_size)
val_dataset = dataset.take(val_size)

# layers
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(512, (3,3), activation='relu'),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# val. prediction
val_images = []
val_labels = []
for images, labels in val_dataset:
    val_images.append(images)
    val_labels.append(labels)

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

# prediction
predictions = model.predict(val_images)
predicted_classes = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(val_labels, predicted_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2])
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz pomyłek")
plt.show()

plt.plot(history.history['accuracy'], label='Dokładność treningu')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
plt.title('Dokładność modelu')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='Strata treningu')
plt.plot(history.history['val_loss'], label='Strata walidacji')
plt.title('Strata modelu')
plt.ylabel('Strata')
plt.xlabel('Epoka')
plt.legend(loc='upper right')
plt.show()

# filters visualization
filters, biases = model.layers[0].get_weights()
filters_min, filters_max = filters.min(), filters.max()
filters = (filters - filters_min) / (filters_max - filters_min)

plt.figure(figsize=(10, 10))
for i in range(filters.shape[-1]):
    ax = plt.subplot(6, 6, i + 1)
    plt.imshow(filters[:, :, 0, i], cmap='viridis')
    plt.axis('off')
    plt.title(f"Filter {i+1}")
plt.show()

# activation map visualization 
test_image = next(iter(val_dataset))[0][0]
test_image = tf.expand_dims(test_image, axis=0)
_ = model.predict(test_image)

activation_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers if 'conv' in layer.name])
activations = activation_model.predict(test_image)

# data visual.
for layer_activation in activations:
    n_filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    display_grid = np.zeros((size, n_filters * size))
    for i in range(n_filters):
        x = layer_activation[0, :, :, i]
        x -= x.mean()
        x /= x.std() + 1e-5
        x = np.clip(x, 0, 1)
        display_grid[:, i * size:(i + 1) * size] = x
    plt.figure(figsize=(20, 20))
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.title(f"Aktywacje dla warstwy {layer_activation.shape}")
    plt.show()
