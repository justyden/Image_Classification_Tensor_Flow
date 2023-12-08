''' A program that uses transfer learning to train a model on bottle images.
The model that is trained is MobileNetv2. The website used for reference is
https://www.tensorflow.org/tutorials/images/transfer_learning#summary which was
used to learn this process.'''
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn

PATH = os.path.dirname('src/TrainingData/')
train_dir = PATH
BATCH_SIZE = 8
IMG_SIZE = (160, 160)
correct_seed = 121 # This is to ensure that the train_dataset and validation_dataset
# do not have the same images.

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    shuffle=True,
    subset="training",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    seed=correct_seed
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    shuffle=True,
    subset="validation",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    seed=correct_seed
)

class_names = train_dataset.class_names

# This prints a few of the images within the data set to show.
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(BATCH_SIZE):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE

# This configures the dataset.
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# This is the augmenting technique.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# This is the process of actually augmenting the data.
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1, -1) # Rescale the images.

IMG_SHAPE = IMG_SIZE + (3,)
# This creates the model and does include the top layer.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False # Ensures that the rest of the layers do not train.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Since there are multiple classes in this case, the model must know that.
num_classes = len(class_names)
prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Constructs the classification head for the model.
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Compiles the model.
base_learning_rate = 0.001 # This can be adjusted. This was the best found value.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

initial_epochs = 10 # This is also can be adjusted. It is important to try different values.

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

model.save('src/model') # Saves the model.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Confusion Matrix
predictions = model.predict(validation_dataset)
y_true = np.concatenate([labels for _, labels in validation_dataset])
y_pred = np.argmax(predictions, axis=1)

# Create confusion matrix
conf_mat = tf.math.confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 8))
seaborn.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Sparse Categorical Crossentropy Loss')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()