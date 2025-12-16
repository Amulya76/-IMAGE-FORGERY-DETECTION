import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
import imghdr   # To detect real image types
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================================================
# 1. PATHS TO YOUR DATA
# =====================================================
AU_DIR = r"C:\Users\amuly\OneDrive\Documents\Au"
TP_DIR = r"C:\Users\amuly\OneDrive\Documents\Tp"
BAD_FILES_LOG = "bad_files.txt"       # Save all bad files here
VALID_FILES_LOG = "valid_files.txt"   # Save valid image paths here

# =====================================================
# 2. HYPERPARAMETERS
# =====================================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "casia_cnn.h5"

# =====================================================
# 3. LOAD ONLY VALID IMAGE FILES & SAVE BAD FILES
# =====================================================
bad_files = []
valid_files = []

def get_image_files(folder):
    all_files = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)

    image_files = []
    for f in all_files:
        if imghdr.what(f) in ["jpeg", "png", "bmp", "gif"]:
            image_files.append(f)
            valid_files.append(f)
        else:
            print(f"❌ Skipped NON-image file: {f}")
            bad_files.append(f)

    return image_files

au_files = get_image_files(AU_DIR)
tp_files = get_image_files(TP_DIR)

# Save logs
with open(BAD_FILES_LOG, "w") as f:
    for b in bad_files:
        f.write(b + "\n")

with open(VALID_FILES_LOG, "w") as f:
    for v in valid_files:
        f.write(v + "\n")

print(f"📁 Bad files saved to: {BAD_FILES_LOG}")
print(f"📁 Valid images saved to: {VALID_FILES_LOG}")
print(f"✔ Valid authentic images: {len(au_files)}")
print(f"✔ Valid tampered images: {len(tp_files)}")

if len(au_files) == 0 or len(tp_files) == 0:
    raise RuntimeError("No valid images found. Check dataset folders!")

all_files = np.array(au_files + tp_files)
labels = np.array([0] * len(au_files) + [1] * len(tp_files))

# =====================================================
# 4. TRAIN/VALIDATION SPLIT
# =====================================================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_files, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")

# =====================================================
# 5. TF DATASET PIPELINE (SAFE VERSION)
# =====================================================
AUTOTUNE = tf.data.AUTOTUNE

def decode_and_resize(filename, label):
    img = tf.io.read_file(filename)

    try:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    except:
        print(f"❌ Bad image skipped at decode stage: {filename}")
        return tf.zeros((256, 256, 3)), tf.cast(label, tf.float32)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)

def build_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = build_dataset(train_paths, train_labels, shuffle=True)
val_ds   = build_dataset(val_paths, val_labels, shuffle=False)

# =====================================================
# 6. CNN MODEL
# =====================================================
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # binary classification
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model(IMG_SIZE + (3,))
model.summary()

# =====================================================
# 7. TRAIN MODEL
# =====================================================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor="val_loss"
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print(f"🎉 Training finished! Model saved to: {MODEL_PATH}")

# =====================================================
# 8. MODEL PREDICTION FOR CONFUSION MATRIX
# =====================================================
val_pred = model.predict(val_ds)
val_pred = (val_pred > 0.5).astype(int)

np.save("Y_true.npy", val_labels)
np.save("Y_pred.npy", val_pred)
print("Saved Y_true.npy & Y_pred.npy for PPT integration.")

# =====================================================
# 9. CONFUSION MATRIX (SAVE IMAGE)
# =====================================================
cm = confusion_matrix(val_labels, val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# =====================================================
# 10. PLOT & SAVE ACCURACY/LOSS
# =====================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_loss_graph.png")
plt.show()

print("🎯 ALL DONE! Now send me these files for PPT:")
print("✔ confusion_matrix.png")
print("✔ accuracy_loss_graph.png")
print("✔ gui_screenshot.png (take manually)")
print("✔ Y_true.npy & Y_pred.npy")
