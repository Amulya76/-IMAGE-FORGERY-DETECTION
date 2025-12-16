import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load trained model
MODEL_PATH = r"C:\Users\amuly\OneDrive\Documents\Project\casia_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (256, 256)

# ----------- PREDICT FUNCTION -------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]  # Probability between 0 and 1
    percent_real = (1 - prob) * 100   # authenticity %
    percent_fake = prob * 100         # forgery %

    result = f"Authentic: {percent_real:.2f}%\nTampered: {percent_fake:.2f}%"
    return result, percent_fake

# ----------- GUI CLASS --------------------
class ForgeryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Forgery Detection")
        self.root.geometry("650x550")

        self.label = tk.Label(root, text="Upload an image to test", font=("Arial", 16))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=10)

        self.button = tk.Button(root, text="Upload Image",
                                command=self.upload_image, font=("Arial", 12))
        self.button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp *.gif")])

        if not file_path:
            return

        # Show image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        # Get prediction
        result, fake_prob = predict_image(file_path)

        if fake_prob > 50:
            color = "red"
        else:
            color = "green"

        self.result_label.config(text=result, fg=color)

# ----------- MAIN DRIVER --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryGUI(root)
    root.mainloop()
