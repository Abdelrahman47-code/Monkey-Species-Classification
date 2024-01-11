import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Global variable to store the background image reference
background_image = None

# Load the saved model
saved_model_path = 'monkey_species_model.h5'
model = load_model(saved_model_path)

# Define class labels
class_labels = ['Bald Uakari', 'Emperor Tamarin', 'Golden Monkey', 'Gray Langur', 'Hamadryas Baboon',
                'Mandril', 'Proboscis Monkey', 'Red Howler', 'Vervet Monkey', 'White Faced Saki']

# Function to classify an image
def classify_image(image_path, model, class_labels):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class]

    return predicted_class_label

# Function to handle image selection and classification
def browse_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        predicted_label = classify_image(file_path, model, class_labels)
        
        # Display the selected image and the predicted class
        display_image(file_path, predicted_label)

# Function to display the image and predicted class
def display_image(image_path, predicted_label):
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    # Update the Tkinter labels
    image_label.config(image=img_tk)
    image_label.image = img_tk
    predicted_label_var.set(f'Predicted Class: {predicted_label}')

# Create the main Tkinter window
root = tk.Tk()
root.title("Monkey Species Classifier")

# Set window size and background color
root.geometry("1200x600")
root.configure(bg="white")

# Load and keep a reference to the background image globally
background_image = ImageTk.PhotoImage(file="monkey_background.png")
background_label = tk.Label(root, image=background_image, bg="white")
background_label.place(relwidth=0.43, relheight=1.55)

# Create and configure the GUI elements with improved style and rounded edges
style = ttk.Style()
style.configure("Rounded.TLabel", background="#88AB8E", font=("Helvetica", 18), padding=10, relief="ridge", borderwidth=10)
style.configure("Rounded.TButton", background="#88AB8E", font=("Helvetica", 14), padding=10, relief="ridge", borderwidth=10)

welcome_label = ttk.Label(root, text="🐒 Welcome to Monkey Species Classifier! 🤖", style="Rounded.TLabel")
browse_button = ttk.Button(root, text="Browse", command=browse_image, style="Rounded.TButton")
image_label = tk.Label(root)
predicted_label_var = tk.StringVar()
predicted_label = ttk.Label(root, textvariable=predicted_label_var, style="Rounded.TLabel")

# Watermark label
watermark_label = tk.Label(root, text="Made by: Abdelrahman Eldaba", font=("Helvetica", 14), bg="white", fg="black")
watermark_label.place(relx=0.5, rely=0.99, anchor="s")

# Place the elements in the window
welcome_label.pack(pady=20)
browse_button.pack(pady=20)
image_label.pack(pady=20)
predicted_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()