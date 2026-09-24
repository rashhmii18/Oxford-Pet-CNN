
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay',
    'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon',
    'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx',
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
    'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees',
    'havanese', 'japanese_chin', 'keeshond', 'leonberger',
    'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug',
    'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
    'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

checkpoint = torch.load(
    "models/resnet_pet.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

selected_image = None

def upload_image():
    global selected_image

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png")
        ]
    )

    if file_path:
        selected_image = Image.open(file_path).convert("RGB")

        display_image = selected_image.copy()
        display_image.thumbnail((350, 350))

        photo = ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)
        image_label.image = photo

        result_label.config(
            text="Image uploaded. Click Predict."
        )

def predict_breed():

    if selected_image is None:
        result_label.config(
            text="Please upload an image first."
        )
        return

    image_tensor = transform(selected_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    breed = class_names[predicted_class.item()]
    confidence = confidence.item() * 100

    breed = breed.replace("_", " ").title()

    result_label.config(
        text=f"Predicted Breed: {breed}\n"
             f"Confidence: {confidence:.2f}%"
    )

root = tk.Tk()

root.title("Oxford-IIIT Pet Breed Classifier")
root.geometry("500x650")

title_label = tk.Label(
    root,
    text="Oxford-IIIT Pet Breed Classifier",
    font=("Arial", 18, "bold")
)

title_label.pack(pady=20)

upload_button = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 12)
)

upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

predict_button = tk.Button(
    root,
    text="Predict Breed",
    command=predict_breed,
    font=("Arial", 12)
)

predict_button.pack(pady=15)

result_label = tk.Label(
    root,
    text="Upload a pet image to begin.",
    font=("Arial", 13),
    wraplength=450
)

result_label.pack(pady=20)

root.mainloop()
