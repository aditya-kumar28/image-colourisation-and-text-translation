import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))
else:
  print("No NVIDIA driver found. Using CPU")


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

model = ColorizationNet().to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)



EPOCHS = 30
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        grayscale_images = rgb_to_gray(images).to(device)
        images = images.to(device)


        outputs = model(grayscale_images)
        loss = criterion(outputs, images)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Finished Training")


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):

    img = img / 2 + 0.5
    npimg = img.numpy()
    if len(img.shape) == 2:
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualize_all_three(original_images, grayscale_images, colorized_images, n=5):
    """
    Display grayscale, colorized, and original images side by side.
    n: number of images to display from the batch
    """
    fig = plt.figure(figsize=(3*n, 4))
    for i in range(n):

        ax = plt.subplot(1, 3*n, 3*i + 1)
        imshow(original_images[i])
        ax.set_title("Original")
        ax.axis("off")


        ax = plt.subplot(1, 3*n, 3*i + 2)
        imshow(grayscale_images[i])
        ax.set_title("Grayscale")
        ax.axis("off")


        ax = plt.subplot(1, 3*n, 3*i + 3)
        imshow(colorized_images[i])
        ax.set_title("Colorized")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def torch_rgb_to_hsv(rgb):
    """
    Convert an RGB image tensor to HSV.

    Parameters:
    - rgb: tensor of shape (batch_size, 3, height, width) in RGB format in the range [0, 1].

    Returns:
    - hsv: tensor of same shape in HSV format in the range [0, 1].
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val


    h = torch.zeros_like(r)
    mask = (max_val == r) & (g >= b)
    h[mask] = (g[mask] - b[mask]) / diff[mask]
    mask = (max_val == r) & (g < b)
    h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
    mask = max_val == g
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
    mask = max_val == b
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
    h = h / 6.0
    h[diff == 0.0] = 0.0


    s = torch.zeros_like(r)
    s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]


    v = max_val

    return torch.stack([h, s, v], dim=1)


def torch_hsv_to_rgb(hsv):
    """
    Convert an HSV image tensor to RGB.

    Parameters:
    - hsv: tensor of shape (batch_size, 3, height, width) in HSV format in the range [0, 1].

    Returns:
    - rgb: tensor of same shape in RGB format in the range [0, 1].
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    r[i_mod == 0.0] = v[i_mod == 0.0]
    g[i_mod == 0.0] = t[i_mod == 0.0]
    b[i_mod == 0.0] = p[i_mod == 0.0]

    r[i_mod == 1.0] = q[i_mod == 1.0]
    g[i_mod == 1.0] = v[i_mod == 1.0]
    b[i_mod == 1.0] = p[i_mod == 1.0]

    r[i_mod == 2.0] = p[i_mod == 2.0]
    g[i_mod == 2.0] = v[i_mod == 2.0]
    b[i_mod == 2.0] = t[i_mod == 2.0]

    r[i_mod == 3.0] = p[i_mod == 3.0]
    g[i_mod == 3.0] = q[i_mod == 3.0]
    b[i_mod == 3.0] = v[i_mod == 3.0]

    r[i_mod == 4.0] = t[i_mod == 4.0]
    g[i_mod == 4.0] = p[i_mod == 4.0]
    b[i_mod == 4.0] = v[i_mod == 4.0]

    r[i_mod == 5.0] = v[i_mod == 5.0]
    g[i_mod == 5.0] = p[i_mod == 5.0]
    b[i_mod == 5.0] = q[i_mod == 5.0]

    return torch.stack([r, g, b], dim=1)

def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    """
    Exaggerate the colors of RGB images.

    Parameters:
    - images: tensor of shape (batch_size, 3, height, width) in RGB format.
    - saturation_factor: factor by which to increase the saturation. Default is 1.5.
    - value_factor: factor by which to increase the value/brightness. Default is 1.2.

    Returns:
    - color_exaggerated_images: tensor of same shape as input, with exaggerated colors.
    """

    images = (images + 1) / 2.0


    images_hsv = torch_rgb_to_hsv(images)

    images_hsv[:, 1, :, :] = torch.clamp(images_hsv[:, 1, :, :] * saturation_factor, 0, 1)
    images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)


    color_exaggerated_images = torch_hsv_to_rgb(images_hsv)

    color_exaggerated_images = color_exaggerated_images * 2.0 - 1.0

    return color_exaggerated_images



with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        grayscale_images = rgb_to_gray(images).to(device)
        colorized_images = model(grayscale_images)


        grayscale_images_cpu = grayscale_images.cpu().squeeze(1)
        colorized_images_cpu = colorized_images.cpu()
        original_images_cpu = images.cpu()


        colorized_images_cpu=exaggerate_colors(colorized_images_cpu)

        visualize_all_three(original_images_cpu, grayscale_images_cpu, colorized_images_cpu)

        if i == 10:
            break

from PIL import Image

imagee='today3.jpg'
img = Image.open(imagee)


gray_img = img.convert("L")

import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),

])


img_tensor = transform(gray_img).unsqueeze(0)


model.eval()


img_tensor = img_tensor.to(device)


with torch.no_grad():
    colorized_tensor = model(img_tensor)



colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())


colorized_img.save("00_16_colorized.jpg")


fig, ax = plt.subplots(1, 3, figsize=(18, 6))


ax[0].imshow(img)
ax[0].set_title("Original Color Image")
ax[0].axis('off')


ax[1].imshow(gray_img, cmap='gray')
ax[1].set_title("Grayscale Image")
ax[1].axis('off')


ax[2].imshow(colorized_img)
ax[2].set_title("Colorized Image")
ax[2].axis('off')

plt.tight_layout()
plt.show()

gray_img.save("coro.jpg")


!pip install keras-ocr

import keras_ocr
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

def extract_text_from_image(image_path):
    # Initialize the Keras-OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Load the image
    images = [keras_ocr.tools.read(image_path)]

    # Get predictions
    prediction_groups = pipeline.recognize(images)

    # Extract and return the text
    predictions = prediction_groups[0]
    extracted_text = ' '.join([text for text, box in predictions])
    return extracted_text

# Example usage
image_path = imagee  # Replace 'example_image.jpg' with your image file path
extracted_text = extract_text_from_image(image_path)
print("Extracted Text:",extracted_text)
output=extracted_text

pip install googletrans

pip install translate

from translate import Translator

translator = Translator(to_lang="hi")  # Specify the target language
text_to_translate = output
translation = translator.translate(text_to_translate)

print(f"Original: {text_to_translate}")
print(f"Translated: {translation}")

import streamlit as st
def main():
    st.title("Text Translator App")

    # Get user input
    text_to_translate = st.text_area("Enter text to translate:", "")

    # Choose target language
    target_language = st.selectbox("Select target language:", ["en", "es", "fr", "de"])

    # Translate text when the user clicks the button
    if st.button("Translate"):
        translation = translate_text(text_to_translate, target_language)
        st.success(f"Translated text: {translation}")

def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

if __name__ == "__main__":
    main()


pip install streamlit



