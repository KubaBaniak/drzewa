import matplotlib.pyplot as plt
import os
from PIL import Image

folder_path = 'puszcza_notecka'
image_files = ['image11.jpg', 'image12.jpg', 'image21.jpg', 'image22.jpg', 'image31.jpg', 'image32.jpg']

# Set up the plot
fig, axes = plt.subplots(3, 2, figsize=(6, 9))
axes = axes.ravel()

# Loop through the images and display them
for i, img_file in enumerate(image_files):
    img_path = os.path.join(folder_path, img_file)
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
