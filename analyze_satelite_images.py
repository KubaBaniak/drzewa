import torch
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

import config
from model import NN
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

checkpoint_path = 'checkpoints/best_model_epoch=14-validation_IoU=0.7436.ckpt'
image_folder = 'puszcza_notecka'
images = [('image11.jpg', 'image12.jpg'), ('image21.jpg', 'image22.jpg'), ('image31.jpg', 'image32.jpg')]

device = config.device
model = NN.load_from_checkpoint(checkpoint_path)
model.cuda()
model.eval()

for image_set in images:
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    img = Image.open(f'{image_folder}/{image_set[0]}').convert('RGB')
    row = 0
    n = 4
    masks = []
    for i in range(0, n):
        for j in range(0, n):
            crop_size = (256, 256)
            crop_position = (i * 256, j * 256)
            cropped_image = img.crop(
                (crop_position[0], crop_position[1], crop_position[0] + crop_size[0], crop_position[1] + crop_size[1]))

            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(cropped_image)

            with torch.no_grad():
                predictions = []

                image = image_tensor.unsqueeze(0)
                image = image.to(device)

                prediction = model(image)
                prediction = (prediction[0].sigmoid() > 0.5) * 255
                prediction = prediction.cpu().numpy()
                prediction_hwc = prediction.transpose(1, 2, 0).astype(np.uint8)
                masks.append(prediction_hwc)

    merged_image = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            binary_mask = masks[i + 4 * j]

            binary_mask = np.squeeze(binary_mask)

            start_row = i * 256
            start_col = j * 256

            merged_image[start_row:start_row + 256, start_col:start_col + 256] = binary_mask

    merged_image_to_save = Image.fromarray(merged_image)
    merged_image_to_save.save(f'{image_folder}/calculated_masks/mask{image_set[0][5:7]}.png')

    original_image_np = np.array(img)[:1024, :1024, :]
    axes[row, 0].set_title('Original image')
    axes[row, 0].imshow(original_image_np)
    axes[row, 1].set_title('Predicted mask')
    axes[row, 1].imshow(merged_image, cmap='gray', vmin=0, vmax=255)
    alpha = 0.5
    blended_image = (original_image_np * (1.0 - alpha) + merged_image[:, :, None] * alpha).astype('uint8')
    axes[row, 2].set_title('Image mask overlay')
    axes[row, 2].imshow(blended_image)

    row = row + 1

    img = Image.open(f'{image_folder}/{image_set[1]}').convert('RGB')
    masks = []
    for i in range(0, n):
        for j in range(0, n):
            crop_size = (256, 256)
            crop_position = (i * 256, j * 256)
            cropped_image = img.crop(
                (crop_position[0], crop_position[1], crop_position[0] + crop_size[0], crop_position[1] + crop_size[1]))

            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(cropped_image)

            with torch.no_grad():
                predictions = []

                image = image_tensor.unsqueeze(0)
                image = image.to(device)

                prediction = model(image)
                prediction = (prediction[0].sigmoid() > 0.5) * 255
                prediction = prediction.cpu().numpy()
                prediction_hwc = prediction.transpose(1, 2, 0).astype(np.uint8)
                masks.append(prediction_hwc)

    merged_image = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            binary_mask = masks[i + 4 * j]

            binary_mask = np.squeeze(binary_mask)

            start_row = i * 256
            start_col = j * 256

            merged_image[start_row:start_row + 256, start_col:start_col + 256] = binary_mask

    merged_image_to_save = Image.fromarray(merged_image)
    merged_image_to_save.save(f'{image_folder}/calculated_masks/mask{image_set[1][5:7]}.png')

    original_image_np = np.array(img)[:1024, :1024, :]
    axes[row, 0].set_title('Original image')
    axes[row, 0].imshow(original_image_np)
    axes[row, 1].set_title('Predicted mask')
    axes[row, 1].imshow(merged_image, cmap='gray', vmin=0, vmax=255)
    alpha = 0.5
    axes[row, 2].set_title('Image mask overlay')
    blended_image = (original_image_np * (1.0 - alpha) + merged_image[:, :, None] * alpha).astype('uint8')
    axes[row, 2].imshow(blended_image)

    plt.tight_layout()
    plt.show()
