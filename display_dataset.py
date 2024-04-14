import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

number_of_images = 3

def display_image_and_mask(csv_file):
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(number_of_images, 3, figsize=(10, 10))

    for i in range(number_of_images):
        if i < len(df):
            image_name = df.iloc[i]['image']
            mask_name = df.iloc[i]['mask']

            image = Image.open(f'Forest Segmented/Forest Segmented/images/{image_name}').convert('RGB')
            mask = Image.open(f'Forest Segmented/Forest Segmented/masks/{mask_name}').convert('RGB')
            blended_image = Image.blend(image, mask, 0.5)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Image {i + 1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Mask {i + 1}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(blended_image)
            axes[i, 2].set_title(f'Image {i + 1} with mask {i + 1}')
            axes[i, 2].axis('off')
        else:
            break

    plt.tight_layout()
    plt.show()


# Example usage
csv_file_path = 'Forest Segmented/Forest Segmented/meta_data.csv'
display_image_and_mask(csv_file_path)
