from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ForestDataset(Dataset):

    def __init__(self, img_path, mask_path, metadata, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_full_path = self.img_path + self.metadata['image'][idx]
        mask_full_path = self.mask_path + self.metadata['mask'][idx]
        image = Image.open(image_full_path).convert('RGB')
        mask = Image.open(mask_full_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

            return image, mask

        return transforms.ToTensor()(image), transforms.ToTensor()(mask)
