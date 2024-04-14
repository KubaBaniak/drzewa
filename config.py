import torch

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 1
BATCH_SIZE = 32
NUM_EPOCHS = 15

IMAGE_PATH = "Forest Segmented/Forest Segmented/images/"
MASK_PATH = "Forest Segmented/Forest Segmented/masks/"
METADATA_CSV_PATH = "Forest Segmented/Forest Segmented/meta_data.csv"
NUM_WORKERS = 12

ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32
