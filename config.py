import torch

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'Forest segmentation'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
N_IMAGE_CHANNELS = 3
N_MASK_CHANNELS = 1
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, N_IMAGE_CHANNELS)
MASK_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, N_MASK_CHANNELS)

NUM_CLASSES = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 16
NUM_EPOCHS = 6

IMAGE_PATH = "Forest Segmented/Forest Segmented/images/"
MASK_PATH = "Forest Segmented/Forest Segmented/masks/"
METADATA_CSV_PATH = "Forest Segmented/Forest Segmented/meta_data.csv"
NUM_WORKERS = 8

ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32