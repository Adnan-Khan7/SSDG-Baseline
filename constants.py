# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
MODEL_NAME = "resnet50"

# Number of classes in the dataset
NUM_CLASSES = 7

# Threshold for the ssl
THRESH = 0.7

# Learning rate

LR = 0.00005
# Batch size for training, validation, and testing.
BATCH_SIZE = 32

# The number of training epochs.
TRAIN_EPOCHS = 60

# Epochs for initial supervised training, should be less than TRAIN_EPOCHS
SUPERVISED_EPOCHS = 15

# Boolean whether to do Semi-Supervised Training or not
DO_SEMI = True

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
FEATURE_EXTRACTOR = False

# DATA PATHS
DATA_TRAIN_SET = "/home/adnan.khan/Desktop/ssl_vit/pacs_dataset/pacs_art_target_unlabeled/train"
DATA_UNLABELED_SET = "/home/adnan.khan/Desktop/ssl_vit/pacs_dataset/pacs_art_target_unlabeled/unlabeled"
DATA_VALID_SET = "/home/adnan.khan/Desktop/ssl_vit/pacs_dataset/pacs_art_target_unlabeled/val"
DATA_TEST_SET = "/home/adnan.khan/Desktop/ssl_vit/pacs_dataset/pacs_art_target_unlabeled/test"

# Image extensions to use from data paths
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Directory for logs to save
OUTPUT_DIR = "/home/adnan.khan/Desktop/SSDG_Baseline/outputs/exp6/"

# Save model at specific epoch
SAVE_EPOCH = 59