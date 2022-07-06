# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
MODEL_NAME = "resnet50"

# Number of classes in the dataset
NUM_CLASSES = 7

# Threshold for the ssl
THRESH = 0.8

# Learning rate

LR = 0.00005
# Batch size for training, validation, and testing.
BATCH_SIZE = 64

# The number of training epochs.
TRAIN_EPOCHS = 30

# Epochs for initial supervised training, should be less than TRAIN_EPOCHS
SUPERVISED_EPOCHS = 5

# Boolean whether to do Semi-Supervised Training or not
DO_SEMI = True

# DATA PATHS
DATA_TRAIN_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/Unlabeled_PACS/pacs_sketch_target_unlabeled/train"
DATA_UNLABELED_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/Unlabeled_PACS/pacs_sketch_target_unlabeled/unlabeled"
#DATA_VALID_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/pacs_dataset/pacs_art_target_unlabeled/val"
DATA_TEST_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/Unlabeled_PACS/pacs_sketch_target_unlabeled/test"

# Image extensions to use from data paths
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Directory for logs to save
OUTPUT_DIR = "/home/adnan.khan/Desktop/SSDG_Baseline/misc/"
