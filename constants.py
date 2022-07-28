# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
MODEL_NAME = "resnet50"

# Number of classes in the dataset
NUM_CLASSES = 7

# Threshold for the ssl
THRESH = 0.2

# Learning rate

LR = 0.00005
# Batch size for training, validation, and testing.
BATCH_SIZE = 32

# The number of training epochs.
TRAIN_EPOCHS = 40

# Epochs for initial supervised training, should be less than TRAIN_EPOCHS
SUPERVISED_EPOCHS = 5

# Boolean whether to do Semi-Supervised Training or not
DO_SEMI = True

# DATA PATHS
DATA_TRAIN_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/pacs_art_target_unlabeled_dummy/train"
DATA_UNLABELED_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/pacs_art_target_unlabeled_dummy/unlabeled"
DATA_TEST_SET = "/home/adnan.khan/Desktop/SSDG_Baseline/pacs_art_target_unlabeled_dummy/test"

# Image extensions to use from data paths
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Directory for logs to save
OUTPUT_DIR = "//home/adnan.khan/Desktop/SSDG_Baseline/misc/"
