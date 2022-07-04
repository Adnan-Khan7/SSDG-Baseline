from constants import *
from torchvision import transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 左右翻转

    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder(DATA_TRAIN_SET, loader=lambda x: Image.open(x), extensions=IMG_EXTENSIONS, transform=train_tfm)
valid_set = DatasetFolder(DATA_VALID_SET, loader=lambda x: Image.open(x), extensions=IMG_EXTENSIONS, transform=test_tfm)
unlabeled_set = DatasetFolder(DATA_UNLABELED_SET, loader=lambda x: Image.open(x), extensions=IMG_EXTENSIONS, transform=train_tfm)
test_set = DatasetFolder(DATA_TEST_SET, loader=lambda x: Image.open(x), extensions=IMG_EXTENSIONS, transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

