import torch
import torch.nn as nn
from constants import *
from torch.utils.data import DataLoader, Dataset
# This is for the progress bar.
from tqdm.auto import tqdm

class noLabeledDataset(Dataset):
    def __init__(self, imgList, labelList):
        n = len(imgList)
        x = torch.cat(([imgList[i] for i in range(n)]), 0)
        del imgList
        y = [label for labels in labelList for label in labels]
        del labelList
        self.len = x.shape[0]
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def get_pseudo_labels(dataset, model, threshold):
    # This functions generates pseudo-labels of a dataset using given model.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    imgList = []
    labelList = []
    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        logits = softmax(logits)

        # Filter the data and construct a new dataset.
        score_list, class_list = logits.max(dim=-1)
        score_list, class_list = score_list.cpu().numpy(), class_list.cpu().numpy()
        # If score is greater than threshold then only assign psuedo label
        score_filter = score_list > threshold
        score_list, class_list = score_list[score_filter], class_list[score_filter]

        imgList.append(img[score_filter])
        labelList.append(class_list)
    dataset = noLabeledDataset(imgList, labelList)
    del imgList
    del labelList
    del data_loader
    # # Turn off the eval mode.
    model.train()
    return dataset