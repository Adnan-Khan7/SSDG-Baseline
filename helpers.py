import torch
import torch.nn as nn
from constants import *
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm.auto import tqdm

logging.basicConfig(filename=OUTPUT_DIR + 'logs.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


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
    print(f"Generating pseudo labels for next epoch...")
    logging.info(f"Generating pseudo labels for next epoch...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    imgList = []
    labelList = []
    total_correct = 0
    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, labels = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        logits = softmax(logits)

        # Filter the data and construct a new dataset.
        score_list, class_list = logits.max(dim=-1)
        score_list, class_list = score_list.cpu().numpy(), class_list.cpu().numpy()
        # If score is greater than threshold then only assign pseudo label
        score_filter = score_list > threshold
        score_list, class_list = score_list[score_filter], class_list[score_filter]
        # Filters also the actual labels for evaluating the pseudo labels
        labels = labels[score_filter]
        labels = labels.numpy()
        n_corr = sum(class_list == labels)
        total_correct = total_correct + n_corr

        imgList.append(img[score_filter])
        labelList.append(class_list)
    dataset = noLabeledDataset(imgList, labelList)
    total_p_labels = len(dataset)
    print(f"Number of total generated pseudo labels: {total_p_labels}")
    logging.info(f"Number of total generated pseudo labels: {total_p_labels}")
    print(f"Number of correct pseudo labels: {total_correct}")
    logging.info(f"Number of correct pseudo labels: {total_correct}")
    if total_p_labels > 0:
        print(f"Pseudo Labels Accuracy: {total_correct} / {total_p_labels} = {total_correct / total_p_labels} ")
        logging.info(f"Pseudo Labels Accuracy: {total_correct} / {total_p_labels} = {total_correct / total_p_labels} ")
    del imgList
    del labelList
    del data_loader
    # # Turn off the eval mode.
    model.train()
    return dataset, total_p_labels, total_correct
