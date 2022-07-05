import torch
import torch.nn as nn
from constants import *
from torch.utils.data import DataLoader, Dataset
import logging
# This is for the progress bar.
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
    dataset2 = noLabeledDataset(imgList, labelList)
    total_p_labels = len(dataset2)
    #####################################################################################
    # Validation of generated psuedo labels
    # model.eval()
    valid_pseudo_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                     pin_memory=False)
    # # These are used to record information in validation.
    valid_loss = []
    valid_corr = []
    predictions_valid = []
    criterion = nn.CrossEntropyLoss()
    #
    print("Validating the generated pseudo labels...")
    logging.info("Validating the generated pseudo labels...")
    # # Iterate the validation set by batches.
    for batch in tqdm(valid_pseudo_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        n_corr = sum((logits.argmax(dim=-1) == labels.to(device)))
        valid_corr.append(n_corr)
        # Take the class with greatest logit as prediction and record it.
        predictions_valid.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        # Record the loss and accuracy.
        valid_loss.append(loss.item())

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_corr) / len(predictions_valid)
    # Print the information.
    print(f"[ Pseudo label eval | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    logging.info(f"[ Pseudo label eval | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    total_correct = len(valid_corr)
    print(f"Number of correct predications on unlabeled dataset: {total_correct}")
    logging.info(f"Number of correct predications on unlabeled dataset: {total_correct}")
    print(f"Number of total generated pseudo labels: {total_p_labels}")
    logging.info(f"Number of total generated pseudo labels: {total_p_labels}")
    print(f"Psuedo Labels accuracy: {total_correct} / {total_p_labels} =  {total_correct / total_p_labels}")
    logging.info(f"Psuedo Labels accuracy: {total_correct} / {total_p_labels} =  {total_correct / total_p_labels}")

    ######################################################################################

    del imgList
    del labelList
    del data_loader
    # # Turn off the eval mode.
    model.train()
    return dataset2
