from datasets import *
from helpers import *
import logging
import torch
import torch.nn as nn
from constants import *
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(filename=OUTPUT_DIR + 'logs.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
print(
    f"Configuration: \n model:{MODEL_NAME}, SSL Threshold: {THRESH}, Learning Rate: {LR}, Batch Size: {BATCH_SIZE}, Epochs: {TRAIN_EPOCHS}")
logging.info(
    f"Configuration: \n model:{MODEL_NAME}, SSL Threshold: {THRESH}, Learning Rate: {LR}, Batch Size: {BATCH_SIZE}, Epochs: {TRAIN_EPOCHS}")


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

# Initialize the model for this run
model_ft, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, use_pretrained=True)
print(model_ft)
model_ft.apply(deactivate_batchnorm)
print(model_ft)

# Print the model we just instantiated
# print(model_ft)
def save_models(epochs, model):
    torch.save(model.state_dict(), OUTPUT_DIR + "custom_model{}.model".format(epochs))
    print("Checkpoint Saved")


def train_supervised(train_loader_labeled):
    model.train()
    train_loss = []
    train_accs = []
    # Iterate the training set by batches.
    for batch in tqdm(train_loader_labeled):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    logging.info(f"[ Train | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    return train_loss, train_acc


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = model_ft.to(device)

model.device = device

best_model = model
best_acc = 0
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# Whether to do semi-supervised learning.
do_semi = DO_SEMI
print("Starting training ")
logging.info("Starting training ")
loss_values = []
acc_values = []
new_labels = []
new_correct = []
for epoch in range(TRAIN_EPOCHS):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then combine the labeled dataset and pseudo-labeled dataset for the training.

    if epoch < SUPERVISED_EPOCHS:
        train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          pin_memory=False)
        train_loss, acc = train_supervised(train_loader_labeled)
        loss_values.append(train_loss)
        acc_values.append(acc)
    elif do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set, total_p_labels, total_correct = get_pseudo_labels(unlabeled_set, model, THRESH)

        if len(pseudo_set) == 0:
            print(f"No new pseudo labels generated at epoch {epoch + 1}..., \n Continue Supervised Training with "
                  f"labeled dataset")
            logging.info(f"No new pseudo labels generated at epoch {epoch + 1}..., \n Continue Supervised Training "
                         f"with labeled dataset")
            train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                              pin_memory=False)
            train_loss, acc = train_supervised(train_loader_labeled)
            loss_values.append(train_loss)
            acc_values.append(acc)
        else:
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            total_train = len(concat_dataset)
            print(f"Number of total training examples are: {total_train}")
            logging.info(f"Number of total training examples are: {total_train}")
            train_loader_labeled = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                              pin_memory=False)
            train_loss, acc = train_supervised(train_loader_labeled)
            loss_values.append(train_loss)
            acc_values.append(acc)
            new_labels.append(total_p_labels)
            new_correct.append(total_correct)
            del concat_dataset
            del pseudo_set
    else:
        train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          pin_memory=False)
        train_loss, acc = train_supervised(train_loader_labeled)
        loss_values.append(train_loss)
        acc_values.append(acc)



# Plot Training Loss
plt.rcParams["figure.figsize"] = [12.50, 7.50]
plt.plot(range(1, TRAIN_EPOCHS + 1), loss_values)
plt.plot(range(1, TRAIN_EPOCHS + 1), acc_values)
plt.xticks(range(1, TRAIN_EPOCHS + 1))
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs", labelpad=20)
plt.ylabel("Training Loss and Accuracy")
plt.legend(['Loss', 'Accuracy'], loc='upper right')
plt.savefig(OUTPUT_DIR + '/loss_accuracy.png')
plt.close()
if len(new_labels) and len(new_correct) > 0:
    # Plot labels
    plt.plot(range(1, len(new_labels)+1), new_labels)
    plt.plot(range(1, len(new_correct)+1), new_correct)
    plt.xticks(range(1, len(new_labels)+1))
    plt.title("Number of new/correct pseudo labels")
    plt.xlabel("Count of Epochs at which new labels are generated", labelpad=7)
    plt.ylabel("Labels count")
    plt.legend(['New Labels Generated', 'Correct Labels'], loc='upper left')
    plt.savefig(OUTPUT_DIR + '/pseudo_labels.png')
    plt.close()
# Testing the model
print("Starting test")
logging.info(" Starting test")
model = best_model
model.eval()

# Initialize a list to store the predictions.
predictions_test = []
# These are used to record information in validation.
batch_corr = []
test_loss = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    imgs, labels = batch

    # We don't need gradient in testing
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # We can still compute the loss (but not the gradient).
    loss = criterion(logits, labels.to(device))
    # Compute the accuracy for current batch.
    n_corr = sum((logits.argmax(dim=-1) == labels.to(device)))
    batch_corr.append(n_corr)

    # Take the class with greatest logit as prediction and record it.
    predictions_test.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # Record the loss and accuracy.
    test_loss.append(loss.item())

# The average loss and accuracy for entire validation set is the average of the recorded values.
test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(batch_corr) / len(predictions_test)
# Print the information.
print(f"[Test] loss = {test_loss:.5f}, acc = {test_acc:.5f}")
logging.info(f" [Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")
