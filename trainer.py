from datasets import *
from helpers import *
import logging
import torch
import torch.nn as nn
from constants import *
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
logging.basicConfig(filename=OUTPUT_DIR + 'logs.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
print(f"Configuration: \n model:{MODEL_NAME}, SSL Threshold: {THRESH}, Learning Rate: {LR}, Batch Size: {BATCH_SIZE}, Epochs: {TRAIN_EPOCHS}")
logging.info(f"Configuration: \n model:{MODEL_NAME}, SSL Threshold: {THRESH}, Learning Rate: {LR}, Batch Size: {BATCH_SIZE}, Epochs: {TRAIN_EPOCHS}")


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

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(MODEL_NAME, NUM_CLASSES, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)


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

    return


# def train_unsupervised(train_loader_labeled, train_loader_unlabeled):
#     # ---------- Training ----------
#     # Make sure the model is in train mode before training.
#     model.train()
#
#     # These are used to record information in training.
#     train_loss, train_total = [], []
#     train_correct = []
#
#     # Iterate the training set by batches.
#     for batch1, batch2 in zip(train_loader_labeled, train_loader_unlabeled):
#         # A batch consists of image data and corresponding labels.
#         imgs1, labels1 = batch1
#         imgs2, labels2 = batch2
#
#         # Gradients stored in the parameters in the previous step should be cleared out first.
#         optimizer.zero_grad()
#         # Forward the data. (Make sure data and model are on the same device.)
#         logits1 = model(imgs1.to(device))
#         logits2 = model(imgs2.to(device))
#
#         # Calculate the cross-entropy loss.
#         # We don't need to apply softmax before computing cross-entropy as it is done automatically.
#         loss1 = criterion(logits1, labels1.to(device))
#         loss2 = criterion(logits2, labels2.to(device))
#
#         loss = loss1 + loss2
#
#         # Compute the gradients for parameters.
#         loss.backward()
#
#         # Clip the gradient norms for stable training.
#         grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
#
#         # Update the parameters with computed gradients.
#         optimizer.step()
#
#         # Compute the accuracy for current batch.
#         correct1 = sum((logits1.argmax(dim=-1) == labels1.to(device)))
#         correct2 = sum((logits2.argmax(dim=-1) == labels2.to(device)))
#         total_correct = correct1 + correct2
#
#         # Record the loss and accuracy.
#         train_loss.append(loss.item())
#         train_correct.append(total_correct)
#
#         train_total.append(len(labels1) + len(labels2))
#
#     # The average loss and accuracy of the training set is the average of the recorded values.
#     train_loss = sum(train_loss) / len(train_loss)
#     train_acc = sum(train_correct) / len(train_total)
#
#     # Print the information.
#     print(f"[ Train | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, supervised "
#           f"training epochs = {SUPERVISED_EPOCHS}")
#     logging.info(f"[ Train | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, "
#                  f"supervised training epochs = {SUPERVISED_EPOCHS}")
#     return


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

for epoch in range(TRAIN_EPOCHS):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then combine the labeled dataset and pseudo-labeled dataset for the training.
    if epoch < SUPERVISED_EPOCHS:
        train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          pin_memory=False)
        train_supervised(train_loader_labeled)
    elif do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model, THRESH)
        if len(pseudo_set) == 0:
            print(f"No new pseudo labels generated at epoch {epoch + 1}..., \n Continue Supervised Training with "
                  f"labeled dataset")
            logging.info(f"No new pseudo labels generated at epoch {epoch + 1}..., \n Continue Supervised Training "
                         f"with labeled dataset")
            train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                              pin_memory=False)
            train_supervised(train_loader_labeled)
        else:
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            total_train = len(concat_dataset)
            print(f"Number of total training examples are: {total_train}")
            logging.info(f"Number of total training examples are: {total_train}")
            train_loader_labeled = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
            train_supervised(train_loader_labeled)
            del concat_dataset
            del pseudo_set
    else:
        train_loader_labeled = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          pin_memory=False)
        train_supervised(train_loader_labeled)

    # # ---------- Validation ----------
    # # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    # model.eval()
    #
    # # These are used to record information in validation.
    # valid_loss = []
    # valid_corr = []
    # predictions_valid = []
    #
    # # Iterate the validation set by batches.
    # for batch in tqdm(valid_loader):
    #     # A batch consists of image data and corresponding labels.
    #     imgs, labels = batch
    #
    #     # We don't need gradient in validation.
    #     # Using torch.no_grad() accelerates the forward process.
    #     with torch.no_grad():
    #         logits = model(imgs.to(device))
    #
    #     # We can still compute the loss (but not the gradient).
    #     loss = criterion(logits, labels.to(device))
    #
    #     # Compute the accuracy for current batch.
    #     n_corr = sum((logits.argmax(dim=-1) == labels.to(device)))
    #     valid_corr.append(n_corr)
    #     # Take the class with greatest logit as prediction and record it.
    #     predictions_valid.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    #
    #     # Record the loss and accuracy.
    #     valid_loss.append(loss.item())
    #
    # # The average loss and accuracy for entire validation set is the average of the recorded values.
    # valid_loss = sum(valid_loss) / len(valid_loss)
    # valid_acc = sum(valid_corr) / len(predictions_valid)
    #
    # if valid_acc > best_acc:
    #     best_model = model
    #     best_acc = valid_acc
    #
    # if epoch == SAVE_EPOCH:
    #     save_models(epoch, best_model)
    #
    # print("best_acc so far: ", best_acc)
    # logging.info(f"best_acc so far: {best_acc}")
    # # Print the information.
    # print(f"[ Valid | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    # logging.info(f"[ Valid | {epoch + 1:03d}/{TRAIN_EPOCHS:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

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
