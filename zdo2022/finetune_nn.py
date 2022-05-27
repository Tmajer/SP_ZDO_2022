from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from datasets import SurgicalToolDataset
from datasets import get_object_detection_model
from datasets import get_transform
import torch
import os
import torch.nn as nn
import pickle
import utils
from engine import train_one_epoch, evaluate


if __name__ == '__main__':
    with open(os.path.join('resources', 'normalized_annotations.pickle'), 'rb') as handle:
        annotation = pickle.load(handle)

    # use our dataset and defined transformations
    torch.manual_seed(1)
    indices = []
    indices.extend([i + 25 for i in torch.randperm(425).tolist()][0:100])
    indices.extend([i + 2000 for i in torch.randperm(168).tolist()][0:50])
    indices.extend([i + 2288 for i in torch.randperm(500).tolist()][0:150])
    indices.extend([i + 4333 for i in torch.randperm(150).tolist()][0:50])
    indices.extend([i + 4685 for i in torch.randperm(500).tolist()][0:120])
    indices.extend([i + 6331 for i in torch.randperm(500).tolist()][0:150])
    indices.extend([i + 7468 for i in torch.randperm(600).tolist()][0:200])
    indices.extend([i + 8795 for i in torch.randperm(200).tolist()][0:100])
    indices.extend([i + 12734 for i in torch.randperm(950).tolist()][0:200])

    annotation = [annotation[idx] for idx in indices]
    indices = torch.randperm(len(annotation)).tolist()

    print('Building datasets')
    dataset = SurgicalToolDataset(annotation, transforms=get_transform(train=True))
    dataset_test = SurgicalToolDataset(annotation, transforms=get_transform(train=False))

    # train test split
    test_split = 0.2
    # tsize = int(len(dataset) * test_split)
    tsize = int(len(indices) * test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

    print('Loading datasets')
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=5, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    num_classes = 4

    print('Loading pretrained model')
    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # training for 10 epochs
    num_epochs = 10

    print('Starting training')
    for epoch in range(num_epochs):
        print(f'Starting epoch number {epoch}')
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print(f'Starting evaluation of epoch {epoch}')
        evaluate(model, data_loader_test, device=device)

    print('Saving model')
    torch.save(model.state_dict(), os.path.join('resources', 'tuned_model.pt'))
    print('Saving done')
