import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io import read_image
import numpy as np
import albumentations as A
from skimage.util import img_as_float


class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transforms=None):
        self.transforms = transforms
        self.annotations = annotations
        self.imgs = [annotation['image_path'] for annotation in annotations]
        self.classes = ['background', 'needle holder', 'tweezers', 'scissors']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        img_path = annot.get('image_path')
        image = torch.tensor(img_as_float(np.array(read_image(img_path))).astype(np.double))

        boxes = []
        labels = []

        for index, label in enumerate(self.classes[1:]):
            tool = annot[label]
            if tool:
                labels.append(index + 1)
                xmin = tool['xmin']
                xmax = xmin + tool['width']
                ymin = tool['ymin']
                ymax = ymin + tool['height']

                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        return image, target


def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
