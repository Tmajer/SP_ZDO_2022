import cv2
import sys
import os
import json

import torchvision
from datasets import get_object_detection_model
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def predict(video_fn):
    num_classes = 4
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = get_object_detection_model(num_classes)
        model.load_state_dict(torch.load(os.path.join('resources', 'tuned_model.pt')))
        model.to(device).eval()
    else:
        device = torch.device('cpu')
        model = get_object_detection_model(num_classes)
        model.load_state_dict(torch.load(os.path.join('resources', 'tuned_model.pt'), map_location=device))

    video_with_suffix = os.path.split(video_fn)[1]
    video_name = os.path.split(video_fn)[1].split('.')[0]

    video = cv2.VideoCapture(video_fn)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    annotation = {
        'filename': [],
        'frame_id': [],
        'object_id': [],
        'x_px': [],
        'y_px': [],
        'annotation_timestamp': []
    }

    if not video.isOpened():
        print('Video not opened')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Couldn\'t read video')
        sys.exit()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.ToTensor()
    tensor = transform(frame)
    model.eval()
    with torch.no_grad():
        prediction = model([tensor.float().to(device)])[0]

    nms_prediction = apply_nms(prediction, iou_thresh=0.2)

    flags = [False for _ in range(3)]
    nhtracker = cv2.TrackerCSRT_create()
    twtracker = cv2.TrackerCSRT_create()
    sctracker = cv2.TrackerCSRT_create()

    labels = nms_prediction['labels'].tolist()
    if 1 in labels and not flags[0]:
        nhidx = labels.index(1)
        nhbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[nhidx]))
        nhtracker.init(frame, nhbox)
        flags[0] = True
        middle = (nhbox[0] + nhbox[2] / 2, nhbox[1] + nhbox[3] / 2)

        annotation['filename'].append(video_with_suffix)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(0)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)
    if 2 in labels and not flags[1]:
        twidx = labels.index(2)
        twbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[twidx]))
        twtracker.init(frame, twbox)
        flags[1] = True
        middle = (twbox[0] + twbox[2] / 2, twbox[1] + twbox[3] / 2)

        annotation['filename'].append(video_with_suffix)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(1)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)
    if 3 in labels and not flags[2]:
        scidx = labels.index(3)
        scbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[scidx]))
        sctracker.init(frame, scbox)
        flags[2] = True
        middle = (scbox[0] + scbox[2] / 2, scbox[1] + scbox[3] / 2)

        annotation['filename'].append(video_with_suffix)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(2)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)

    count = 0

    print(f'Processing frames: [>          ] ({count:05d}/{length:05d} done)\r', end="")

    while True:
        ok, frame = video.read()
        if not ok:
            break

        count += 1

        print_prog(count, length)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flags[0]:
            (nhsuccess, nhbox) = nhtracker.update(frame)

        if flags[1]:
            (twsuccess, twbox) = twtracker.update(frame)

        if flags[2]:
            (scsuccess, scbox) = sctracker.update(frame)

        transform = transforms.ToTensor()
        tensor = transform(frame)
        model.eval()
        with torch.no_grad():
            prediction = model([tensor.float().to(device)])[0]

        nms_prediction = apply_nms(prediction, iou_thresh=0.2)

        labels = nms_prediction['labels'].tolist()

        if 1 in labels:
            nhidx = labels.index(1)
            nhbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[nhidx]))
            if flags[0]:
                # if not (nhbox[0] - nhbbox[0] < 100) or not (nhbox[1] - nhbbox[1] < 100):
                # nhtracker = cv2.TrackerCSRT_create()
                nhtracker.init(frame, nhbbox)
                nhbox = nhbbox
            else:
                nhtracker.init(frame, nhbbox)
                nhbox = nhbbox
                flags[0] = True
        elif flags[0]:
            if not nhsuccess:
                flags[0] = False
        if 2 in labels:
            twidx = labels.index(2)
            twbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[twidx]))
            if flags[1]:
                # if not (twbox[0] - twbbox[0] < 100) or not (twbox[1] - twbbox[1] < 100):
                # twtracker = cv2.TrackerCSRT_create()
                twtracker.init(frame, twbbox)
                twbox = twbbox
            else:
                twtracker.init(frame, twbbox)
                twbox = twbbox
                flags[1] = True
        elif flags[1]:
            if not twsuccess:
                flags[1] = False
        if 3 in labels:
            scidx = labels.index(3)
            scbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[scidx]))
            if flags[2]:
                # if not (scbox[0] - scbbox[0] < 100) or not (scbox[1] - scbbox[1] < 100):
                # sctracker = cv2.TrackerCSRT_create()
                sctracker.init(frame, scbbox)
                scbox = scbbox
            else:
                sctracker.init(frame, scbbox)
                scbox = scbbox
                flags[2] = True
        elif flags[2]:
            if not scsuccess:
                flags[2] = False

        if flags[0]:
            middle = (nhbox[0] + nhbox[2] / 2, nhbox[1] + nhbox[3] / 2)

            annotation['filename'].append(video_with_suffix)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(0)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

        if flags[1]:
            middle = (twbox[0] + twbox[2] / 2, twbox[1] + twbox[3] / 2)

            annotation['filename'].append(video_with_suffix)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(1)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

        if flags[2]:
            middle = (scbox[0] + scbox[2] / 2, scbox[1] + scbox[3] / 2)

            annotation['filename'].append(video_with_suffix)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(2)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

    print(f'Processing frames: [==========>] ({length:05d}/{length:05d} done)')

    with open(os.path.join('resources', f'tracking_{video_name}.json'), 'w') as fp:
        json.dump(annotation, fp)

    return annotation


def print_prog(count, length):
    if count < int(length / 10):
        print(f'Processing frames: [>          ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 9*int(length / 10):
        print(f'Processing frames: [=========> ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 8*int(length / 10):
        print(f'Processing frames: [========>  ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 7*int(length / 10):
        print(f'Processing frames: [=======>   ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 6*int(length / 10):
        print(f'Processing frames: [======>    ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 5*int(length / 10):
        print(f'Processing frames: [=====>     ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 4*int(length / 10):
        print(f'Processing frames: [====>      ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 3*int(length / 10):
        print(f'Processing frames: [===>       ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > 2*int(length / 10):
        print(f'Processing frames: [==>        ] ({count:05d}/{length:05d} done)\r', end="")
    elif count > int(length / 10):
        print(f'Processing frames: [=>         ] ({count:05d}/{length:05d} done)\r', end="")


def pascal_voc_to_coco(box):
    return [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        box = box.tolist()
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


def apply_nms(orig_prediction, iou_thresh=0.2):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    new_prediction = orig_prediction
    new_prediction['boxes'] = new_prediction['boxes'][keep]
    new_prediction['scores'] = new_prediction['scores'][keep]
    new_prediction['labels'] = new_prediction['labels'][keep]

    best_scissors = (None, 0)
    best_tweezers = (None, 0)
    best_needle_holder = (None, 0)
    labels = new_prediction['labels'].tolist()
    scores = new_prediction['scores'].tolist()

    for index, label in enumerate(labels):
        if label == 3:
            if best_scissors[1] < scores[index]:
                best_scissors = (index, scores[index])
        if label == 2:
            if best_tweezers[1] < scores[index]:
                best_tweezers = (index, scores[index])
        if label == 1:
            if best_needle_holder[1] < scores[index]:
                best_needle_holder = (index, scores[index])

    keep = []
    if best_scissors[0] is not None:
        keep.append(best_scissors[0])
    if best_needle_holder[0] is not None:
        keep.append(best_needle_holder[0])
    if best_tweezers[0] is not None:
        keep.append(best_tweezers[0])

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


if __name__ == '__main__':
    num_classes = 4
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = get_object_detection_model(num_classes)
        model.load_state_dict(torch.load(os.path.join('resources', 'tuned_model.pt')))
        model.to(device).eval()
    else:
        device = torch.device('cpu')
        model = get_object_detection_model(num_classes)
        model.load_state_dict(torch.load(os.path.join('resources', 'tuned_model.pt'), map_location=device))

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[-1]

    video_file = os.path.join('resources', 'video', '9.MP4')

    video_name = os.path.split(video_file)[1].split('.')[0]

    video = cv2.VideoCapture(video_file)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    annotation = {
        'filename': [],
        'frame_id': [],
        'object_id': [],
        'x_px': [],
        'y_px': [],
        'annotation_timestamp': []
    }

    if not video.isOpened():
        print('Video not opened')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Couldn\'t read video')
        sys.exit()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.ToTensor()
    tensor = transform(frame)
    model.eval()
    with torch.no_grad():
        prediction = model([tensor.float().to(device)])[0]

    nms_prediction = apply_nms(prediction, iou_thresh=0.2)

    flags = [False for _ in range(3)]
    nhtracker = cv2.TrackerCSRT_create()
    twtracker = cv2.TrackerCSRT_create()
    sctracker = cv2.TrackerCSRT_create()

    labels = nms_prediction['labels'].tolist()
    if 1 in labels and not flags[0]:
        nhidx = labels.index(1)
        nhbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[nhidx]))
        nhtracker.init(frame, nhbox)
        flags[0] = True
        middle = (nhbox[0] + nhbox[2] / 2, nhbox[1] + nhbox[3] / 2)

        annotation['filename'].append(video_file)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(0)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)
    if 2 in labels and not flags[1]:
        twidx = labels.index(2)
        twbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[twidx]))
        twtracker.init(frame, twbox)
        flags[1] = True
        middle = (twbox[0] + twbox[2] / 2, twbox[1] + twbox[3] / 2)

        annotation['filename'].append(video_file)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(1)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)
    if 3 in labels and not flags[2]:
        scidx = labels.index(3)
        scbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[scidx]))
        sctracker.init(frame, scbox)
        flags[2] = True
        middle = (scbox[0] + scbox[2] / 2, scbox[1] + scbox[3] / 2)

        annotation['filename'].append(video_file)
        annotation['frame_id'].append(0)
        annotation['object_id'].append(2)
        annotation['x_px'].append(middle[0])
        annotation['y_px'].append(middle[1])
        annotation['annotation_timestamp'].append(0)

    count = 0

    print(f'Processing frames: [>          ] ({count:05d}/{length:05d} done)\r', end="")

    while True:
        ok, frame = video.read()
        if not ok:
            break

        count += 1

        print_prog(count, length)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flags[0]:
            (nhsuccess, nhbox) = nhtracker.update(frame)

        if flags[1]:
            (twsuccess, twbox) = twtracker.update(frame)

        if flags[2]:
            (scsuccess, scbox) = sctracker.update(frame)

        transform = transforms.ToTensor()
        tensor = transform(frame)
        model.eval()
        with torch.no_grad():
            prediction = model([tensor.float().to(device)])[0]

        nms_prediction = apply_nms(prediction, iou_thresh=0.2)

        labels = nms_prediction['labels'].tolist()

        if 1 in labels:
            nhidx = labels.index(1)
            nhbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[nhidx]))
            if flags[0]:
                # if not (nhbox[0] - nhbbox[0] < 100) or not (nhbox[1] - nhbbox[1] < 100):
                # nhtracker = cv2.TrackerCSRT_create()
                nhtracker.init(frame, nhbbox)
                nhbox = nhbbox
            else:
                nhtracker.init(frame, nhbbox)
                nhbox = nhbbox
                flags[0] = True
        elif flags[0]:
            if not nhsuccess:
                flags[0] = False
        if 2 in labels:
            twidx = labels.index(2)
            twbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[twidx]))
            if flags[1]:
                # if not (twbox[0] - twbbox[0] < 100) or not (twbox[1] - twbbox[1] < 100):
                # twtracker = cv2.TrackerCSRT_create()
                twtracker.init(frame, twbbox)
                twbox = twbbox
            else:
                twtracker.init(frame, twbbox)
                twbox = twbbox
                flags[1] = True
        elif flags[1]:
            if not twsuccess:
                flags[1] = False
        if 3 in labels:
            scidx = labels.index(3)
            scbbox = tuple(pascal_voc_to_coco(nms_prediction['boxes'].tolist()[scidx]))
            if flags[2]:
                # if not (scbox[0] - scbbox[0] < 100) or not (scbox[1] - scbbox[1] < 100):
                # sctracker = cv2.TrackerCSRT_create()
                sctracker.init(frame, scbbox)
                scbox = scbbox
            else:
                sctracker.init(frame, scbbox)
                scbox = scbbox
                flags[2] = True
        elif flags[2]:
            if not scsuccess:
                flags[2] = False

        if flags[0]:
            middle = (nhbox[0] + nhbox[2] / 2, nhbox[1] + nhbox[3] / 2)

            annotation['filename'].append(video_file)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(0)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

        if flags[1]:
            middle = (twbox[0] + twbox[2] / 2, twbox[1] + twbox[3] / 2)

            annotation['filename'].append(video_file)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(1)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

        if flags[2]:
            middle = (scbox[0] + scbox[2] / 2, scbox[1] + scbox[3] / 2)

            annotation['filename'].append(video_file)
            annotation['frame_id'].append(count)
            annotation['object_id'].append(2)
            annotation['x_px'].append(middle[0])
            annotation['y_px'].append(middle[1])
            annotation['annotation_timestamp'].append(0)

    print(f'Processing frames: [==========>] ({length:05d}/{length:05d} done)')

    with open(os.path.join('resources', f'tracking_{video_name}.json'), 'w') as fp:
        json.dump(annotation, fp)
