import cv2
import sys
import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    video_file = os.path.join('resources', 'video', '9.MP4')

    video_name = os.path.split(video_file)[1].split('.')[0]

    with open(os.path.join('resources', f'tracking_{video_name}.json')) as json_file:
        annotation = json.load(json_file)

    video = cv2.VideoCapture(video_file)

    fps = video.get(cv2.CAP_PROP_FPS)

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    ovideo = cv2.VideoWriter(os.path.join('outputs', f'edited_{video_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not video.isOpened():
        print('Video not opened')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Couldn\'t read video')
        sys.exit()

    idx = 0

    object_colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}

    indices = [i for i, x in enumerate(annotation['frame_id']) if x == idx]

    for i in indices:
        # plt.scatter(annotation['x_px'][i], annotation['y_px'][i], c=object_colors[annotation['object_id'][i]])
        cv2.circle(frame, (int(annotation['x_px'][i]), int(annotation['y_px'][i])), 20, object_colors[annotation['object_id'][i]], -1)

    ovideo.write(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        idx += 1

        indices = [i for i, x in enumerate(annotation['frame_id']) if x == idx]

        for i in indices:
            # plt.scatter(annotation['x_px'][i], annotation['y_px'][i], c=object_colors[annotation['object_id'][i]])
            cv2.circle(frame, (int(annotation['x_px'][i]), int(annotation['y_px'][i])), 20,
                       object_colors[annotation['object_id'][i]], -1)

        ovideo.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
