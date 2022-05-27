import random
import cv2
import skimage
import pickle
from pathlib import Path
import os
import lxml
from lxml import etree
import pandas as pd


def load_folder(annotation, folder):
    pth = Path(os.path.join('resources', folder, 'annotations.xml'))
    tree = etree.parse(pth)
    updated = tree.xpath('//updated')[0].text  # date of last change in CVAT

    for track in tree.xpath('track'):
        for point in track.xpath('points'):
            bb_width = 200
            bb_height = 200

            frame_anot = annotation.setdefault(f'{folder}_{int(point.get("frame")):06d}', {})
            pts = point.get('points').split(',')
            x, y = pts
            frame_anot.setdefault('filename', []).append(pth)
            frame_anot.setdefault('object', []).append(track.get('label'))
            not_visible = point.get('occluded') == 1 or point.get('outside') == 1
            frame_anot.setdefault('not_visible', []).append(not_visible)

            if 3839.1 < float(x):
                frame_anot.setdefault('x_px', []).append(str(float(x) - 1))
            elif float(x) < 0.9:
                frame_anot.setdefault('x_px', []).append(str(float(x) + 1))
            else:
                frame_anot.setdefault('x_px', []).append(x)
            if float(x) - bb_width / 2 < 0:
                bb_width = 100 + (100 + float(x) - bb_width / 2)
                frame_anot.setdefault('bb_xmin', []).append(0)
                frame_anot.setdefault('bb_width', []).append(bb_width)
            elif 3839 <= float(x) + bb_width / 2:
                frame_anot.setdefault('bb_xmin', []).append(float(x) - bb_width / 2)
                bb_width = 200 - (100 + float(x) - 3838)
                frame_anot.setdefault('bb_width', []).append(bb_width)
            else:
                frame_anot.setdefault('bb_xmin', []).append(float(x) - bb_width / 2)
                frame_anot.setdefault('bb_width', []).append(bb_width)

            if 2159.1 < float(y):
                frame_anot.setdefault('y_px', []).append(str(float(y) - 1))
            elif float(y) < 0.9:
                frame_anot.setdefault('y_px', []).append(str(float(y) + 1))
            else:
                frame_anot.setdefault('y_px', []).append(y)
            if float(y) - bb_height / 2 < 0:
                bb_height = 100 + (100 + float(y) - bb_height / 2)
                frame_anot.setdefault('bb_ymin', []).append(0)
                frame_anot.setdefault('bb_height', []).append(bb_height)
            elif 2159 <= float(y) + bb_height / 2:
                frame_anot.setdefault('bb_ymin', []).append(float(y) - bb_height / 2)
                bb_height = 200 - (100 + float(y) - 2158)
                frame_anot.setdefault('bb_height', []).append(bb_height)
            else:
                frame_anot.setdefault('bb_ymin', []).append(float(y) - bb_height / 2)
                frame_anot.setdefault('bb_height', []).append(bb_height)
            frame_anot.setdefault('frame_id', []).append(point.get('frame'))
            frame_anot.setdefault('annotation_timestamp', []).append(updated)

    with open(os.path.join('resources', 'annotation.pickle'), 'wb') as handle:
        pickle.dump(annotation, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    annotations = {}

    folders = ['202', '204', '206', '220', '221', '224', '225']

    for folder in folders:
        load_folder(annotations, folder)

    with open(os.path.join('resources', 'annotation.pickle'), 'wb') as handle:
        pickle.dump(annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)
