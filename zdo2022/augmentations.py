import albumentations as A
import pickle

import cv2
import matplotlib.pyplot as plt
import skimage.io
import os
import numpy as np

from PIL import Image
import PIL


def transform_image(transform, folder, frame, transformed_annots):
    print(folder, frame)

    one = annotation[f'{folder}_{frame:06d}']

    bboxes = []
    keypoints = []
    if 'needle holder' in one['object']:
        nh_index = one['object'].index('needle holder')
        bboxes.append([one["bb_xmin"][nh_index], one["bb_ymin"][nh_index],
                       one['bb_width'][nh_index], one['bb_height'][nh_index], 'needle holder'])
        keypoints.append([float(one["x_px"][nh_index]), float(one["y_px"][nh_index])])
    if 'scissors' in one['object']:
        s_index = one['object'].index('scissors')
        bboxes.append([one["bb_xmin"][s_index], one["bb_ymin"][s_index],
                       one['bb_width'][s_index], one['bb_height'][s_index], 'scissors'])
        keypoints.append([float(one["x_px"][s_index]), float(one["y_px"][s_index])])
    if 'tweezers' in one['object']:
        t_index = one['object'].index('tweezers')
        bboxes.append([one["bb_xmin"][t_index], one["bb_ymin"][t_index],
                       one['bb_width'][t_index], one['bb_height'][t_index], 'tweezers'])
        keypoints.append([float(one["x_px"][t_index]), float(one["y_px"][t_index])])

    image = skimage.io.imread(
        os.path.join('resources', str(folder), 'images', f'frame_{int(one["frame_id"][0]):06d}.PNG'))

    transformed = transform(image=image, bboxes=bboxes, keypoints=keypoints)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    x_min, y_min, width, height, _ = transformed_bboxes[0]

    transformed_fn = f'tframe_{folder}_{frame_number:06d}.PNG'
    t_annot = transformed_annots.setdefault(transformed_fn, {})

    for tool in transformed_bboxes:
        if tool[-1] == 'needle holder':
            t_annot.setdefault('nhbbox', tool)
        if tool[-1] == 'scissors':
            t_annot.setdefault('sbbox', tool)
        if tool[-1] == 'tweezers':
            t_annot.setdefault('tbbox', tool)

    pil_image = Image.fromarray(transformed_image)
    pil_image.save(os.path.join('resources', 'transformations', transformed_fn))

    # skimage.io.imsave(os.path.join('resources', 'transformations', transformed_fn), transformed_image)


if __name__ == '__main__':
    with open(os.path.join('resources', 'annotation.pickle'), 'rb') as handle:
        annotation = pickle.load(handle)

    transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.7),
            A.Rotate(limit=5, interpolation=2, p=0.6)
        ], p=1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=20, p=0.55),
            A.RGBShift(r_shift_limit=35, g_shift_limit=35, b_shift_limit=35, p=0.55)
        ], p=1),
        A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.0), p=0.5),
        A.RandomShadow(shadow_roi=(0.1, 0.1, 0.9, 0.9), shadow_dimension=7, p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=0.6),
            A.GaussianBlur(blur_limit=(3, 9), p=0.6),
        ], p=1)
    ], bbox_params=A.BboxParams(format='coco'), keypoint_params=A.KeypointParams(format='xy'))

    # folder, frame_number = (202, 115)
    # folder, frame_number = (202, 818)
    #
    # one = annotation[f'{folder}_{frame_number:06d}']
    #
    # bboxes = []
    # keypoints = []
    # if 'needle holder' in one['object']:
    #     nh_index = one['object'].index('needle holder')
    #     bboxes.append([one["bb_xmin"][nh_index], one["bb_ymin"][nh_index],
    #                    one['bb_width'][nh_index], one['bb_height'][nh_index], 'needle holder'])
    #     keypoints.append([float(one["x_px"][nh_index]), float(one["y_px"][nh_index])])
    # if 'scissors' in one['object']:
    #     s_index = one['object'].index('scissors')
    #     bboxes.append([one["bb_xmin"][s_index], one["bb_ymin"][s_index],
    #                    one['bb_width'][s_index], one['bb_height'][s_index], 'scissors'])
    #     keypoints.append([float(one["x_px"][s_index]), float(one["y_px"][s_index])])
    # if 'tweezers' in one['object']:
    #     t_index = one['object'].index('tweezers')
    #     bboxes.append([one["bb_xmin"][t_index], one["bb_ymin"][t_index],
    #                    one['bb_width'][t_index], one['bb_height'][t_index], 'tweezers'])
    #     keypoints.append([float(one["x_px"][t_index]), float(one["y_px"][t_index])])
    #
    # image = skimage.io.imread(os.path.join('resources', str(folder), 'images', f'frame_{int(one["frame_id"][0]):06d}.PNG'))
    #
    # x_min, y_min, _, _, _ = bboxes[0]
    #
    # image_copy = np.copy(image)
    #
    # for bb in bboxes:
    #     start_point = (int(bb[0]), int(bb[1]))
    #     end_point = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
    #     color = (0, 255, 0)
    #     thickness = 3
    #     image_copy = cv2.rectangle(image_copy, start_point, end_point, color, thickness)
    #
    # plt.imshow(image_copy)
    # for kp in keypoints:
    #     plt.scatter(float(kp[0]), float(kp[1]))
    # plt.show()
    #
    # transformed = transform(image=image, bboxes=bboxes, keypoints=keypoints)
    # transformed_image = transformed['image']
    # transformed_bboxes = transformed['bboxes']
    # transformed_keypoints = transformed['keypoints']
    #
    # x_min, y_min, width, height, _ = transformed_bboxes[0]
    #
    # transformed_image_copy = np.copy(transformed_image)
    # for kp in transformed_keypoints:
    #     plt.scatter(kp[0], kp[1])
    # for bb in transformed_bboxes:
    #     start_point = (int(bb[0]), int(bb[1]))
    #     end_point = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
    #     color = (0, 255, 0)
    #     thickness = 3
    #     transformed_image_copy = cv2.rectangle(transformed_image_copy, start_point, end_point, color, thickness)
    #
    # plt.imshow(transformed_image_copy)
    # plt.show()
    #
    # transformed_fn = f'tframe_{folder}_{frame_number:06d}.PNG'
    # transformed_annots = {}
    # t_annot = transformed_annots.setdefault(transformed_fn, {})
    #
    # for tool in transformed_bboxes:
    #     if tool[-1] == 'needle holder':
    #         t_annot.setdefault('nhbbox', tool)
    #     if tool[-1] == 'scissors':
    #         t_annot.setdefault('sbbox', tool)
    #     if tool[-1] == 'tweezers':
    #         t_annot.setdefault('tbbox', tool)

    transformed_annots = {}

    for key in annotation.keys():
        folder, frame_number = key.split('_')
        folder = int(folder)
        frame_number = int(frame_number)
        if folder == 225:
            transform_image(transform, folder, frame_number, transformed_annots)

    with open(os.path.join('resources', 'transformed_annotation_225.pickle'), 'wb') as handle:
        pickle.dump(transformed_annots, handle, protocol=pickle.HIGHEST_PROTOCOL)
