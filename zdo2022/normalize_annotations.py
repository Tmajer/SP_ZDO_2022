import os
import pickle


if __name__ == '__main__':
    new_annotations = []

    with open(os.path.join('resources', 'annotation.pickle'), 'rb') as handle:
        annotation = pickle.load(handle)

    for key in annotation.keys():
        annot = annotation[key]
        folder, frame_number = key.split('_')
        image_path = os.path.join('resources', folder, 'images', f'frame_{frame_number}.PNG')
        tweezers = {}
        needle_holder = {}
        scissors = {}
        for id, object in enumerate(annot['object']):
            if object == 'needle holder':
                needle_holder = {'xmin': annot['bb_xmin'][id], 'ymin': annot['bb_ymin'][id],
                                 'width': annot['bb_width'][id], 'height': annot['bb_height'][id],
                            'occluded': annot['not_visible'][id]}
            elif object == 'scissors':
                scissors = {'xmin': annot['bb_xmin'][id], 'ymin': annot['bb_ymin'][id],
                            'width': annot['bb_width'][id], 'height': annot['bb_height'][id],
                            'occluded': annot['not_visible'][id]}
            elif object == 'tweezers':
                tweezers = {'xmin': annot['bb_xmin'][id], 'ymin': annot['bb_ymin'][id],
                            'width': annot['bb_width'][id], 'height': annot['bb_height'][id],
                            'occluded': annot['not_visible'][id]}
        new_annot = {'image_path': image_path, 'needle holder': needle_holder,
                     'scissors': scissors, 'tweezers': tweezers}
        new_annotations.append(new_annot)

    transformed_annotations_filenames = [os.path.join('resources', f'transformed_annotation_{num}.pickle')
                                         for num in [202, 204, 220, 225]]

    for fn in transformed_annotations_filenames:
        with open(fn, 'rb') as handle:
            annotation = pickle.load(handle)

        for key in annotation.keys():
            annot = annotation[key]
            image_path = os.path.join('resources', 'transformations', key)
            tweezers = {}
            needle_holder = {}
            scissors = {}
            for key in annot:
                tool_bbox = annot[key]
                if key == 'nhbbox':
                    needle_holder = {'xmin': tool_bbox[0], 'ymin': tool_bbox[1],
                                     'width': tool_bbox[2], 'height': tool_bbox[3],
                                     'occluded': False}
                elif key == 'sbbox':
                    scissors = {'xmin': tool_bbox[0], 'ymin': tool_bbox[1],
                                     'width': tool_bbox[2], 'height': tool_bbox[3],
                                     'occluded': False}
                elif key == 'tbbox':
                    tweezers = {'xmin': tool_bbox[0], 'ymin': tool_bbox[1],
                                     'width': tool_bbox[2], 'height': tool_bbox[3],
                                     'occluded': False}
            new_annot = {'image_path': image_path, 'needle holder': needle_holder,
                         'scissors': scissors, 'tweezers': tweezers}
            new_annotations.append(new_annot)

    a = 5

    with open(os.path.join('resources', 'normalized_annotations.pickle'), 'wb') as handle:
        pickle.dump(new_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)
