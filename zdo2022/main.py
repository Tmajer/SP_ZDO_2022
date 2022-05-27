from tracking import predict
import os
import json


class InstrumentTracker():
    def __init__(self):
        pass

    def predict(self, video_filename):
        video_name = os.path.split(video_filename)[1].split('.')[0]
        if os.path.exists(os.path.join('resources', f'tracking_{video_name}.json')):
            with open(os.path.join('resources', f'tracking_{video_name}.json')) as json_file:
                annotation = json.load(json_file)
        else:
            annotation = predict(video_filename)

        return annotation
