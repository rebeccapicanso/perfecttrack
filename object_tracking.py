# takes the labeled .json and reorients to track objects in an individual scene.
import json
import os
from PIL import Image
from transformers import pipeline
from labeling_shots import label_scenes
from shots import get_shots

videofile = "tennis.mp4"
# test content, will need to alter the model to something with more defined tracking
# using microsoft's transformer detection instead of something smaller.
# https://huggingface.co/microsoft/table-transformer-detection/tree/main

# for testing purposes we're just tracking a tennis ball in a tennis video.
LABEL = "ball"

# run the code only ever X frames (5 by default)
SKIP = 5

# build out pipeline
MODEL = "microsoft/table-transformer-detection"

labeler = pipeline("object-detection", model=MODEL)

clip_set = []

# import all spliced clips from directory into a list
def setup(path):
    # set user input as path
    clip_set_path = path

    # create array of ungraded person
    for file in os.listdir(clip_set_path):
        # check if file contains mp4 and is not the source video, append to file

        if file.endswith(".mp4") and file != videofile:
            # set append path to file

            file = clip_set_path + "/" + file
            # append file to array
          
            clip_set.append(file)





