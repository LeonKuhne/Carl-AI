from keras.models import load_model
import util
import numpy as np
import os
import argparse
import json

config = util.read_config()
DISPLAY = config['displays']
MODEL_NAME = "carl.model"
frame_count = 1000

# parse arguments
parser = argparse.ArgumentParser(description='"i am god" - carl')
parser.add_argument("-r", "--recent", help="Use the most recent model, named 'recent.model'", action="store_true")
parser.add_argument("-d", "--duration", help="number of frames until done")
args = parser.parse_args()
if args.recent:
    MODEL_NAME = "recent.model"
if args.duration:
    frame_count = int(args.duration)


model = load_model(MODEL_NAME)

currDisplayId = 0

for i in range(frame_count):
    # get some data
    frame = util.get_frame()
    x = np.asarray([frame])

    # reshape
    x = x.reshape([-1, 28, 28, 1])

    # predict
    prediction = int(model.predict_classes(x)[0])

    if(currDisplayId != prediction):
        currDisplayId = prediction
        display = DISPLAY[prediction]
        if(display != "none"):
            os.system(f"sway focus output {display}")
        else:
            print("ur lookin away")

