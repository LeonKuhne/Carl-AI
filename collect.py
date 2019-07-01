import util
import numpy as np
import pickle
from argparse import ArgumentParser
from keras.utils import to_categorical
from threading import Thread
import os.path

# config
config = util.read_config()
saveName = 'recent'

# args
parser = ArgumentParser(description="carl loves you")
parser.add_argument('-a', "--append", help="data file to append to")
args = parser.parse_args()

# globals
running = True
paused = False

def listen_to_input():
    global paused
    global running

    print("type 'quit' to exit and 'p' to toggle pause")
    while running:
        command = input("")
        if(command == "quit"):
            print("exiting...")
            running = False
        elif(command == "p"):
            paused = not paused
            if paused:
                print("paused")
            else:
                print("running")
        else:
            print(f"unknown command '{command}'")

def record_observation(start_frames, start_displays):
    # record the user while he uses the displays
    print("\trecording...")

    frames = start_frames.tolist()
    displays = start_displays.tolist()
    while running:
        # get the display
        if paused:
            displayId = len(config['displays']) - 1 # no display
        else:
            display = util.get_display()
            displayId = config['displays'].index(display)
        
        # get the frame
        frame = util.get_frame()

        frames.append(frame.tolist())
        displays.append(displayId)

    print("\tdone")
    return (np.asarray(frames), np.asarray(displays))

def collect_data(start_x, start_y):
    # collect data of the user looking at the monitor
    waiting = Thread(target=listen_to_input)
    waiting.daemon = True
    waiting.start()
    
    (x_train, y_train) = record_observation(start_x, start_y)

    # reshape
    #x_train = x_train.reshape([-1, 28, 28, 1])
   
    # turn the array into a matrix (one-hot encoding)
    y_train = to_categorical(y_train)

    return (x_train, y_train)

if __name__ == '__main__':
    # setup - disable printing extra details
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # import
    print("loading data..")
    start_x = np.empty(0)
    start_y = np.empty(0)
    if args.append:
        saveName = args.append
        if os.path.isfile(f"data/{saveName}.xdata"): # file exists
            (start_x, start_y) = util.get_data(saveName)
    print("\tdone")
    
    # collect
    (x, y) = collect_data(start_x, start_y)

    print('shapes')
    print(x.shape)
    print(y.shape)

    # save
    util.save_data(saveName, x, y)

