import util
import time
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, MaxPooling2D, Flatten, LSTM, Embedding
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

# args
parser = ArgumentParser(description="carl promises to be good boy")
parser.add_argument('-f', '--file', help="file to save to, otherwise uses recent")
parser.add_argument('-l', '--log', help="log comment, default is timestamp")
args = parser.parse_args()

# get name
name = "recent"
log_comment = int(time.time())
if args.file:
    name = args.file
if args.log:
    log_comment = args.log
(x, y) = util.get_data(name)

# config
config = util.read_config()
MODEL_NAME = f"carl-{config['type']}:{config['version']}-{name}-{config['epochs']}-{log_comment}"
tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}")

# get the data
print('Raw Data:')
print(x.shape)
print(y.shape)

# create validation and training set
#x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=7, test_size=0.2) #cnn
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False) #rnn

# TODO create multiple 'video' training sets

# print some neat details
print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# create the model
model = Sequential()
# filters is number of segments (of picture), kernal_size is the size of the filter
model.add(ConvLSTM2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(None, 28, 28, 1)))
model.add(BatchNormalization())
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Conv2D(37, (3, 3), activation='relu'))
#model.add(Conv2D(150, (3, 3), activation='relu'))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(LSTM(128))
#model.add(Dense(128, activation='relu'))
model.add(Dense(len(config['displays']), activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #cnn
# TODO
#model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy']) #rnn

# train
#cbk_early_stopping = EarlyStopping(monitor='val_acc', mode='max')
#model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[cbk_early_stopping])
model.fit(x_train, y_train, epochs=config['epochs'], validation_data=(x_val, y_val), callbacks=[tensorboard])

# save
model.save(f"models/{MODEL_NAME}.model")
model.save(f"recent.model")

isDone = False
while not isDone:
    makeDefault = input('update carl (y,n)? ')
    if(makeDefault == 'y'):
        model.save("carl.model")
        isDone = True
    elif(makeDefault == 'n'):
        isDone = True

