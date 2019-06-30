import util
import time
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

# args
parser = ArgumentParser(description="carl promises to be good boy")
parser.add_argument('-f', '--file', help="file to save to, otherwise uses recent")
args = parser.parse_args()

# get name
name = "recent"
if args.file:
    name = args.file
(x, y) = util.get_data(name)

# config
config = util.read_config()
MODEL_NAME = f"carl-{config['type']}:{config['version']}-{name}-{config['epochs']}-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}")

# get the data
print('Raw Data:')
print(x.shape)
print(y.shape)

# create validation and training set
x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=7, test_size=0.2)

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
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(config['displays']), activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

