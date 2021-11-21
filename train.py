import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import argparse
import os
import numpy as np
# import json

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
#from keras.optimizers import SGD
#from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
# from keras.utils import multi_gpu_model


#input 64x64
class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		

		
		model.add(Conv2D(18, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(18, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(36, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(36, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(Dropout(0.7))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		# return the constructed network architecture
		return model


def _load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, 'training.npy'))
    y_train = np.load(os.path.join(base_dir, 'training_label.npy'))
    return X_train, y_train


def _load_validation_data(base_dir):
    X_val = np.load(os.path.join(base_dir, 'validation.npy'))
    y_val = np.load(os.path.join(base_dir, 'validation_label.npy'))
    return X_val, y_val


# def _parse_args():
#     parser = argparse.ArgumentParser()

#     # Data, model, and output directories
#     # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
#     parser.add_argument('--model_dir', type=str)
#     parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
#     parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

#     return parser.parse_known_args()


# if __name__ == "__main__":
#     args, unknown = _parse_args()

#     train_data, train_labels = _load_training_data(args.train)
#     eval_data, eval_labels = _load_validation_data(args.train)

#     mdl = model(train_data, train_labels, eval_data, eval_labels)

#     if args.current_host == args.hosts[0]:
#         # save model to an S3 directory with version number '00000001'
#         mdl.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
if __name__ == '__main__':
    print ("tensorflow version:", tf.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--learning-rate', type=float, default=0.01)
    # parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=(1e-5)/4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    os.makedirs(model_dir, exist_ok=True)
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # input image dimensions
    img_rows, img_cols = 64, 64

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    K.set_image_data_format('channels_last')  
    print(K.image_data_format())

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (None, 1, img_rows, img_cols)
        batch_norm_axis=1
    else:
        # channels last
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (None, img_rows, img_cols, 1)
        batch_norm_axis=-1

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    # Normalize pixel values
    x_train  = x_train.astype('float32')
    x_val    = x_val.astype('float32')
    x_train /= 255
    x_val   /= 255
    
    # Convert class vectors to binary class matrices
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val   = keras.utils.to_categorical(y_val, num_classes)


    INIT_LR = (1e-5)/4
    BS = 32
    EPOCHS = 2
    # adam_opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
    optimizer=tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model = LivenessNet.build(width=64, height=64, depth=1, classes=2)

    print(model.summary())

    print("[INFO] compiling model...")
    #configure the learning process
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer= optimizer, 
        metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_val, y_val), 
                    epochs=epochs,
                    steps_per_epoch=len(x_train) / batch_size,
                    verbose=1)
    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    tf.saved_model.save(model, os.path.join(model_dir, "1"))
