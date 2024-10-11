#Modules
import tensorflow_datasets as tfds
import tensorflow as tf
from keras import models as m
from keras import layers as l
import matplotlib.pyplot as plt

#We download the dataset
(train_dataset,test_dataset),dataset_info=tfds.load('mnist',with_info=True,as_supervised=True,
                                                    shuffle_files=True,split=['train','test'])

width=height=28

#We need to change the dtype of the images because is tf.uint8, for that we are going to use tf.cast.
def units(data,label):
    return tf.cast(data,tf.float32)/255.0,label

train_dataset = train_dataset.map(units, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset=test_dataset.map(units,num_parallel_calls=tf.data.AUTOTUNE)

#Shuffle the data
train_dataset=train_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset=test_dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

#Building the model
model=m.Sequential([
    l.InputLayer(shape=(width,height,1)),
    l.Conv2D(filters=32,kernel_size=5,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.25),
    l.Flatten(),
    l.Dense(64, activation='relu'),
    l.BatchNormalization(),
    l.Dropout(rate=0.5),
    l.Dense(10,activation='softmax'),

])
print(model.summary())

#Compile the model
model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
epochs = 3
history = model.fit(train_dataset, batch_size=32, epochs=epochs, validation_data=(test_dataset))

#Two graphs that show what happen in the training
plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#Saving the model
model.save('datasetmnist.h5')
model.save('datasetmnist.keras')