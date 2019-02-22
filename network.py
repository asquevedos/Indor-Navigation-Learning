# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
# Helper libraries
import numpy as np

import pandas as pd 



print(tf.__version__)

from sklearn.model_selection import train_test_split



column_names=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','TARGET']

SpamDS = pd.read_csv("entrenamientoAllDataNotHeaders.csv",names=column_names)



features = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11']

target= ['TARGET']

# Dividimos la Data para entrenar el modelo    
x_vals =  pd.DataFrame(np.c_[SpamDS ['a1'],SpamDS ['a2'],SpamDS ['a3'],SpamDS ['a4'],SpamDS ['a5'],
SpamDS ['a6'],SpamDS ['a7'],SpamDS ['a8'],SpamDS ['a9'],SpamDS ['a10'],SpamDS ['a11']], columns = features)        
  
y_vals= pd.DataFrame(np.c_[SpamDS['TARGET']])


x_vals=x_vals.values
y_vals=y_vals.values

train_images,test_images,train_labels, test_labels= train_test_split(x_vals, y_vals, test_size = 0.4,shuffle =True,random_state= 5) 
class_names = ['1', '2', '3', '4']


train_images = train_images / 100

test_images = test_images / 100


model = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)

])
sgd = keras.optimizers.SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_images, y=train_labels, validation_split=0.33, epochs=250, batch_size=10, )

#history = model.fit(x=train_images, y=train_labels, epochs=100)


test_loss, test_acc = model.evaluate(test_images, test_labels)


plt.plot(history.history['acc'], 'k-', label='Training Accuracy')
plt.plot(history.history['val_acc'], 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[1]

np.argmax(predictions[364])
print('Test Loss:', test_loss)
