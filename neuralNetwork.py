import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.Sequential([
    keras.layers.Dense(4, input_shape=(3,), activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# input data
X_train = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
# output data
y_train = [[0, 0], [1, 0], [0, 1]]
model.fit(X_train, y_train, epochs=1000)
# input data for testing
X_test = [[0, 0, 0]]
# predict the output
y_pred = model.predict(X_test)
# evaluate the model on the training data
loss, accuracy = model.evaluate(X_train, y_train)
print('Loss:', loss)
print('Accuracy:', accuracy)
# plt.plot(X_test)
print(y_pred)