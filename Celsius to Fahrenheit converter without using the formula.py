import tensorflow as tf
import numpy as np
import logging
logger=tf.get_logger()
logger.setLevel(logging.ERROR)
celsius=np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit=np.array([-40,14,32,46,59,72,10],dtype=float)
l0=tf.keras.layers.Dense(input_shape=[1],units=1)
model=tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history=model.fit(celsius,fahrenheit,epochs=500,verbose=False)

print(model.predict([24]))
