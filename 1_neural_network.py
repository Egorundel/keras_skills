'''
base mathematics:
y = m*x + b

y - выход (может быть несколько)
x - вход (разные форматы, размеры)
m - вес (факторы, как оценки за четверть, посещаемость)
b - предвзятость (часто вычисляется самой нейронной сетью)
'''

# ЗАДАЧА: Сколько будет стоить дом определенной комнатности?
import numpy as np
import tensorflow as tf
import keras

def house(rooms):

    # варианты количества комнат
    xs = np.array([0, 1, 2, 3, 4, 5])
    # xs = np.arange(0, 6, 1, dtype=int)

    # цены для квартир
    ys = np.array([50, 101, 148, 200, 249, 300])

    # создание модели
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape = [1])])

    # компиляция модели
    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(xs, ys, epochs=600)

    return model.predict(rooms)

prediction = house([7, 5, 3, 1, 10, 12])
print(prediction)