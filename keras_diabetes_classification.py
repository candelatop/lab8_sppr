from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy

# задаем для воспроизводимости результатов
numpy.random.seed(2)

# загружаем датасет, соответствующий последним пяти годам до определение диагноза 
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")

# разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
X = dataset[:,0:8]
Y = dataset[:,8]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# создаем модели, добавляем слои один за другим
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu')) # входной слой требует задать input_dim
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid')) # сигмоида вместо relu для определения вероятности

# компилируем модель
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# обучаем нейронную сеть
model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test))

# сохраняем и выводим
model.save('weights.h5')
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
