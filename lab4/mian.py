import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianDropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD


max_features = 10000  
maxlen = 500         

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(GaussianDropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(GaussianDropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)


loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

num = 10
to_predict = x_test[:num]
predictions = model.predict(to_predict)

print("Предсказанная и фактическая оценка")
print("predict    Заданные значения")
for i in range(num):
    print(predictions[i])
    if float(predictions[i][0]) > 0.5:
        print(f'1 ===== {predictions[i][0]:.2f} ~ {y_test[i]}')
    else:
        print(f'0 ===== {predictions[i][0]:.2f} ~ {y_test[i]}')
    print('\n')
