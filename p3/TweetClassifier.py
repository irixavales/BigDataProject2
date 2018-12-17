import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path

np.random.seed(1237)

# path of tdirectory where training data set is
path_train = '/home/irixa/PycharmProjects/BigDataProject/input'
# file path of testing data set
path_test = '/home/irixa/PycharmProjects/BigDataProject/input'

files_train = skds.load_files(path_train, load_content=False)
files_test = skds.load_files(path_test, load_content=False)

label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ["text","classification"]
data_list = []

i = 0
for f in labelled_files:
    data_list.append((f, label_names[label_index[i]], Path(f).read_text()))
    i += 1

data = pd.DataFrame.from_records(data_list, columns=data_tags)


train_size = int(len(data))

train_text = data['text'][:train_size]
train_classification = data['classification'][:train_size]

test_text = data['text'][:train_size]
test_classification = data['classification'][:train_size]


num_labels = 20
vocab_size = 15000
batch_size = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_text)

x_train = tokenizer.texts_to_matrix(train_text, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_text, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_classification)
y_train = encoder.transform(train_classification)
y_test = encoder.transform(test_classification)


model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])

text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction[0])]
    print(test_text.iloc[i])
    print('Actual label:' + test_classification.iloc[i])
    print("Predicted label: " + predicted_label)


model.model.save('tweets-classified.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = load_model('tweets-classified.h5')

tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

encoder.classes_


labels = np.array([0, 1, 2])

x_data = []
for t_f in test_text:
    t_f_data = Path(t_f).read_text()
    x_data.append(t_f_data)

x_data_series = pd.Series(x_data)
x_tokenized = tokenizer.texts_to_matrix(x_data_series, mode='tfidf')

i = 0
for x_t in x_tokenized:
    prediction = model.predict(np.array([x_t]))
    predicted_label = labels[np.argmax(prediction[0])]
    print("File ->", test_text[i], "Predicted label: " + predicted_label)
    i += 1