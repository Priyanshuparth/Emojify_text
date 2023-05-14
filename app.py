import flask
from flask import render_template,request,Flask
import tensorflow as tf
import numpy as np
import pandas as pd
import emoji
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('emoji_data.csv', header = None)
data=data.dropna()
X = data[0].values
Y = data[1].values
X=np.delete(X,29)
Y=np.delete(Y,29)
file = open('glove.6B.100d.txt', 'r', encoding = 'utf8')
content = file.readlines()
file.close()
embeddings = {}
for line in content:
    line = line.split()
    embeddings[line[0]] = np.array(line[1:], dtype = float)
def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2index = tokenizer.word_index
Xtokens = tokenizer.texts_to_sequences(X)
maxlen = get_maxlen(Xtokens)
Xtrain = pad_sequences(Xtokens, maxlen = maxlen,  padding = 'post', truncating = 'post')
Ytrain = to_categorical(Y)
embed_size = 100
embedding_matrix = np.zeros((len(word2index)+1, embed_size))
for word, i in word2index.items():
    embed_vector = embeddings[word]
    embedding_matrix[i] = embed_vector
model = Sequential([
    Embedding(input_dim = len(word2index) + 1,
              output_dim = embed_size,
              input_length = maxlen,
              weights = [embedding_matrix],
              trainable = False
             ),
    
    LSTM(units = 16, return_sequences = True),
    LSTM(units = 4),
    Dense(5, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(Xtrain, Ytrain, epochs = 100)
emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}

def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    test = [request.form['entry']]
    print(test)

    test_seq = tokenizer.texts_to_sequences(test)
    Xtest = pad_sequences(test_seq, maxlen = 10, padding = 'post', truncating = 'post')

    y_pred = model.predict(Xtest)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis = 1)
    print(y_pred)
    for i in range(len(test)):
        outtext=test[i]+label_to_emoji(y_pred[i])
    return render_template('result.html',outtext=outtext)

if __name__ == '__main__':
    app.run()