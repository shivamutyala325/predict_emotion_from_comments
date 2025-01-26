import pickle
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data=pd.read_csv(r"path_of_training_dataset")
test_data=pd.read_csv(r"path_of_testing_dataset")


train_data=pd.DataFrame(train_data)
test_data=pd.DataFrame(test_data)


train_sentences=train_data['text']
train_labels=train_data['label']
train_labels=to_categorical(train_labels)

test_sentences=test_data['text']
test_labels=test_data['label']
test_labels=to_categorical(test_labels)

tokenizer=Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(train_sentences)
word_indices=tokenizer.word_index

#saving the tokenizer to use the same kind of tokenisation while prediction
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


train_sequences=tokenizer.texts_to_sequences(train_sentences)
test_sequences=tokenizer.texts_to_sequences(test_sentences)

maxlen=max(len(sent) for sent in train_sentences)

paded_train_sequences=pad_sequences(train_sequences,maxlen=maxlen,padding='post')
paded_test_sequences=pad_sequences(test_sequences,maxlen=maxlen,padding='post')


def get_embeddings(maxlen, vocab_size, embd_dim):
    inputs = layers.Input(shape=(maxlen,))
    token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embd_dim)(inputs)
    position_embeddings = layers.Embedding(input_dim=maxlen, output_dim=embd_dim)(tf.range(start=0, limit=maxlen, delta=1))
    embeddings = token_embeddings + position_embeddings
    return keras.Model(inputs, embeddings)


def transformer_encoder(inputs, embed_dim, dense_dim, num_heads):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    proj_inputs = layers.LayerNormalization()(attention_output + inputs)
    proj_outputs = layers.Dense(dense_dim, activation='relu')(proj_inputs)
    proj_outputs = layers.Dense(embed_dim)(proj_outputs)
    final_res = layers.LayerNormalization()(proj_inputs + proj_outputs)
    return final_res


vocab_size = len(word_indices) + 1
embd_dim = 32
n_heads = 3
dense_dim = 32
batch_size = 64

inputs = keras.Input(shape=(maxlen,))

embedding_layer = get_embeddings(maxlen, vocab_size, embd_dim)
x = embedding_layer(inputs)
x = transformer_encoder(x, embd_dim, dense_dim, n_heads)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(6, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history=model.fit(paded_train_sequences, train_labels,validation_split=0.2,batch_size=batch_size, epochs=15)

test_loss, test_accuracy = model.evaluate(paded_test_sequences, test_labels)
print(f'Test accuracy: {test_accuracy}')


#saving the trained data into a file
model.save('text_base.keras', overwrite=True)


#ploting the training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

