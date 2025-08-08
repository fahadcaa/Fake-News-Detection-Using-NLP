__Using Deep Learning (LSTM)__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['processed_text'])
X = tokenizer.texts_to_sequences(df['processed_text'])
X = pad_sequences(X, maxlen=200)

# Model
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
.
 ____Using BERT (Transformers)___
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def bert_encode(texts, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return np.array(input_ids), np.array(attention_masks)

# Build BERT model
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bert_model.fit(X_train, y_train, epochs=2, batch_size=8, validation_data=(X_test, y_test))
