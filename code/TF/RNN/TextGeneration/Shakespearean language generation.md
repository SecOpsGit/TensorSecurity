#
```
TensorFlow Machine Learning Cookbook Second Edition
TensorFlow Machine Learning Cookbook - Second Edition
Nick McClure
August 30, 2018

https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook-second-edition?
utm_source=github&utm_medium=repository&utm_campaign=9781789131680


https://github.com/PacktPublishing/TensorFlow-Machine-Learning-Cookbook-Second-Edition

a sequence RNN with LSTM cells to try to predict the next
words, trained on the works of Shakespeare
```


```
import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session
sess = tf.Session()

# Set RNN Parameters
min_word_freq = 5  # Trim the less frequent words off
rnn_size = 128  # RNN Model size
epochs = 10  # Number of epochs to cycle through data
batch_size = 100  # Train on this many examples at once
learning_rate = 0.001  # Learning rate
training_seq_len = 50  # how long of a word group to consider
embedding_size = rnn_size  # Word embedding size
save_every = 500  # How often to save model checkpoints
eval_every = 50  # How often to evaluate the test sentences
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# Download/store Shakespeare data
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)

# Declare punctuation to remove, everything except hyphens and apostrophes
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

# Make Model Directory
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Loading Shakespeare Data')
# Check if file is downloaded.
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Not found, downloading Shakespeare texts from www.gutenberg.org')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    # Get Shakespeare text
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    s_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    s_text = s_text[7675:]
    # Remove newlines
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
else:
    # If file has been saved, load from that file
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

# Clean text
print('Cleaning Text')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text).strip().lower()


# Build word vocabulary function
def build_vocab(text, min_freq):
    word_counts = collections.Counter(text.split(' '))
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (i_x+1) for i_x, key in enumerate(words)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
    
    return ix_to_vocab_dict, vocab_to_ix_dict


# Build Shakespeare vocabulary
print('Building Shakespeare Vocab')
ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert(len(ix2vocab) == len(vocab2ix))

# Convert text to word vectors
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except KeyError:
        s_text_ix.append(0)
s_text_ix = np.array(s_text_ix)


# Define LSTM RNN Model
class LSTM_Model():
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        
        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len
        
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        
        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        
            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
                                            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # Get embedded vector
            out = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return out
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # Non inferred outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])])
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word
        return out_sentence


# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)


# Create model saver
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h
        
        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)
        
        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
        
        # Save the model and the vocab
        if iteration_count % save_every == 0:
            # Save model
            model_file_name = os.path.join(full_model_dir, 'model')
            saver.save(sess, model_file_name, global_step=iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # Save vocabulary
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        
        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                
        iteration_count += 1


# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

```

### 20191023執行成果
```
Loading Shakespeare Data
Not found, downloading Shakespeare texts from www.gutenberg.org
Cleaning Text
Building Shakespeare Vocab
Vocabulary Length = 8009
WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

WARNING:tensorflow:From <ipython-input-1-41a36ef311f3>:129: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/rnn_cell_impl.py:735: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/rnn_cell_impl.py:739: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/clip_ops.py:301: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Starting Epoch #1 of 10.
Iteration: 10, Epoch: 1, Batch: 10 out of 182, Loss: 9.88
Iteration: 20, Epoch: 1, Batch: 20 out of 182, Loss: 9.16
Iteration: 30, Epoch: 1, Batch: 30 out of 182, Loss: 8.57
Iteration: 40, Epoch: 1, Batch: 40 out of 182, Loss: 8.49
Iteration: 50, Epoch: 1, Batch: 50 out of 182, Loss: 8.27
thou art more
to be or not to the
wherefore art thou hew seen let me and
Iteration: 60, Epoch: 1, Batch: 60 out of 182, Loss: 7.92
Iteration: 70, Epoch: 1, Batch: 70 out of 182, Loss: 7.79
Iteration: 80, Epoch: 1, Batch: 80 out of 182, Loss: 7.55
Iteration: 90, Epoch: 1, Batch: 90 out of 182, Loss: 7.37
Iteration: 100, Epoch: 1, Batch: 100 out of 182, Loss: 7.25
thou art more a
to be or not to the
wherefore art thou art
Iteration: 110, Epoch: 1, Batch: 110 out of 182, Loss: 7.06
Iteration: 120, Epoch: 1, Batch: 120 out of 182, Loss: 6.74
Iteration: 130, Epoch: 1, Batch: 130 out of 182, Loss: 6.80
Iteration: 140, Epoch: 1, Batch: 140 out of 182, Loss: 6.86
Iteration: 150, Epoch: 1, Batch: 150 out of 182, Loss: 6.91
thou art more than
to be or not to the
wherefore art thou art
Iteration: 160, Epoch: 1, Batch: 160 out of 182, Loss: 6.85
Iteration: 170, Epoch: 1, Batch: 170 out of 182, Loss: 6.53
Iteration: 180, Epoch: 1, Batch: 180 out of 182, Loss: 6.51
Starting Epoch #2 of 10.
Iteration: 190, Epoch: 2, Batch: 9 out of 182, Loss: 6.52
Iteration: 200, Epoch: 2, Batch: 19 out of 182, Loss: 6.47
thou art more than
to be or not to the
wherefore art thou art
Iteration: 210, Epoch: 2, Batch: 29 out of 182, Loss: 6.38
Iteration: 220, Epoch: 2, Batch: 39 out of 182, Loss: 6.53
Iteration: 230, Epoch: 2, Batch: 49 out of 182, Loss: 6.45
Iteration: 240, Epoch: 2, Batch: 59 out of 182, Loss: 6.45
Iteration: 250, Epoch: 2, Batch: 69 out of 182, Loss: 6.33
thou art more than
to be or not to the
wherefore art thou art
Iteration: 260, Epoch: 2, Batch: 79 out of 182, Loss: 6.22
Iteration: 270, Epoch: 2, Batch: 89 out of 182, Loss: 6.15
Iteration: 280, Epoch: 2, Batch: 99 out of 182, Loss: 6.22
Iteration: 290, Epoch: 2, Batch: 109 out of 182, Loss: 6.39
Iteration: 300, Epoch: 2, Batch: 119 out of 182, Loss: 6.18
thou art more than
to be or not to the
wherefore art thou art
Iteration: 310, Epoch: 2, Batch: 129 out of 182, Loss: 6.20
Iteration: 320, Epoch: 2, Batch: 139 out of 182, Loss: 6.26
Iteration: 330, Epoch: 2, Batch: 149 out of 182, Loss: 6.27
Iteration: 340, Epoch: 2, Batch: 159 out of 182, Loss: 6.33
Iteration: 350, Epoch: 2, Batch: 169 out of 182, Loss: 6.21
thou art more than
to be or not to be
wherefore art thou art
Iteration: 360, Epoch: 2, Batch: 179 out of 182, Loss: 6.31
Starting Epoch #3 of 10.
Iteration: 370, Epoch: 3, Batch: 8 out of 182, Loss: 6.83
Iteration: 380, Epoch: 3, Batch: 18 out of 182, Loss: 6.68
Iteration: 390, Epoch: 3, Batch: 28 out of 182, Loss: 6.70
Iteration: 400, Epoch: 3, Batch: 38 out of 182, Loss: 6.63
thou art more than
to be or not to the
wherefore art thou art
Iteration: 410, Epoch: 3, Batch: 48 out of 182, Loss: 6.33
Iteration: 420, Epoch: 3, Batch: 58 out of 182, Loss: 6.19
Iteration: 430, Epoch: 3, Batch: 68 out of 182, Loss: 6.16
Iteration: 440, Epoch: 3, Batch: 78 out of 182, Loss: 6.16
Iteration: 450, Epoch: 3, Batch: 88 out of 182, Loss: 6.43
thou art more than
to be or not to the
wherefore art thou art
Iteration: 460, Epoch: 3, Batch: 98 out of 182, Loss: 6.07
Iteration: 470, Epoch: 3, Batch: 108 out of 182, Loss: 6.41
Iteration: 480, Epoch: 3, Batch: 118 out of 182, Loss: 6.17
Iteration: 490, Epoch: 3, Batch: 128 out of 182, Loss: 6.23
Iteration: 500, Epoch: 3, Batch: 138 out of 182, Loss: 6.17
Model Saved To: temp/shakespeare_model/model
thou art more than
to be or not to the
wherefore art thou art
Iteration: 510, Epoch: 3, Batch: 148 out of 182, Loss: 6.06
Iteration: 520, Epoch: 3, Batch: 158 out of 182, Loss: 6.13
Iteration: 530, Epoch: 3, Batch: 168 out of 182, Loss: 6.27
Iteration: 540, Epoch: 3, Batch: 178 out of 182, Loss: 6.41
Starting Epoch #4 of 10.
Iteration: 550, Epoch: 4, Batch: 7 out of 182, Loss: 6.10
thou art more than
to be or not to the
wherefore art thou art
Iteration: 560, Epoch: 4, Batch: 17 out of 182, Loss: 5.99
Iteration: 570, Epoch: 4, Batch: 27 out of 182, Loss: 6.13
Iteration: 580, Epoch: 4, Batch: 37 out of 182, Loss: 6.18
Iteration: 590, Epoch: 4, Batch: 47 out of 182, Loss: 6.04
Iteration: 600, Epoch: 4, Batch: 57 out of 182, Loss: 6.09
thou art more than
to be or not to the
wherefore art thou art
Iteration: 610, Epoch: 4, Batch: 67 out of 182, Loss: 6.06
Iteration: 620, Epoch: 4, Batch: 77 out of 182, Loss: 6.01
Iteration: 630, Epoch: 4, Batch: 87 out of 182, Loss: 6.05
Iteration: 640, Epoch: 4, Batch: 97 out of 182, Loss: 6.28
Iteration: 650, Epoch: 4, Batch: 107 out of 182, Loss: 5.98
thou art more than
to be or not to the
wherefore art thou art
Iteration: 660, Epoch: 4, Batch: 117 out of 182, Loss: 6.09
Iteration: 670, Epoch: 4, Batch: 127 out of 182, Loss: 6.15
Iteration: 680, Epoch: 4, Batch: 137 out of 182, Loss: 6.28
Iteration: 690, Epoch: 4, Batch: 147 out of 182, Loss: 6.00
Iteration: 700, Epoch: 4, Batch: 157 out of 182, Loss: 5.95
thou art more than
to be or not to the
wherefore art thou art
Iteration: 710, Epoch: 4, Batch: 167 out of 182, Loss: 6.08
Iteration: 720, Epoch: 4, Batch: 177 out of 182, Loss: 6.09
Starting Epoch #5 of 10.
Iteration: 730, Epoch: 5, Batch: 6 out of 182, Loss: 6.12
Iteration: 740, Epoch: 5, Batch: 16 out of 182, Loss: 6.30
Iteration: 750, Epoch: 5, Batch: 26 out of 182, Loss: 6.05
thou art more than
to be or not to the
wherefore art thou art
Iteration: 760, Epoch: 5, Batch: 36 out of 182, Loss: 6.07
Iteration: 770, Epoch: 5, Batch: 46 out of 182, Loss: 6.14
Iteration: 780, Epoch: 5, Batch: 56 out of 182, Loss: 6.16
Iteration: 790, Epoch: 5, Batch: 66 out of 182, Loss: 6.35
Iteration: 800, Epoch: 5, Batch: 76 out of 182, Loss: 5.92
thou art more than
to be or not to the
wherefore art thou art
Iteration: 810, Epoch: 5, Batch: 86 out of 182, Loss: 6.03
Iteration: 820, Epoch: 5, Batch: 96 out of 182, Loss: 5.70
Iteration: 830, Epoch: 5, Batch: 106 out of 182, Loss: 5.82
Iteration: 840, Epoch: 5, Batch: 116 out of 182, Loss: 5.93
Iteration: 850, Epoch: 5, Batch: 126 out of 182, Loss: 5.83
thou art more than
to be or not to the
wherefore art thou hast
Iteration: 860, Epoch: 5, Batch: 136 out of 182, Loss: 6.05
Iteration: 870, Epoch: 5, Batch: 146 out of 182, Loss: 6.05
Iteration: 880, Epoch: 5, Batch: 156 out of 182, Loss: 6.16
Iteration: 890, Epoch: 5, Batch: 166 out of 182, Loss: 5.78
Iteration: 900, Epoch: 5, Batch: 176 out of 182, Loss: 6.09
thou art more than
to be or not to the
wherefore art thou art
Starting Epoch #6 of 10.
Iteration: 910, Epoch: 6, Batch: 5 out of 182, Loss: 5.86
Iteration: 920, Epoch: 6, Batch: 15 out of 182, Loss: 6.01
Iteration: 930, Epoch: 6, Batch: 25 out of 182, Loss: 6.00
Iteration: 940, Epoch: 6, Batch: 35 out of 182, Loss: 5.92
Iteration: 950, Epoch: 6, Batch: 45 out of 182, Loss: 6.10
thou art more than
to be or not to the
wherefore art thou art
Iteration: 960, Epoch: 6, Batch: 55 out of 182, Loss: 5.99
Iteration: 970, Epoch: 6, Batch: 65 out of 182, Loss: 5.89
Iteration: 980, Epoch: 6, Batch: 75 out of 182, Loss: 5.88
Iteration: 990, Epoch: 6, Batch: 85 out of 182, Loss: 6.24
Iteration: 1000, Epoch: 6, Batch: 95 out of 182, Loss: 5.99
Model Saved To: temp/shakespeare_model/model
thou art more than
to be or not to the
wherefore art thou art
Iteration: 1010, Epoch: 6, Batch: 105 out of 182, Loss: 5.92
Iteration: 1020, Epoch: 6, Batch: 115 out of 182, Loss: 5.67
Iteration: 1030, Epoch: 6, Batch: 125 out of 182, Loss: 5.97
Iteration: 1040, Epoch: 6, Batch: 135 out of 182, Loss: 5.94
Iteration: 1050, Epoch: 6, Batch: 145 out of 182, Loss: 5.95
thou art more than
to be or not to the
wherefore art thou art a
Iteration: 1060, Epoch: 6, Batch: 155 out of 182, Loss: 5.99
Iteration: 1070, Epoch: 6, Batch: 165 out of 182, Loss: 6.21
Iteration: 1080, Epoch: 6, Batch: 175 out of 182, Loss: 6.01
Starting Epoch #7 of 10.
Iteration: 1090, Epoch: 7, Batch: 4 out of 182, Loss: 5.87
Iteration: 1100, Epoch: 7, Batch: 14 out of 182, Loss: 5.67
thou art more
to be or not to the
wherefore art thou art
Iteration: 1110, Epoch: 7, Batch: 24 out of 182, Loss: 5.80
Iteration: 1120, Epoch: 7, Batch: 34 out of 182, Loss: 5.79
Iteration: 1130, Epoch: 7, Batch: 44 out of 182, Loss: 6.00
Iteration: 1140, Epoch: 7, Batch: 54 out of 182, Loss: 5.98
Iteration: 1150, Epoch: 7, Batch: 64 out of 182, Loss: 5.88
thou art more than
to be or not to the
wherefore art thou hast
Iteration: 1160, Epoch: 7, Batch: 74 out of 182, Loss: 6.06
Iteration: 1170, Epoch: 7, Batch: 84 out of 182, Loss: 5.74
Iteration: 1180, Epoch: 7, Batch: 94 out of 182, Loss: 5.63
Iteration: 1190, Epoch: 7, Batch: 104 out of 182, Loss: 5.98
Iteration: 1200, Epoch: 7, Batch: 114 out of 182, Loss: 5.84
thou art more than
to be or not to the
wherefore art thou art a
Iteration: 1210, Epoch: 7, Batch: 124 out of 182, Loss: 5.85
Iteration: 1220, Epoch: 7, Batch: 134 out of 182, Loss: 5.86
Iteration: 1230, Epoch: 7, Batch: 144 out of 182, Loss: 5.96
Iteration: 1240, Epoch: 7, Batch: 154 out of 182, Loss: 5.91
Iteration: 1250, Epoch: 7, Batch: 164 out of 182, Loss: 5.57
thou art more than
to be or not to the
wherefore art thou hast done all the
Iteration: 1260, Epoch: 7, Batch: 174 out of 182, Loss: 6.02
Starting Epoch #8 of 10.
Iteration: 1270, Epoch: 8, Batch: 3 out of 182, Loss: 5.97
Iteration: 1280, Epoch: 8, Batch: 13 out of 182, Loss: 5.80
Iteration: 1290, Epoch: 8, Batch: 23 out of 182, Loss: 5.92
Iteration: 1300, Epoch: 8, Batch: 33 out of 182, Loss: 5.90
thou art more
to be or not to the
wherefore art thou art
Iteration: 1310, Epoch: 8, Batch: 43 out of 182, Loss: 5.83
Iteration: 1320, Epoch: 8, Batch: 53 out of 182, Loss: 5.86
Iteration: 1330, Epoch: 8, Batch: 63 out of 182, Loss: 6.15
Iteration: 1340, Epoch: 8, Batch: 73 out of 182, Loss: 5.87
Iteration: 1350, Epoch: 8, Batch: 83 out of 182, Loss: 6.08
thou art more than a
to be or not to the
wherefore art thou art
Iteration: 1360, Epoch: 8, Batch: 93 out of 182, Loss: 5.75
Iteration: 1370, Epoch: 8, Batch: 103 out of 182, Loss: 5.95
Iteration: 1380, Epoch: 8, Batch: 113 out of 182, Loss: 5.78
Iteration: 1390, Epoch: 8, Batch: 123 out of 182, Loss: 5.70
Iteration: 1400, Epoch: 8, Batch: 133 out of 182, Loss: 6.07
thou art more
to be or not to be
wherefore art thou art not so i am a
Iteration: 1410, Epoch: 8, Batch: 143 out of 182, Loss: 6.01
Iteration: 1420, Epoch: 8, Batch: 153 out of 182, Loss: 5.83
Iteration: 1430, Epoch: 8, Batch: 163 out of 182, Loss: 5.97
Iteration: 1440, Epoch: 8, Batch: 173 out of 182, Loss: 5.88
Starting Epoch #9 of 10.
Iteration: 1450, Epoch: 9, Batch: 2 out of 182, Loss: 5.82
thou art more
to be or not to the
wherefore art thou art
Iteration: 1460, Epoch: 9, Batch: 12 out of 182, Loss: 5.94
Iteration: 1470, Epoch: 9, Batch: 22 out of 182, Loss: 5.87
Iteration: 1480, Epoch: 9, Batch: 32 out of 182, Loss: 5.83
Iteration: 1490, Epoch: 9, Batch: 42 out of 182, Loss: 5.79
Iteration: 1500, Epoch: 9, Batch: 52 out of 182, Loss: 5.63
Model Saved To: temp/shakespeare_model/model
thou art more
to be or not to the
wherefore art thou hast done
Iteration: 1510, Epoch: 9, Batch: 62 out of 182, Loss: 5.43
Iteration: 1520, Epoch: 9, Batch: 72 out of 182, Loss: 5.73
Iteration: 1530, Epoch: 9, Batch: 82 out of 182, Loss: 5.82
Iteration: 1540, Epoch: 9, Batch: 92 out of 182, Loss: 5.75
Iteration: 1550, Epoch: 9, Batch: 102 out of 182, Loss: 5.76
thou art more
to be or not to the
wherefore art thou hast
Iteration: 1560, Epoch: 9, Batch: 112 out of 182, Loss: 6.09
Iteration: 1570, Epoch: 9, Batch: 122 out of 182, Loss: 5.77
Iteration: 1580, Epoch: 9, Batch: 132 out of 182, Loss: 5.96
Iteration: 1590, Epoch: 9, Batch: 142 out of 182, Loss: 5.83
Iteration: 1600, Epoch: 9, Batch: 152 out of 182, Loss: 5.84
thou art more
to be or not to the
wherefore art thou hast done all the
Iteration: 1610, Epoch: 9, Batch: 162 out of 182, Loss: 5.53
Iteration: 1620, Epoch: 9, Batch: 172 out of 182, Loss: 5.84
Starting Epoch #10 of 10.
Iteration: 1630, Epoch: 10, Batch: 1 out of 182, Loss: 5.80
Iteration: 1640, Epoch: 10, Batch: 11 out of 182, Loss: 5.78
Iteration: 1650, Epoch: 10, Batch: 21 out of 182, Loss: 5.92
thou art more
to be or not to the
wherefore art thou hast done
Iteration: 1660, Epoch: 10, Batch: 31 out of 182, Loss: 5.99
Iteration: 1670, Epoch: 10, Batch: 41 out of 182, Loss: 5.93
Iteration: 1680, Epoch: 10, Batch: 51 out of 182, Loss: 5.76
Iteration: 1690, Epoch: 10, Batch: 61 out of 182, Loss: 5.71
Iteration: 1700, Epoch: 10, Batch: 71 out of 182, Loss: 5.96
thou art more than
to be or not to be
wherefore art thou art not so i am a
Iteration: 1710, Epoch: 10, Batch: 81 out of 182, Loss: 5.98
Iteration: 1720, Epoch: 10, Batch: 91 out of 182, Loss: 5.94
Iteration: 1730, Epoch: 10, Batch: 101 out of 182, Loss: 5.70
Iteration: 1740, Epoch: 10, Batch: 111 out of 182, Loss: 5.50
Iteration: 1750, Epoch: 10, Batch: 121 out of 182, Loss: 5.90
thou art more
to be or not to the
wherefore art thou hast done all the
Iteration: 1760, Epoch: 10, Batch: 131 out of 182, Loss: 5.73
Iteration: 1770, Epoch: 10, Batch: 141 out of 182, Loss: 5.59
Iteration: 1780, Epoch: 10, Batch: 151 out of 182, Loss: 5.67
Iteration: 1790, Epoch: 10, Batch: 161 out of 182, Loss: 5.89
Iteration: 1800, Epoch: 10, Batch: 171 out of 182, Loss: 6.01
thou art more
to be or not to be
wherefore art thou hast done
Iteration: 1810, Epoch: 10, Batch: 181 out of 182, Loss: 5.84

```
