# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np


class data_wrapper:
    def __init__(self, batch_size):
        self.file_name = 'data/encodings.npz'
        self.inputs = self.read_file()
        self.batch_size = batch_size
        self.cur_batch = 0
        self.total_batch = int(len(self.inputs)/batch_size-1)

    def read_file(self):
        inputs = np.load(self.file_name)
        inputs = inputs['data']
        return inputs

    def get_next_batch(self):
        val = self.inputs[self.cur_batch*self.batch_size:self.cur_batch*self.batch_size+self.batch_size]
        self.cur_batch += 1
        if self.cur_batch*self.batch_size+self.batch_size+200 >= len(self.inputs):
            self.cur_batch = 0
        return val,val

    def get_unseen_data(self):
        val = self.inputs[-200:]
	return val,val

# Parameters
learning_rate = 0.01
training_epochs = 60
batch_size = 1
display_step = 1
examples_to_show = 10
dropout = .5

# Network Parameters
n_hidden_2 = 1 # 2nd layer num features
n_hidden_1 = 100 # 1st layer num features

n_input_1 = 300 # data dimension 1 
n_input_2 = 300 # data dimension 2

# Data Class
my_data = data_wrapper(batch_size)


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input_1, n_input_2])
dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input_2, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input_2])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([1,n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([1,n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([1,n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([1,n_input_2])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1

    assert x.get_shape().as_list() == [None,n_input_1,n_input_2]
    term = tf.reshape(x,[-1,n_input_2])
    assert term.get_shape().as_list() == [None,n_input_2]
    term = tf.matmul(term,weights['encoder_h1']) + biases['encoder_b1']
    assert term.get_shape().as_list() == [None,n_hidden_1]
    layer_1 = tf.nn.sigmoid(term)
    tf.nn.dropout(layer_1, dropout_placeholder)

    assert layer_1.get_shape().as_list() == [None,n_hidden_1]
    term2 = tf.matmul(layer_1,weights['encoder_h2']) + biases['encoder_b2']
    assert term2.get_shape().as_list() == [None,n_hidden_2]
    layer_2 = tf.nn.sigmoid(term2)
    assert layer_2.get_shape().as_list() == [None,n_hidden_2]
    result = tf.reshape(layer_2,[-1,n_input_2,n_hidden_2])
    assert result.get_shape().as_list() == [None,n_input_1,n_hidden_2]
    tf.nn.dropout(result, droupout_placeholder)
    return result



# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1

    assert x.get_shape().as_list() == [None,n_input_1,n_hidden_2]
    term = tf.reshape(x,[-1,n_hidden_2])
    assert term.get_shape().as_list() == [None,n_hidden_2]
    term = tf.matmul(term,weights['decoder_h1']) + biases['decoder_b1']
    assert term.get_shape().as_list() == [None,n_hidden_1]
    layer_1 = tf.nn.sigmoid(term)
    tf.nn.dropout(layer_1, dropout_placeholder)

    assert layer_1.get_shape().as_list() == [None,n_hidden_1]
    term2 = tf.matmul(layer_1,weights['decoder_h2']) + biases['decoder_b2']
    assert term2.get_shape().as_list() == [None,n_input_2]

    layer_2 = tf.nn.sigmoid(term2)
    assert layer_2.get_shape().as_list() == [None,n_input_2]
    result = tf.reshape(layer_2,[-1,n_input_1,n_input_2])
    assert result.get_shape().as_list() == [None,n_input_1,n_input_2]

    return result



# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = my_data.total_batch
    # Training cycle
    count = 0
    for epoch in range(training_epochs):
        # Loop over all batches
	epoch_cost = []
        for i in range(total_batch):
            batch_xs, batch_ys = my_data.get_next_batch()
            # Run optimization op (backprop) and cost op (to get loss value)
            output_feed = [optimizer, cost,decoder_op]
            input_feed = {X: batch_xs, dropout_placeholder: dropout}
            _, c, decoded = sess.run(output_feed, input_feed)
            count += 1
	    if count % 1000 == 0:
		    print('result')
                    for l in xrange(0,len(decoded[0])):
#			print( batch_xs[0][l])
			print( float(sum((decoded[0][l])))/len(decoded[0][1]))
#			print( " ")
            epoch_cost.append(c)
        # Display logs per epoch step
	epoch_cost = float(sum(epoch_cost))/len(epoch_cost)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(epoch_cost))
    print("unseen set")
    batch_xs, batch_ys = my_data.get_unseen_data()
    output_feed = [cost]
    input_feed = {X: batch_xs, dropout_placeholder: 1}
    c = sess.run(output_feed, input_feed)
    print(c)

    print("Optimization Finished!")

    # Applying encode and decode over test set
#    encode_decode = sess.run(
#        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
