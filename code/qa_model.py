from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.nn import dynamic_rnn

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim, FLAGS):
        self.size = size
        self.vocab_dim = vocab_dim
        self.FLAGS = FLAGS

    def encode(self, inputs, masks, encoder_state_input = None):    # LSTM Preprocessing and Match-LSTM Layers
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        #Encode question
        # (fw_out, bw_out), _ = bidirectional_dynamic_rnn(self.cell,
        #                                                 self.cell,
        #                                                 inp,
        #                                                 srclen,
        #                                                 scope = scope,
        #                                                 time_major = True,
        #                                                 dtype = dtypes.float32)

        ### Right now the way we set everything up, the first dimension of each input is arbitrary
        ### If we don't want to handle batching for now, I'm not exactly what to change but I think
        ### it can be traced back to self.paragraph_placeholder

        cell = tf.contrib.rnn.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag

        #Got the below from the following link. 
        #https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
        #We have a tensor of size [None, embedding, n_words]
        #We convert to [n_words, None, embedding] in order to do this
        x_1 =  tf.transpose(inputs[0], [2, 0, 1])
        # Reshaping to (n_steps*batch_size, n_input)
        x_1 = tf.reshape(x_1, [-1, self.FLAGS.embedding_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x_1 = tf.split(x_1, self.FLAGS.max_paragraph_size, 0)
        _, HP = tf.contrib.rnn.static_rnn(cell, x_1, dtype=tf.float64)
        # I want this to be size "self.size" by Passage

        cell_2 = tf.contrib.rnn.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag
        x_2 =  tf.transpose(inputs[0], [2, 0, 1])
        # Reshaping to (n_steps*batch_size, n_input)
        x_2 = tf.reshape(x_2, [-1, self.FLAGS.embedding_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x_2 = tf.split(x_2, self.FLAGS.max_question_size, 0)
        _, HQ = tf.contrib.rnn.static_rnn(cell_2, x_2, dtype=tf.float64)
        # I want this to be size "self.size" by Question


        ### Calculate equation 2, https://arxiv.org/pdf/1608.07905.pdf
        ### The equation is kind of weird because of h arrow, so this calculation will
        ### involve the LSTM, and the LSTM's states will become H^r right
        WQ = tf.Variable(tf.zeros([self.size,self.size])) #l by l (aka self.size by self.size)
        WP = tf.Variable(tf.zeros([self.size,self.size])) #l by l (aka self.size by self.size)
        WR = tf.Variable(tf.zeros([self.size,self.size])) #l by l (aka self.size by self.size)
        bP = tf.Variable(tf.zeros([self.size])) #l (aka self.size)
        w = tf.Variable(tf.zeros([self.size])) #l (aka self.size)

        term_1 = tf.matmul(WQ,HQ)


        ### Calculate everything we just did but backwards (should be pretty much the same code)
        ### Doesn't initialize new variables because they are reused


        ### Append the two things calculated above into H^R
        ### Return H^R (Or multiple H^R if handling batching)

        #Encode paragraphs
        return inputs

class Decoder(object):
    def __init__(self, output_size, FLAGS):
        self.output_size = output_size
        self.FlAGS = FLAGS

    def decode(self, knowledge_rep):    # Answer Pointer Layer
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        # We need to calculate equations 7 and 8 from the paper to get Beta_k variables
        # I imagine this code will look similar to the encoding part where we had to 
        # do equation 2 from the paper.

        # Beta_k will be a P length vector, storing a probability for each word (basically
        # will this word be the start word)
        # This is what we need to use to calculate the loss function they give us,
        # and importantly, it's already softmax'd. The loss will need to be changed
        # to be -log(Beta_1_i * Beta_2_j) where i is the actual start token, and j is the actual end token.

        # I think we just need two Beta_k's, one for start and one for end. If
        # we just return these as our predictions that should be the end of this function.

        
        #Terrible start code:
        #Disclaimer: this could all be wrong, some dimensions could be wrong, 
        #but this is my current idea of what the paper looks like

        l = self.FLAGS.state_size

        V = tf.get_variable("V", [l,2*l], initializer=tf.contrib.layers.xavier_initializer())   #Use xavier initialization for weights, zeros for biases
        Wa = tf.get_variable("Wa", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        ba = tf.Variable(tf.zeros([l]), name = "ba")
        v = tf.Variable(tf.zeros([l]), name = "v")
        c = tf.Variable(tf.zeros([1]), name = "c")

        Hr = knowledge_rep
        Hr_tilda = tf.concat([ Hr, tf.zeros([ 1, tf.shape(Hr_tilda)[1] ]) ], axis = 0)

        cell = tf.contrib.rnn.BasicLSTMCell(l) #self.size passed in through initialization from "state_size" flag

        B_k_1 = None
        B_k_2 = None
        state = cell.zero_state()
        for i in xrange(0, 2):
            # just two iterations for the start point and end point
            term2 = Wa*state + ba  # should be an l dimensional vec
            eq = tf.ones()
            term2 = #term2.broadcast across passage length to end up with an l x passage_length matrix where every column is the same
            F_k = tf.tanh(tf.matmul(V,Hr_tilda) + term2)
            B_k_term1 = tf.matmul(v.T, F_k)
            #B_k_term2 = #c.broadcast across passage length to end up with a 1 x passage_length matrix where every element is the same
            if i == 0:
                B_k_1 = tf.softmax(B_k_term1 + B_k_term2)
                _, state = cell.step(tf.matmul(knowledge_rep,B_k_1.T))  
            else:
                B_k_2 = tf.softmax(B_k_term1 + B_k_term2)
                _, state = cell.step(tf.matmul(knowledge_rep,B_k_2.T))


        return ([0.0]*300,[0.0]*300)


class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.FLAGS = FLAGS

        # ==== set up variables ========
        self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False)
        self.global_step = tf.Variable(int(0), trainable = True)

        # ==== set up placeholder tokens ========
        self.paragraph_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        self.question_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_question_size), name="question_placeholder")
        self.start_answer_placeholder = tf.placeholder(tf.int32, (None), name="start_answer_placeholder")
        self.end_answer_placeholder = tf.placeholder(tf.int32, (None), name="end_answer_placeholder")
        self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (None,self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ==
        # params = tf.trainable_variables()

        opt_function = get_optimizer(self.FLAGS.optimizer)  #Default is Adam
        optimizer = opt_function(self.learning_rate)

        # gradients = tf.gradients(self.loss, params)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
        # self.gradient_norms = norm

        # self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]
        self.grad_norm = tf.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)
        grads_and_vars = [(grads[i], variables[i]) for i, v in enumerate(variables)]
        train_op = optimizer.apply_gradients(grads_and_vars)

        self.saver = tf.train.Saver(tf.global_variables())


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        Hr = self.encoder.encode((self.paragraph_embedding, self.question_embedding), self.paragraph_mask_placeholder)
        self.Beta_s, self.Beta_e = self.decoder.decode(Hr)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            p = self.Beta_s[:,self.start_answer_placeholder] * self.Beta_e[:,self.end_answer_placeholder]   #First column is for batches?
            self.loss = -tf.reduce_sum(tf.log(p))

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embed_file = np.load(self.FLAGS.embed_path)
            pretrained_embeddings = embed_file['glove']
            embeddings = tf.Variable(pretrained_embeddings, name = "embeddings")
            self.paragraph_embedding = tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder)
            self.question_embedding = tf.nn.embedding_lookup(embeddings,self.question_placeholder)

    '''
    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        # TODO: Look at evaluate function from evaluate.py
        f1_list = []
        em_list = []
        subdataset = random.sample(dataset, sample)
        for (q,p,a) in subdataset:
            a_s,a_e = self.answer(session,(q,p))
            answer = p[a_s, a_e + 1]
            true_answer = p[true_s, true_e + 1]
            f1_list.append(f1_score(answer, true_answer))
            em_list.append(exact_match_score(answer, true_answer))

        f1 = sum(f1_list)/float(len(f1_list) + 10**(-5))
        em = sum(em_list)/float(len(em_list) + 10**(-5))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em
    '''

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}


        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        #print(dataset[:5])
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # For 10 Epochs
        #   Train on Epoch
        #   Evaluate
        #   Save Parameters