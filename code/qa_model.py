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
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input = None):
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

        #Encode paragraphs
        return inputs

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
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
        # with vs.variable_scope("answer_start"):
        #     a_s = rnn_cell._linear([h_q, h_p], output_size = self.output_size)
        # with vs.variable_scope("answer_end"):
        #     a_e = rnn_cell._linear([h_q, h_p], output_size = self.output_size)
        # return a_s, a_e

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

        opt_function = get_optimizer(self.FLAGS.optimizer)
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
        hr = self.encoder.encode((self.paragraph_embedding, self.question_embedding), self.paragraph_mask_placeholder)
        self.a_s, self.a_e = self.decoder.decode(hr)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            l1 = sparse_softmax_cross_entropy_with_logits(self.a_s, self.start_answer_placeholder)
            l2 = sparse_softmax_cross_entropy_with_logits(self.a_e, self.end_answer_placeholder)
            self.loss = l1 + l2 #Simple additive loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embed_file = np.load(self.FLAGS.embed_path)
            embeddings = embed_file['glove']
            self.paragraph_embedding = tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder)
            self.question_embedding = tf.nn.embedding_lookup(embeddings,self.question_placeholder)

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
        print(dataset[:5])
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
