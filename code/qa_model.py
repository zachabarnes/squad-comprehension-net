from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

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

    def encode(self, input_question, input_paragraph, masks, encoder_state_input = None):    # LSTM Preprocessing and Match-LSTM Layers
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
        question_shape = input_question.get_shape().as_list()
        paragraph_shape = input_paragraph.get_shape().as_list()
        assert question_shape == [self.FLAGS.embedding_size, self.FLAGS.max_question_size]
        assert paragraph_shape == [self.FLAGS.embedding_size, self.FLAGS.max_paragraph_size]


        ### Right now the way we set everything up, the first dimension of each input is arbitrary
        ### If we don't want to handle batching for now, I'm not exactly what to change but I think
        ### it can be traced back to self.paragraph_placeholder

        input_question = tf.expand_dims(input_question, axis = 0)
        input_paragraph = tf.expand_dims(input_paragraph, axis = 0)

        print (input_question.get_shape().as_list())

        with tf.variable_scope("question_encode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag
            HQ, _ = tf.nn.dynamic_rnn(cell, tf.transpose(input_question,[0,2,1]), dtype = tf.float32)
            HQ = tf.transpose(HQ, [0,2,1])
        with tf.variable_scope("paragraph_encode"):
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            HP, _ = tf.nn.dynamic_rnn(cell2, tf.transpose(input_paragraph,[0,2,1]), dtype = tf.float32)
            HP = tf.transpose(HP, [0,2,1])

        ### Calculate equation 2, https://arxiv.org/pdf/1608.07905.pdf
        ### The equation is kind of weird because of h arrow, so this calculation will
        ### involve the LSTM, and the LSTM's states will become H^r right
        l = self.size
        Q = self.FLAGS.max_question_size
        P = self.FLAGS.max_paragraph_size
        WQ = tf.get_variable("WQ", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        WP = tf.get_variable("WB", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        WR = tf.get_variable("WR", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        bP = tf.Variable(tf.zeros([self.size,1])) #l (aka self.size)
        w = tf.Variable(tf.zeros([self.size,1])) #l (aka self.size)
        b = tf.Variable(tf.zeros([1,1]))

        HQ = tf.squeeze(HQ)
        HP = tf.squeeze(HP)
        print (HQ.get_shape().as_list())

        term_1 = tf.matmul(WQ, HQ)

        HPs = tf.unstack(HP, axis = 1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(200, state_is_tuple=False)
        hr = cell.zero_state(1,tf.float32)
        print(hr)
        hrs = []
        for i, hp_i in enumerate(HPs):
            hp_i = tf.expand_dims(hp_i, axis=1)
            #hr = tf.expand_dims(hr, axis=1)
            term2 = tf.matmul(WP,hp_i) + tf.matmul(WR,hr) +bP
            eQ = tf.ones([1, Q]) 
            G_i = tf.matmul(term2, eQ)
            print(tf.matmul(b,eQ))
            a_i = tf.nn.softmax(tf.matmul(tf.transpose(w),G_i) + b*eQ)
            #print(a_i)

            z_i = tf.concat_v2([hp_i, tf.matmul(HQ,tf.transpose(a_i))],0)
            hr, _ = cell(z_i, hr)
            hrs.append(hr)


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

        l = self.FLAGS.state_size
        P = self.FLAGS.max_paragraph_size

        V = tf.get_variable("V", [l,2*l], initializer=tf.contrib.layers.xavier_initializer())   #Use xavier initialization for weights, zeros for biases
        Wa = tf.get_variable("Wa", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        ba = tf.Variable(tf.zeros([l]), name = "ba")
        v = tf.Variable(tf.zeros([l]), name = "v")
        c = tf.Variable(tf.zeros([1]), name = "c")

        Hr = knowledge_rep  #The first (0th) dimension of this will be of size batch_size

        cell = tf.nn.BasicLSTMCell(l) #self.size passed in through initialization from "state_size" flag

        B = [None, None]
        state = cell.zero_state()
        for i, _ in enumerate(B):  # just two iterations for the start point and end point
            # Fk calculation
            Whb = Wa*state + ba  # should be an l dimensional vec
            eP = tf.ones([1, P]) 
            Fk = tf.tanh(tf.matmul(V,Hr) + tf.matmul(tf.transpose(Whb), eP))  #Replicate Whb P+1 times
            
            # Bs and Be calculation
            B[i] = tf.softmax(tf.matmul(tf.transpose(v), Fk) + tf.matmul(tf.transpose(c), eP))     #Replicate c P+1 times
            _, state = cell.step(tf.matmul(knowledge_rep, tf.transpose(B[i])))  

        return tuple(B) # Bs, Be


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
        self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False, name = "learning_rate")
        self.global_step = tf.Variable(int(0), trainable = True, name = "global_step")

        # # ==== set up placeholder tokens ========
        # self.paragraph_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        # self.question_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_question_size), name="question_placeholder")
        # self.start_answer_placeholder = tf.placeholder(tf.int32, (None), name="start_answer_placeholder")
        # self.end_answer_placeholder = tf.placeholder(tf.int32, (None), name="end_answer_placeholder")
        # self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (None, self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
        # self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

  # ==== set up placeholder tokens ======== 2d
        self.paragraph_placeholder = tf.placeholder(tf.int32, (self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        self.question_placeholder = tf.placeholder(tf.int32, (self.FLAGS.max_question_size), name="question_placeholder")
        self.start_answer_placeholder = tf.placeholder(tf.int32, (), name="start_answer_placeholder")
        self.end_answer_placeholder = tf.placeholder(tf.int32, (), name="end_answer_placeholder")
        self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
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
        Hr = self.encoder.encode(self.question_embedding, self.paragraph_embedding, self.paragraph_mask_placeholder)
        self.Beta_s, self.Beta_e = self.decoder.decode(Hr)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        # This is what we need to use to calculate the loss function they give us,
        # and importantly, it's already softmax'd. The loss will need to be changed
        # to be -log(Beta_1_i * Beta_2_j) where i is the actual start token, and j is the actual end token.

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
            embeddings = tf.cast(embeddings, tf.float32)
            self.paragraph_embedding = tf.transpose(tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder), [1,0])
            self.question_embedding = tf.transpose(tf.nn.embedding_lookup(embeddings,self.question_placeholder), [1,0])

    def decode(self, session, question, paragraph):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed['question_placeholder'] = self.question_placeholder
        input_feed['paragraph_placeholder'] = self.paragraph_placeholder

        output_feed = [self.Beta_s, self.Beta_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, question, paragraph):

        B_s, B_e = self.decode(session, question, paragraph)

        a_s = np.argmax(B_s, axis=1)
        a_e = np.argmax(B_e, axis=1)

        return a_s, a_e

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

        f1_list, em_list = [], []
        temp_dataset = zip(dataset["val_questions"], dataset["val_context"], dataset["val_span"])
        subdataset = random.sample(dataset, sample)
        for question, paragraph, answer in subdataset:
            a_s,a_e = self.answer(session,question,paragraph)
            true_s, true_e = answer[0], answer[1]
            our_answer = p[a_s : a_e + 1]           #The slice of the context paragraph that is our answer
            true_answer = p[true_s : true_e + 1]    #The slice of the context paragraph that is the true answer
            f1_list.append(f1_score(our_answer, true_answer))
            em_list.append(exact_match_score(our_answer, true_answer))

        f1 = sum(f1_list)/float(len(f1_list))
        em = sum(em_list)/float(len(em_list))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em
    

    def optimize(self, session, train_q, train_q_mask, train_p, train_p_mask, train_span):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        start_ans = train_span[0]
        end_ans = train_span[1]

        input_feed['qustion_placeholder'] = np.array(train_q)
        input_feed['paragraph_placeholder'] = np.array(train_p)
        input_feed['start_answer_placeholder'] = start_ans
        input_feed['end_answer_placeholder'] = end_ans
        input_feed['paragraph_mask_placeholder'] = np.array(train_p_mask)
        input_feed['dropout_placeholder'] = self.FLAGS.dropout

        output_feed = []

        output_feed.append(self.train_op)
        output_feed.append(self.loss)

        _, loss = session.run(output_feed, input_feed)

        return loss

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

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        #Info for saving models
        saver = tf.train.Saver()
        start_time = "{:%d/%m/%Y_%H:%M:%S}".format(datetime.now())
        model_name = "match-lstm"

        train_data = zip(dataset["train_questions"], dataset["train_questions_mask"], dataset["train_context"], dataset["train_context_mask"], dataset["train_span"])
        for cur_epoch in range(self.FLAGS.epochs):
            losses = []
            for i in range(train_data):

                (q, q_mask, p, p_mask, a) = random.choice(train_data)

                loss = self.optimize(session, q, q_mask, p, p_mask, span)
                losses.append(loss)
                if i % 100 == 0 and i != 0:
                    mean_loss = sum(losses)/(len(losses) + 10**-7)
                    print("Loss: %s" % mean_loss)

            self.evaluate_answer(session, dataset, sample=100, log=True)

            #Save model after each epoch
            checkpoint_path = os.path.join(train_dir, model_name, start_time,"model.ckpt")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = saver.save(sess,  checkpoint_path)
            print("Model saved in file: %s" % save_path)

