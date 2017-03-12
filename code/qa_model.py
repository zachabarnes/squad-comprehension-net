from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
import random
import sys
from datetime import datetime

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

    def encode(self, input_question, input_paragraph, question_length, paragraph_length, encoder_state_input = None):    # LSTM Preprocessing and Match-LSTM Layers
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
        #question_length = tf.reshape(question_length)
        #paragraph_length = tf.expand_dims(paragraph_length, axis = 0)

        #Preprocessing LSTM
        with tf.variable_scope("question_encode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag
            HQ, _ = tf.nn.dynamic_rnn(cell, tf.transpose(input_question,[0,2,1]), sequence_length = question_length,  dtype = tf.float32)
            HQ = tf.transpose(HQ, [0,2,1])
        with tf.variable_scope("paragraph_encode"):
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            HP, _ = tf.nn.dynamic_rnn(cell2, tf.transpose(input_paragraph,[0,2,1]), sequence_length = paragraph_length, dtype = tf.float32)   #sequence length masks dynamic_rnn
            HP = tf.transpose(HP, [0,2,1])

        ### Calculate equation 2, https://arxiv.org/pdf/1608.07905.pdf
        ### The equation is kind of weird because of h arrow, so this calculation will
        ### involve the LSTM, and the LSTM's states will become H^r right
        l = self.size
        Q = self.FLAGS.max_question_size
        P = self.FLAGS.max_paragraph_size
        WQ = tf.get_variable("WQ", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0)) # Uniform distribution, as opposed to xavier, which is normal
        WP = tf.get_variable("WB", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        WR = tf.get_variable("WR", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        bP = tf.Variable(tf.zeros([self.size,1])) #l (aka self.size)
        w = tf.Variable(tf.zeros([self.size,1])) #l (aka self.size)
        b = tf.Variable(tf.zeros([1,1]))

        HQ = tf.squeeze(HQ)
        HP = tf.squeeze(HP)

        term1 = tf.matmul(WQ, HQ)
        HPs = tf.unstack(HP, axis = 1)

        # Forward Match-LSTM
        with tf.variable_scope("right_match_LSTM"):
            cell_r = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            cell_state = cell_r.zero_state(self.FLAGS.batch_size,tf.float32)
            hr = tf.transpose(cell_state[1])
            hrs = []
            for i, hp_i in enumerate(HPs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                hp_i = tf.expand_dims(hp_i, axis=1)
                term2 = tf.matmul(WP,hp_i) + tf.matmul(WR,hr) +bP
                eQ = tf.ones([1, Q]) 
                G_i = tf.nn.tanh(term1 + tf.matmul(term2, eQ))
                a_i = tf.nn.softmax(tf.matmul(tf.transpose(w),G_i) + b*eQ)

                attn = tf.matmul(HQ,tf.transpose(a_i))
                z_i = tf.concat(0,[hp_i, attn])
                hr, cell_state = cell_r(tf.transpose(z_i), cell_state)
                hr = tf.transpose(hr)
                hrs.append(hr)
            HR_right = tf.concat(1,hrs)

        ### Calculate everything we just did but backwards (should be pretty much the same code)
        ### Doesn't initialize new variables because they are reused
        with tf.variable_scope("left_match_LSTM"):
            cell_l = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            cell_state = cell_l.zero_state(self.FLAGS.batch_size,tf.float32)
            hr = tf.transpose(cell_state[1])
            hrs = []
            for i, hp_i in enumerate(reversed(HPs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                hp_i = tf.expand_dims(hp_i, axis=1)
                term2 = tf.matmul(WP,hp_i) + tf.matmul(WR,hr) + bP
                eQ = tf.ones([1, Q]) 
                G_i = tf.nn.tanh(term1 + tf.matmul(term2, eQ))
                a_i = tf.nn.softmax(tf.matmul(tf.transpose(w),G_i) + b*eQ)

                attn = tf.matmul(HQ,tf.transpose(a_i))
                z_i = tf.concat(0,[hp_i, attn])
                hr, cell_state = cell_l(tf.transpose(z_i), cell_state)
                hr = tf.transpose(hr)
                hrs.append(hr)
            HR_left = tf.concat(1,hrs)
        
        ### Append the two things calculated above into H^R
        HR = tf.concat(0,[HR_right,HR_left])
        print("HR dims: " + str(HR.get_shape().as_list()))
        return HR

class Decoder(object):
    def __init__(self, output_size, FLAGS):
        self.output_size = output_size
        self.FLAGS = FLAGS

    def decode(self, knowledge_rep, paragraph_mask):    # Answer Pointer Layer
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
        #paragraph_mask = tf.cast(paragraph_mask, tf.float32)
        l = self.FLAGS.state_size
        P = self.FLAGS.max_paragraph_size

        V = tf.get_variable("V", [l,2*l], initializer=tf.uniform_unit_scaling_initializer(1.0))   #Use xavier initialization for weights, zeros for biases
        Wa = tf.get_variable("Wa", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        ba = tf.Variable(tf.zeros([l,1]), name = "ba")
        v = tf.Variable(tf.zeros([l,1]), name = "v")
        c = tf.Variable(tf.zeros([1]), name = "c")

        Hr = knowledge_rep  #The first (0th) dimension of this will be of size batch_size

        cell = tf.nn.rnn_cell.BasicLSTMCell(l) #self.size passed in through initialization from "state_size" flag

        preds = [None, None]
        cell_state = cell.zero_state(self.FLAGS.batch_size, tf.float32)
        hk = tf.transpose(cell_state[1])
        for i, _ in enumerate(preds):  # just two iterations for the start point and end point
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            Whb = tf.matmul(Wa,hk) + ba  # should be an l dimensional vec
            eP = tf.ones([1, P]) 
            Fk = tf.tanh(tf.matmul(V,Hr) + tf.matmul(Whb, eP))  #Replicate Whb P+1 times
            
            # Bs and Be calculation
            pred = tf.matmul(tf.transpose(v), Fk) + c*eP       # Replicate c P+1 times

            preds[i] = pred    #Softmax doen in loss function
            cell_input = tf.matmul(Hr, tf.transpose(tf.nn.softmax(pred)))
            hk, cell_state = cell(tf.transpose(cell_input), cell_state)
            hk = tf.transpose(hk)

        print("Beta_s Dims:" + str(preds[0].get_shape().as_list()))
        print("Beta_e Dims:" + str(preds[1].get_shape().as_list()))
        return tuple(preds) # Bs, Be [batchsize, paragraph_length]


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
        self.global_step = tf.Variable(int(0), trainable = False, name = "global_step")

        # # ==== set up placeholder tokens ======== 3d (because of batching)
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
        self.paragraph_length = tf.placeholder(tf.int32, ([1]), name="paragraph_length")
        self.question_length = tf.placeholder(tf.int32, ([1]), name="question_length")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ==
        opt_function = get_optimizer(self.FLAGS.optimizer)  #Default is Adam
        optimizer = opt_function(self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())

        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]

        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = self.global_step, name = "apply_clipped_grads")

        self.saver = tf.train.Saver(tf.global_variables())


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        Hr = self.encoder.encode(self.question_embedding, self.paragraph_embedding, self.question_length, self.paragraph_length)
        self.Beta_s, self.Beta_e = self.decoder.decode(Hr, self.paragraph_mask_placeholder)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        # This is what we need to use to calculate the loss function they give us,
        # and importantly, it's already softmax'd. The loss will need to be changed
        # to be -log(Beta_1_i * Beta_2_j) where i is the actual start token, and j is the actual end token.
        '''
        with vs.variable_scope("loss"):
            p = self.Beta_s[:,self.start_answer_placeholder] * self.Beta_e[:,self.end_answer_placeholder]   #First column is for batches?
            self.loss = -tf.reduce_sum(tf.log(p))
        '''
        # I think that these losses are equivalent
        with vs.variable_scope("loss"):
            l1 = sparse_softmax_cross_entropy_with_logits(tf.boolean_mask(self.Beta_s[0,:], self.paragraph_mask_placeholder), self.start_answer_placeholder)
            l2 = sparse_softmax_cross_entropy_with_logits(tf.boolean_mask(self.Beta_e[0,:], self.paragraph_mask_placeholder), self.end_answer_placeholder)
            self.loss = l1 + l2

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embed_file = np.load(self.FLAGS.embed_path)
            pretrained_embeddings = embed_file['glove']
            embeddings = tf.Variable(pretrained_embeddings, name = "embeddings", dtype=tf.float32, trainable = False)
            self.paragraph_embedding = tf.transpose(tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder), [1,0])
            self.question_embedding = tf.transpose(tf.nn.embedding_lookup(embeddings,self.question_placeholder), [1,0])

    def decode(self, session, question, paragraph, question_mask, paragraph_mask):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[self.question_placeholder] = np.array(question)
        input_feed[self.paragraph_placeholder] = np.array(paragraph)
        input_feed[self.paragraph_mask_placeholder] = np.array(paragraph_mask).T
        input_feed[self.paragraph_length] = np.reshape(np.sum(paragraph_mask),[-1])   # Sum and make into a list
        input_feed[self.question_length] = np.reshape(np.sum(question_mask),[-1])    # Sum and make into a list

        output_feed = [self.Beta_s, self.Beta_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, question, paragraph, question_mask, paragraph_mask):

        B_s, B_e = self.decode(session, question, paragraph, question_mask, paragraph_mask)

        a_s = np.argmax(B_s, axis=1)
        a_e = np.argmax(B_e, axis=1)

        return a_s[0], a_e[0]

    def evaluate_answer(self, session, dataset, rev_vocab, sample=100, log=False):
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
        
        #sample_dataset = random.sample(zip(dataset["val_questions"], dataset["val_context"], dataset["val_answer"]), sample)
        our_answers = []
        their_answers = []
        for question, question_mask, paragraph, paragraph_mask, span, true_answer in random.sample(dataset, sample):
            a_s, a_e = self.answer(session, question, paragraph, question_mask, paragraph_mask)
            token_answer = paragraph[a_s : a_e + 1]      #The slice of the context paragraph that is our answer
            
            print(a_s, "\t", span[0])
            print(a_e, "\t", span[1])
            print(token_answer)

            sentence = []
            for token in token_answer:
                word = rev_vocab[token]
                sentence.append(word)

            our_answer = ' '.join(word for word in sentence)
            our_answers.append(our_answer)
            their_answer = ' '.join(word for word in true_answer)
            their_answers.append(their_answer)
            print(their_answer, "\t", our_answer)

        f1 = exact_match = total = 0
        answer_tuples = zip(their_answers, our_answers)
        for ground_truth, prediction in answer_tuples:
            total += 1
            exact_match += exact_match_score(prediction, ground_truth)
            f1 += f1_score(prediction, ground_truth)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, exact_match, sample))
            logging.info("Samples:")
            for i in xrange(min(10, sample)):
                ground_truth, our_answer = answer_tuples[i]
                logging.info("Ground Truth: {}, Our Answer: {}".format(ground_truth, our_answer))

        return f1, exact_match
    

    def optimize(self, session, train_q, train_q_mask, train_p, train_p_mask, train_span):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        start_ans = train_span[0]
        end_ans = train_span[1]

        input_feed[self.question_placeholder] = np.array(train_q)
        input_feed[self.paragraph_placeholder] = np.array(train_p)
        input_feed[self.start_answer_placeholder] = start_ans
        input_feed[self.end_answer_placeholder] = end_ans
        input_feed[self.paragraph_mask_placeholder] = np.array(train_p_mask).T
        input_feed[self.paragraph_length] = np.reshape(np.sum(train_p_mask),[-1])   # Sum and make into a list
        input_feed[self.question_length] = np.reshape(np.sum(train_q_mask),[-1])    # Sum and make into a list
        input_feed[self.dropout_placeholder] = self.FLAGS.dropout


        output_feed = []

        output_feed.append(self.train_op)
        output_feed.append(self.loss)

        _, loss = session.run(output_feed, input_feed)

        return loss

    def train(self, session, dataset, train_dir, rev_vocab):
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
        start_time = "{:%d-%m-%Y_%H:%M:%S}".format(datetime.now())
        model_name = "match-lstm"

        train_data = zip(dataset["train_questions"], dataset["train_questions_mask"], dataset["train_context"], dataset["train_context_mask"], dataset["train_span"], dataset["train_answer"])
        #num_data = len(train_data)
        num_data = 10

        small_data = random.sample(train_data, num_data)
        for i, (q, q_mask, p, p_mask, span, answ) in enumerate(small_data):
            while span[1] >= 300:    # Simply dont process any questions with answers outside of the possible range
                (q, q_mask, p, p_mask, span, answ) = random.choice(train_data)
                small_data[i] = (q, q_mask, p, p_mask, span, answ)

        for cur_epoch in range(self.FLAGS.epochs):
            losses = []
            for i in range(num_data):
                (q, q_mask, p, p_mask, span, answ) = random.choice(small_data)
                #while span[1] >= 300:    # Simply dont process any questions with answers outside of the possible range
                #    (q, q_mask, p, p_mask, span) = random.choice(train_data)

                loss = self.optimize(session, q, q_mask, p, p_mask, span)
                losses.append(loss)

                if i % self.FLAGS.print_every == 0 or i == 0 or i==num_data:
                    mean_loss = sum(losses)/(len(losses) + 10**-7)
                    num_complete = int(20*float(i)/num_data)
                    sys.stdout.write('\r')
                    sys.stdout.write("EPOCH: %d ==> (Loss:%f) [%-20s] (Completion:%d/%d)" % (cur_epoch + 1, mean_loss,'='*num_complete, i, num_data))
                    sys.stdout.flush()
            sys.stdout.write('\n')

            self.evaluate_answer(session, small_data, rev_vocab, sample=5, log=True)

            #Save model after each epoch
            checkpoint_path = os.path.join(train_dir, model_name, start_time,"model.ckpt")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = saver.save(session, checkpoint_path)
            print("Model saved in file: %s" % save_path)

