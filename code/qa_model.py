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

class MatchLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Extension of LSTM cell to do matching and magic. Designed to be fed to dynammic_rnn
    """
    def __init__(self, hidden_size, HQ, term1, mats, vecs, FLAGS):
        self.HQ = HQ
        self.term1 = term1
        self.hidden_size = hidden_size
        self.FLAGS = FLAGS
        self.mats = mats
        self.vecs = vecs
        super(MatchLSTMCell, self).__init__(hidden_size)

    def __call__(self, inputs, state, scope = None):
        """
        inputs: a batch representation (HP at each word i) that is inputs = hp_i and are [None, l]
        state: a current state for our cell which is LSTM so its a tuple of (c_mem, h_state), both are [None, l]
        """
        
        #For naming convention load in from self the params and rename
        term1 = self.term1
        WQ, WP, WR = self.mats
        bP, w, b = self.vecs
        l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size
        HQ = self.HQ
        hr = state[1]
        hp_i = inputs

        # Check correct input dimensions
        assert hr.get_shape().as_list() == [None, l]
        assert hp_i.get_shape().as_list() == [None, l]

        # Way to extent a [None, l] matrix by dim Q (kinda a hack)
        term2 = tf.matmul(hp_i,WP) + tf.matmul(hr, WR) + bP
        term2 = tf.transpose(tf.stack([term2 for _ in range(Q)]), [1,0,2])

        # Check correct term dimensions for use
        assert term1.get_shape().as_list() == [None, Q, l]
        assert term2.get_shape().as_list() == [None, Q, l]

        # Yeah pretty sure we need this lol
        G_i = tf.tanh(term1 + term2)

        # Reshape to multiply against w
        G_i_shaped = tf.reshape(G_i, [-1, l])
        a_i = tf.matmul(G_i_shaped, w) + b
        a_i = tf.reshape(a_i, [-1, Q, 1])

        # Check that the attention matrix is properly shaped (3rd dim useful for batch_matmul in next step)
        assert a_i.get_shape().as_list() == [None, Q, 1]

        # Prepare dims, and mult attn with question representation in each element of the batch
        HQ_shaped = tf.transpose(HQ, [0,2,1])
        z_comp = tf.batch_matmul(HQ_shaped, a_i)
        z_comp = tf.squeeze(z_comp, [2])

        # Check dims of above operation
        assert z_comp.get_shape().as_list() == [None, l]

        # Concatenate elements for feed into LSTM
        z_i = tf.concat(1,[hp_i, z_comp])

        # Check dims of LSTM input
        assert z_i.get_shape().as_list() == [None, 2*l]

        # Return resultant hr and state from super class (BasicLSTM) run with z_i as input and current state given to our cell
        hr, state = super(MatchLSTMCell, self).__call__(z_i, state)
        return hr, state

# class DecodeLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
#     """
#     Extension of LSTM cell to do decoding and magic. Designed to be fed to dynammic_rnn
#     """
#     def __init__(self, hidden_size, HR, mats, vecs, FLAGS):
#         self.HR = HR
#         self.hidden_size = hidden_size
#         self.FLAGS = FLAGS
#         self.mats = mats
#         self.vecs = vecs
#         super(DecodeLSTMCell, self).__init__(hidden_size)

#     def __call__(self, inputs, state, scope = None):
#         """
#         inputs: a batch representation (HP at each word i) that is inputs = hp_i and are [None, l]
#         state: a current state for our cell which is LSTM so its a tuple of (c_mem, h_state), both are [None, l]
#         """
        
#         #For naming convention load in from self the params and rename
#         V, Wa = self.mats
#         ba, v, c = self.vecs
#         l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size
#         Hr = self.HR
#         hk = state[1]

#         # Check correct input dimensions
#         assert hk.get_shape().as_list() == [None, l] 
#         assert inputs.get_shape().as_list() == [None, 1] 

#         term2 = tf.matmul(hk,Wa) + ba 
#         term2 = tf.transpose(tf.stack([term2 for _ in range(P)]), [1,0,2]) 
#         assert term2.get_shape().as_list() == [None, P, l] 
        
#         Hr_shaped = tf.reshape(Hr, [-1, 2*l])
#         term1 = tf.matmul(Hr_shaped, V)
#         term1 = tf.reshape(term1, [-1, P, l])
#         assert term1.get_shape().as_list() == [None, P, l] 

#         Fk = tf.tanh(term1 + term2)
#         assert Fk.get_shape().as_list() == [None, P, l] 

#         Fk_shaped = tf.reshape(Fk, [-1, l])
#         beta_term = tf.matmul(Fk_shaped, v) + c
#         beta_term = tf.reshape(beta_term ,[-1, P, 1])
#         assert beta_term.get_shape().as_list() == [None, P, 1] 

#         beta = tf.nn.softmax(beta_term)
#         assert beta.get_shape().as_list() == [None, P, 1] 

#         Hr_shaped_cell = tf.transpose(Hr, [0, 2, 1])
#         cell_input = tf.squeeze(tf.batch_matmul(Hr_shaped_cell, beta), [2])
#         assert cell_input.get_shape().as_list() == [None, 2*l] 

#         hk, state = super(DecodeLSTMCell, self).__call__(cell_input, state)

#         beta_return = tf.squeeze(beta [2])
#         return hk, state

class Encoder(object):
    def __init__(self, size, vocab_dim, FLAGS):
        self.size = size
        self.vocab_dim = vocab_dim
        self.FLAGS = FLAGS

    def encode(self, input_question, input_paragraph, question_length, paragraph_length, encoder_state_input = None):    # LSTM Preprocessing and Match-LSTM Layers
        """
        Description:
        """

        assert input_question.get_shape().as_list() == [None, self.FLAGS.max_question_size, self.FLAGS.embedding_size]
        assert input_paragraph.get_shape().as_list() == [None, self.FLAGS.max_paragraph_size, self.FLAGS.embedding_size]

        #Preprocessing LSTM
        with tf.variable_scope("question_encode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag
            HQ, _ = tf.nn.dynamic_rnn(cell, input_question, sequence_length = question_length,  dtype = tf.float32)

        with tf.variable_scope("paragraph_encode"):
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            HP, _ = tf.nn.dynamic_rnn(cell2, input_paragraph, sequence_length = paragraph_length, dtype = tf.float32)   #sequence length masks dynamic_rnn

        assert HQ.get_shape().as_list() == [None, self.FLAGS.max_question_size, self.FLAGS.state_size]
        assert HP.get_shape().as_list() == [None, self.FLAGS.max_paragraph_size, self.FLAGS.state_size]

        # Encoding params
        l = self.size
        Q = self.FLAGS.max_question_size
        P = self.FLAGS.max_paragraph_size

        # Uniform distribution, as opposed to xavier, which is normal
        WQ = tf.get_variable("WQ", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0)) 
        WP = tf.get_variable("WP", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        WR = tf.get_variable("WR", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))

        bP = tf.Variable(tf.zeros([1, l]))
        w = tf.Variable(tf.zeros([l,1])) 
        b = tf.Variable(tf.zeros([1,1]))

        # Calculate term1 by resphapeing to l
        HQ_shaped = tf.reshape(HQ, [-1, l])
        term1 = tf.matmul(HQ_shaped, WQ)
        term1 = tf.reshape(term1, [-1, Q, l])

        # Initialize forward and backward matching LSTMcells with same matching params
        with tf.variable_scope("forward"):
            cell_f = MatchLSTMCell(l, HQ, term1, (WQ, WP, WR), (bP, w, b), self.FLAGS) 
        with tf.variable_scope("backward"):
            cell_b = MatchLSTMCell(l, HQ, term1, (WQ, WP, WR), (bP, w, b), self.FLAGS)

        # Calculate encodings for both forward and backward directions
        (HR_right, HR_left), _ = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, HP, sequence_length = paragraph_length, dtype = tf.float32)
        
        ### Append the two things calculated above into H^R
        HR = tf.concat(2,[HR_right, HR_left])
        assert HR.get_shape().as_list() == [None, P, 2*l]
        print("HR dims: " + str(HR.get_shape().as_list()))
        return HR

class Decoder(object):
    def __init__(self, output_size, FLAGS):
        self.output_size = output_size
        self.FLAGS = FLAGS

    def decode(self, knowledge_rep, paragraph_mask): 
        """
        takes in a knowledge representation  and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        # Decode Params
        l = self.FLAGS.state_size
        P = self.FLAGS.max_paragraph_size
        Hr = knowledge_rep  

        # Decode variables
        V = tf.get_variable("V", [2*l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))   
        Wa = tf.get_variable("Wa", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        ba = tf.Variable(tf.zeros([1,l]), name = "ba")
        v = tf.Variable(tf.zeros([l,1]), name = "v")
        c = tf.Variable(tf.zeros([1]), name = "c")
       
        # Basic LSTM for decoding
        cell = tf.nn.rnn_cell.BasicLSTMCell(l)

        # Preds[0] for predictions start span, and Preds[1] for end of span
        preds = [None, None]
        
        # This is a dumb hack fix to get the inital mem and state to be [None, l] of zeros
        hk = Hr[:,:l,1]*0
        cell_state = (hk, hk)
        assert hk.get_shape().as_list() == [None, l] 

        # Just two iterations of decoding for the start point and then the end point
        for i, _ in enumerate(preds):  
            if i > 0: #Round 2 should reuse variables from before
                tf.get_variable_scope().reuse_variables()

            # Mult and extend using hack to get shape compatable
            term2 = tf.matmul(hk,Wa) + ba 
            term2 = tf.transpose(tf.stack([term2 for _ in range(P)]), [1,0,2]) 
            assert term2.get_shape().as_list() == [None, P, l] 
            
            # Reshape and matmul
            Hr_shaped = tf.reshape(Hr, [-1, 2*l])
            term1 = tf.matmul(Hr_shaped, V)
            term1 = tf.reshape(term1, [-1, P, l])
            assert term1.get_shape().as_list() == [None, P, l] 

            # Add terms and tanh them
            Fk = tf.tanh(term1 + term2)
            assert Fk.get_shape().as_list() == [None, P, l] 

            # Generate beta_term v^T * Fk + c * e(P)
            Fk_shaped = tf.reshape(Fk, [-1, l])
            beta_term = tf.matmul(Fk_shaped, v) + c
            beta_term = tf.reshape(beta_term ,[-1, P, 1])
            assert beta_term.get_shape().as_list() == [None, P, 1] 

            # Get Beta (prob dist over the paragraph)
            beta = tf.nn.softmax(beta_term)
            assert beta.get_shape().as_list() == [None, P, 1] 

            # Setup input to LSTM
            Hr_shaped_cell = tf.transpose(Hr, [0, 2, 1])
            cell_input = tf.squeeze(tf.batch_matmul(Hr_shaped_cell, beta), [2])
            assert cell_input.get_shape().as_list() == [None, 2*l] 

            # Ouput and State for next iteration
            hk, cell_state = cell(cell_input, cell_state)

            #Save a 2D rep of Beta as output
            preds[i] = tf.squeeze(beta, [2])

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
        self.paragraph_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        self.question_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_question_size), name="question_placeholder")
        self.start_answer_placeholder = tf.placeholder(tf.int32, (None), name="start_answer_placeholder")
        self.end_answer_placeholder = tf.placeholder(tf.int32, (None), name="end_answer_placeholder")
        self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (None, self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
        self.paragraph_length = tf.placeholder(tf.int32, (None), name="paragraph_length")
        self.question_length = tf.placeholder(tf.int32, (None), name="question_length")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

        # ==== set up placeholder tokens ======== 2d
        # self.paragraph_placeholder = tf.placeholder(tf.int32, (self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        # self.question_placeholder = tf.placeholder(tf.int32, (self.FLAGS.max_question_size), name="question_placeholder")
        # self.start_answer_placeholder = tf.placeholder(tf.int32, (), name="start_answer_placeholder")
        # self.end_answer_placeholder = tf.placeholder(tf.int32, (), name="end_answer_placeholder")
        # self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
        # self.paragraph_length = tf.placeholder(tf.int32, ([1]), name="paragraph_length")
        # self.question_length = tf.placeholder(tf.int32, ([1]), name="question_length")
        # self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

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
        ps, pe = self.decoder.decode(Hr, self.paragraph_mask_placeholder)
        self.pred_s = tf.boolean_mask(ps, self.paragraph_mask_placeholder)     # For loss
        self.pred_e = tf.boolean_mask(pe, self.paragraph_mask_placeholder)     # For loss
        self.Beta_s = tf.nn.softmax(self.pred_s)   # For decode
        self.Beta_e = tf.nn.softmax(self.pred_e)   # For decode

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
            l1 = sparse_softmax_cross_entropy_with_logits(self.pred_s, self.start_answer_placeholder)
            l2 = sparse_softmax_cross_entropy_with_logits(self.pred_e, self.end_answer_placeholder)
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
            self.paragraph_embedding = tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder)
            self.question_embedding = tf.nn.embedding_lookup(embeddings,self.question_placeholder)

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

        output_feed = [self.Beta_s, self.Beta_e]    # Get the softmaxed outputs

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
        num_data = 1

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
                    num_complete = int(20*float(i+1)/num_data)
                    sys.stdout.write('\r')
                    sys.stdout.write("EPOCH: %d ==> (Loss:%f) [%-20s] (Completion:%d/%d)" % (cur_epoch + 1, mean_loss,'='*num_complete, i+1, num_data))
                    sys.stdout.flush()
            sys.stdout.write('\n')

            self.evaluate_answer(session, small_data, rev_vocab, sample=1, log=True)

            #Save model after each epoch
            checkpoint_path = os.path.join(train_dir, model_name, start_time,"model.ckpt")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = saver.save(session, checkpoint_path)
            print("Model saved in file: %s" % save_path)


