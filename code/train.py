from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained embeddings.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("max_paragraph_size", 300, "The length to cut paragraphs off at")   # As per Frank's histogram
tf.app.flags.DEFINE_integer("max_question_size", 20, "The length to cut question off at")   # As per Frank's histogram

FLAGS = tf.app.flags.FLAGS


#Restores checkpoints, or 
def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("No checkpoints found in " + train_dir)
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def convert_to_vocab_number(filename):
    return_val = []
    if tf.gfile.Exists(filename):
        return_val = []
        with tf.gfile.GFile(filename, mode="rb") as f:
            return_val.extend(f.readlines())
        return_val = [ [int(word) for word in line.strip('\n').split()] for line in return_val]
        return return_val
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def convert_to_vocab_number_except_dont(filename):
    return_val = []
    if tf.gfile.Exists(filename):
        return_val = []
        with tf.gfile.GFile(filename, mode="rb") as f:
            return_val.extend(f.readlines())
        return_val = [ [word for word in line.strip('\n').split()] for line in return_val]
        return return_val
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def pad_inputs(data, max_length):
    padded_data = []
    mask_data = []
    for data in data:
        length = len(data)
        if length >= max_length:
            padded_data.append(data[:max_length])
            mask_data.append([1]*max_length)
        else:
            pad_length = max_length-length
            padded_data.append(data + [0]*pad_length)
            mask_data.append([1]*length + [0]*pad_length)
    return padded_data, mask_data

def get_dataset():

    train_questions_path = os.path.join(FLAGS.data_dir, "train.ids.question")
    train_answer_path = os.path.join(FLAGS.data_dir, "train.answer")
    train_span_path = os.path.join(FLAGS.data_dir, "train.span")
    train_context_path = os.path.join(FLAGS.data_dir, "train.ids.context")
    val_questions_path = os.path.join(FLAGS.data_dir, "val.ids.question")
    val_answer_path = os.path.join(FLAGS.data_dir, "val.answer")
    val_span_path = os.path.join(FLAGS.data_dir, "val.span")
    val_context_path = os.path.join(FLAGS.data_dir, "val.ids.context")

    train_questions = convert_to_vocab_number(train_questions_path)
    train_span = convert_to_vocab_number(train_span_path)
    train_context = convert_to_vocab_number(train_context_path)
    val_questions = convert_to_vocab_number(val_questions_path)
    val_span = convert_to_vocab_number(val_span_path)
    val_context = convert_to_vocab_number(val_context_path)

    train_answer = convert_to_vocab_number_except_dont(train_answer_path)
    val_answer = convert_to_vocab_number_except_dont(val_answer_path)

    train_questions_padded, train_questions_mask = pad_inputs(train_questions, FLAGS.max_question_size)
    train_context_padded, train_context_mask = pad_inputs(train_context, FLAGS.max_paragraph_size)
    val_questions_padded, val_questions_mask = pad_inputs(val_questions, FLAGS.max_question_size)
    val_context_padded, val_context_mask = pad_inputs(val_context, FLAGS.max_paragraph_size)

    return {"train_questions": train_questions_padded,
            "train_questions_mask":train_questions_mask,
            "train_context": train_context_padded,
            "train_context_mask":train_context_mask,
            "train_answer": train_answer,
            "train_span": train_span,
            "val_questions": val_questions_padded,
            "val_questions_mask":val_questions_mask,
            "val_context": val_context_padded,
            "val_context_mask":val_context_mask,
            "val_answer": val_answer,
            "val_span": val_span}

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = get_dataset()

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    FLAGS.embed_path = embed_path
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, FLAGS=FLAGS)
    decoder = Decoder(output_size=FLAGS.output_size, FLAGS=FLAGS)

    qa = QASystem(encoder, decoder, FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
