from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np
import os
import io
import json
import sys
import logging

import tensorflow as tf

# from qa_model import Model
from Model import *
from data_batcher import *


########################################################################################
embedding_size = 50


########################################################################################
data_dir = "../Data/"


########################################################################################


_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1

def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print("Looking for model at %s..." % train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    print(ckpt)
    if ckpt and (tf.io.gfile.exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print("There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir)
            session.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


def main():

    # Initialize bestmodel directory
    # bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")
    # Define path for glove vecs
    glove_path = data_dir + "Glove/" + "glove.6B.{}d.txt".format(embedding_size)

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(glove_path, embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = data_dir + "Train/" + "context"
    train_qn_path = data_dir + "Train/" + "question"
    train_ans_path = data_dir + "Train/" + "span"

    dev_context_path = data_dir + "Dev/" + "context"
    dev_qn_path = data_dir + "Dev/" + "question"
    dev_ans_path = data_dir + "Dev/" + "span"

    tf.reset_default_graph()
    qa_model = QAModel(id2word, word2id, emb_matrix)

    train_dir_path = data_dir + "Train/"
    bestmodel_dir = train_dir_path + "best_checkpoint/"

    # Split by mode
    if mode == "train":
        file_handler = logging.FileHandler( train_dir_path + "Logs/" + "log.txt")
        logging.getLogger().addHandler(file_handler)

        # # Make bestmodel dir if necessary
        # if not os.path.exists(bestmodel_dir):
        #     os.makedirs(bestmodel_dir)

        with tf.Session() as sess:

            # Load most recent model
            initialize_model(sess, qa_model,train_dir_path, expect_exists=False)

            # Train
            qa_model.train(train_dir_path , sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)
            # _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)
            
    with tf.Session() as sess:

        # Load best model
        initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

        # Show examples with F1/EM scores
        _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)

        dev_f1, dev_em = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
        print("Dev F1 score: %f, Dev EM score: %f" % (dev_f1, dev_em))


    # elif FLAGS.mode == "official_eval":
    #     if FLAGS.json_in_path == "":
    #         raise Exception("For official_eval mode, you need to specify --json_in_path")
    #     if FLAGS.ckpt_load_dir == "":
    #         raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

    #     # Read the JSON data from file
    #     qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

    #     with tf.Session(config=config) as sess:

    #         # Load model from ckpt_load_dir
    #         initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

    #         # Get a predicted answer for each example in the data
    #         # Return a mapping answers_dict from uuid to answer
    #         answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

    #         # Write the uuid->answer mapping a to json file in root dir
    #         print "Writing predictions to %s..." % FLAGS.json_out_path
    #         with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
    #             f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
    #             print "Wrote predictions to %s" % FLAGS.json_out_path


    # else:
    #     raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    # tf.app.run()
    main()

