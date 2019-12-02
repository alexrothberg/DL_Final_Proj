import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys


def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 

	for i in range(len(train_french) // (model.batch_size)):
		print(i * model.batch_size)
		print(train_english)
		
		englishTrainInputs = train_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , :-1]
		englishTrainLabels = train_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , 1:]

		frenchTrainInputs = train_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]

		maskNumpy = englishTrainLabels != eng_padding_index
		maskNumpy = tf.cast(maskNumpy, dtype=tf.float32)

		with tf.GradientTape() as tape:
			predictions = model.call(frenchTrainInputs, englishTrainInputs)

			loss = model.loss_function(predictions, englishTrainLabels, maskNumpy)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return None

def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!

	loss = 0
	accuracy = 0
	allPredictions = 0
	predictions = 0

	for i in range(len(test_french) // model.batch_size):

		englishTestInputs = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size),:-1]
		englishTestLabels = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size),1:]

		frenchTestInputs = test_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]

		maskNumpy = englishTestLabels != eng_padding_index
		maskNumpy = tf.cast(maskNumpy, dtype=tf.float32)

		probabilities = model.call(frenchTestInputs, englishTestInputs)

		loss = (model.loss_function(probabilities, englishTestLabels, maskNumpy)) + loss
		allPredictions = (tf.reduce_sum(tf.cast(maskNumpy, tf.float32))) + allPredictions

		accuracy = model.accuracy_function(probabilities, englishTestLabels, maskNumpy)
		predictions = (accuracy * tf.reduce_sum(tf.cast(maskNumpy, tf.float32))) + predictions


	perplexity = np.exp(loss / allPredictions)
	totalAccuracy = predictions / allPredictions

	return perplexity, totalAccuracy

def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	print("Running preprocessing...")
	#train_english,test_english, train_french,test_french, english_vocab,french_vocab,eng_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
	vels_train, vels_test, pitches_train, pitches_test, durations, starts, tempo, index = preprocessing('../jazz')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE,len(pitches_train),ENGLISH_WINDOW_SIZE, len(vels_train))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	#elif sys.argv[1] == "TRANSFORMER":
		#model = Transformer_Seq2Seq(*model_args) 
	
	train(model, pitches_train, vels_train, -50)

	perplexity, accuracy = test(model, pitches_test, vels_test, -50)

	print(perplexity)
	print(accuracy)

if __name__ == '__main__':
   main()


