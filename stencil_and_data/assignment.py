import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from rnn_model import RNN_Seq2Seq
import sys


def train(model, train_pitch, train_vel, padding_index):

	for i in range(len(train_pitch) // (model.batch_size)):
		
		velTrainInputs = train_vel[i * model.batch_size : (i * model.batch_size + model.batch_size) , :-1]
		velTrainLabels = train_vel[i * model.batch_size : (i * model.batch_size + model.batch_size) , 1:]

		pitchTrainInputs = train_pitch[i * model.batch_size : (i * model.batch_size + model.batch_size),:]

		maskNumpy = velTrainLabels != padding_index
		maskNumpy = tf.cast(maskNumpy, dtype=tf.float32)

		with tf.GradientTape() as tape:
			predictions = model.call(pitchTrainInputs, velTrainInputs)

			loss = model.loss_function(predictions, velTrainLabels, maskNumpy)
			print(loss)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return None

def test(model, test_pitch, test_vel, padding_index):

	loss = 0
	accuracy = 0
	allPredictions = 0
	predictions = 0

	for i in range(len(test_pitch) // model.batch_size):

		velTestInputs = test_vel[i * model.batch_size : (i * model.batch_size + model.batch_size),:-1]
		velTestLabels = test_vel[i * model.batch_size : (i * model.batch_size + model.batch_size),1:]

		pitchTestInputs = test_pitch[i * model.batch_size : (i * model.batch_size + model.batch_size),:]

		maskNumpy = velTestLabels != padding_index
		maskNumpy = tf.cast(maskNumpy, dtype=tf.float32)

		probabilities = model.call(pitchTestInputs, velTestInputs)

		loss = (model.loss_function(probabilities, velTestLabels, maskNumpy)) + loss
		allPredictions = (tf.reduce_sum(tf.cast(maskNumpy, tf.float32))) + allPredictions

		accuracy = model.accuracy_function(probabilities, velTestLabels, maskNumpy)
		predictions = (accuracy * tf.reduce_sum(tf.cast(maskNumpy, tf.float32))) + predictions


	perplexity = np.exp(loss / allPredictions)
	totalAccuracy = predictions / allPredictions

	return perplexity, totalAccuracy


def make_music(model, test_pitch, test_vel, padding_index):

	decoded_symbols = []
	for i in range(len(test_pitch) // model.batch_size):
		velTestInputs = test_vel[i * model.batch_size : (i * model.batch_size + model.batch_size),:-1]
		velTestLabels = test_vel[i * model.batch_size : (i * model.batch_size + model.batch_size),1:]
		pitchTestInputs = test_pitch[i * model.batch_size : (i * model.batch_size + model.batch_size),:]
		probabilities = model.call(pitchTestInputs, velTestInputs)
		decoded_symbols.append(tf.argmax(input=probabilities, axis=2)[0])
		decoded_symbols.append(tf.argmax(input=probabilities, axis=2)[1])

	return decoded_symbols


def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	vels_train, vels_test, pitches_train, pitches_test, durations, starts, tempo, index = preprocessing('../jazz')
	model_args = (CLASSICAL_WINDOW_SIZE,len(pitches_train),JAZZ_WINDOW_SIZE, len(vels_train))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args) 
	
	train(model, np.array(pitches_train), np.array(vels_train), -50)
	train(model, np.array(pitches_train), np.array(vels_train), -50)
	train(model, np.array(pitches_train), np.array(vels_train), -50)


	perplexity, accuracy = test(model, np.array(pitches_test), np.array(vels_test), -50)

	print("****************")
	print(perplexity)
	print(accuracy)
	print("************")

	datums = post_processing('../classical')

	vels_found = make_music(model, np.array(datums[2]), np.array(datums[0]), -50)
	with open('f.txt','w+') as x:
		for i in vels_found[0]:
			x.write(str(int(i)))
			x.write(',')
		x.write(str(np.array(datums[9][0])))

	try:
		print(len(vels_found))
		print(len(vels_found[0]))
		print(datums[9][0])
		print(len(datums[2]))
		print(len(datums[2][0]))


		print(len(datums[6]))
		print(len(datums[6][0]))


	except:
		pass
	try:
		print(vels_found.shape)
	except:
		pass

	print(len(datums[2][0]))
	print(len(      vels_found[0]   ))
	print(len(   datums[5][0]      )) 
	print(len(    datums[4][0]     ))
	print(len(   datums[5][0]      )) 
	print(len(    datums[4][0]     ))
	print(   datums[6][0]     )
	print(len(      datums[9][0]   ))


	recreate_song_from_data((datums[2][0]),vels_found[0],(datums[5][0]),(datums[4][0]),(datums[6][0]),(datums[9][0]))
	for i in range(0,len(vels_found)):
		recreate_song_from_data((datums[2][i]),vels_found[i],(datums[5][i]),(datums[4][i]),(datums[6][i]),(datums[9][i]))

	
	print(perplexity)
	print(accuracy)

if __name__ == '__main__':
   main()


