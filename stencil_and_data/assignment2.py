import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys

WINDOW_SIZE = 20
'''
def generate(model, train_french, eng_padding_index):
	#current_english = tf.convert_to_tensor( [PAD_TOKEN]*model.batch_size

	for i in range(0,2000):

		englishTrainInputs = train_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , :-1]
		frenchTrainInputs = train_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]
		predictions = model.call(frenchTrainInputs, englishTrainInputs)

def generate(model, test_french, test_english, eng_padding_index):

	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:returns: perplexity of the test set, per symbol accuracy on test set


	# Note: Follow the same procedure as in train() to construct batches of data!


	for i in range(len(test_french) // model.batch_size):
		englishTestInputs = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size),:-1]
		englishTestLabels = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size),1:]
		frenchTestInputs = test_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]
		probabilities = model.call(frenchTestInputs, englishTestInputs)
		decoded_symbols = tf.argmax(input=probabilities, axis=2)

	print("original: ")
	print(test_english)

	print("guesses:")
	print(decoded_symbols)
	return decoded_symbols
'''

def generate_2(model, train_french, train_english):
    for i in range(len(train_french) // (model.batch_size)):
        #iterate over songs
        print("english vocab size is "+str(model.english_vocab_size))
        hop_size = 4
        song_len = np.shape(train_english)[1]

        #initialize previous notes (kinda fake)
        big_guessed_notes = np.reshape(train_english[0,:-1], (1, -1) )
        print("original big guessed notes shape is "+str(np.shape(big_guessed_notes)))

        #initiliaze probability of each velocity
        prob_shape = (1, song_len,model.english_vocab_size)
        big_note_probabilities = np.zeros(prob_shape)

        #initiliaze the note array
        bigFrenchInputs =np.reshape(train_french[0,:-1], (1, -1) )

        for hop in range(WINDOW_SIZE,song_len-hop_size,hop_size):

            guessed_notes = big_guessed_notes[:, hop:hop+hop_size]
            frenchTrainInputs = bigFrenchInputs[:, hop:hop+hop_size]
            predictions = model.call(frenchTrainInputs, guessed_notes)


            top_amount = hop
            bottom_amount = song_len-hop-hop_size

            paddings = [ [0,0], [top_amount, bottom_amount], [0,0] ]
            padded_predictions = tf.pad(predictions, paddings, "CONSTANT")
            print("padded prediction shape: "+str(np.shape(padded_predictions)))
            print("big_note_probabilities shape: "+str(np.shape(big_note_probabilities)))

            #update
            big_note_probabilities = np.add(big_note_probabilities,padded_predictions)
            big_guessed_notes = tf.argmax(input=big_note_probabilities, axis=2)
            print("new big guessed notes shape is "+str(np.shape(big_guessed_notes)))
        print("\n\n\n\n\n\n\n\n big guessed notes")
        print(big_guessed_notes)
        for i in range(song_len):
            print(big_guessed_notes[0][i])

        return big_guessed_notes
    return None



def train(model, train_french, train_english, eng_padding_index):
    for i in range(len(train_french) // (model.batch_size)):
        #iterate over songs
        hop_size = 4
        song_len = np.shape(train_english)[1]
        bigEnglishTrainInputs = train_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , :-1]
        bigEnglishTrainLabels = train_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , 1:]
        bigFrenchTrainInputs = train_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]
        print(np.shape(bigFrenchTrainInputs))
        for hop in range(WINDOW_SIZE,song_len-hop_size,hop_size):
            print(np.shape(bigFrenchTrainInputs))
            englishTrainInputs = bigEnglishTrainInputs[:, hop:hop+hop_size]
            englishTrainLabels = bigEnglishTrainLabels[:, hop:hop+hop_size]
            frenchTrainInputs = bigFrenchTrainInputs[:, hop:hop+hop_size]
            maskNumpy = englishTrainLabels != eng_padding_index
            maskNumpy = tf.cast(maskNumpy, dtype=tf.float32)
            with tf.GradientTape() as tape:
                predictions = model.call(frenchTrainInputs, englishTrainInputs)
                loss = model.loss_function(predictions, englishTrainLabels, maskNumpy)
                print(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return None


def test(model, test_french, test_english, eng_padding_index):
    loss = 0
    accuracy = 0
    allPredictions = 0
    predictions = 0
    for i in range(len(test_french) // (model.batch_size)):
        hop_size = 4
        song_len = np.shape(test_french)[1]
        bigEnglishTestInputs = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , :-1]
        bigEnglishTestLabels = test_english[i * model.batch_size : (i * model.batch_size + model.batch_size) , 1:]
        bigFrenchTestInputs = test_french[i * model.batch_size : (i * model.batch_size + model.batch_size),:]

        for hop in range(WINDOW_SIZE,song_len-hop_size,hop_size):
            englishTestInputs = bigEnglishTestInputs[:, hop:hop+hop_size]
            englishTestLabels = bigEnglishTestLabels[:, hop:hop+hop_size]
            frenchTestInputs = bigFrenchTestInputs[:, hop:hop+hop_size]

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


def change_model(model, window_size, learning_rate, embedding_size, vels_train, vels_test, pitches_train, pitches_test):
    WINDOW_SIZE = window_size
    model_args = (WINDOW_SIZE,len(pitches_train),WINDOW_SIZE, len(vels_train))
    model = RNN_Seq2Seq(*model_args)
    model.optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.french_embedding = tf.keras.layers.Embedding(model.french_vocab_size,model.embedding_size,batch_input_shape=[model.batch_size, model.french_window_size])
    model.eng_embedding = tf.keras.layers.Embedding(model.english_vocab_size,model.embedding_size,batch_input_shape=[model.batch_size, model.english_window_size])

def mini_main(model, window_size, learning_rate, embedding_size, vels_train, vels_test, pitches_train, pitches_test):
    change_model(model, window_size, learning_rate, embedding_size, vels_train, vels_test, pitches_train, pitches_test)
    train(model, np.array(pitches_train), np.array(vels_train), -50)
    perplexity, accuracy = test(model, np.array(pitches_test), np.array(vels_test), -50)
    message = "window: "+str(window_size)
    message += ", lr:"+str(learning_rate)
    message += ", emb: "+str(embedding_size)
    message += "-----"
    message += ", perplexity: "+str(perplexity)
    message += ", accuracy: "+str(accuracy)
    print(message)

def hyperparamter_tune(model, vels_train, vels_test, pitches_train, pitches_test):
    windows = [10,20,30,40]
    learning_rates = [.01, 0.005, 0.001]
    embeddings = [30, 50, 80]
    for i in range(len(windows)):
        for k in range(len(learning_rates)):
            for j in range(len(embeddings)):
                mini_main(model, windows[i], learning_rates[j],embeddings[k],vels_train, vels_test, pitches_train, pitches_test)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()
    print("Running preprocessing...")
    vels_train, vels_test, pitches_train, pitches_test, durations, starts, tempo, index = preprocessing('../jazz')
    print("Preprocessing complete.")
    model_args = (WINDOW_SIZE,len(pitches_train),WINDOW_SIZE, len(vels_train))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)

    hyperparamter_tune(model, vels_train, vels_test, pitches_train, pitches_test)
    #only used to get the blank parts
    train(model, np.array(pitches_train), np.array(vels_train), -50)
    perplexity, accuracy = test(model, np.array(pitches_test), np.array(vels_test), -50)
    new_song = generate_2(model, np.array(pitches_train), np.array(vels_train))

    #print("perplexity is "+str(perplexity))
    #print("accuracy is "+str(accuracy))
    #new_song = generate(model, np.array(pitches_test), np.array(vels_test), -50)
    #new_song = generate(model, np.array(pitches_test), np.array(vels_test), -50)



if __name__ == '__main__':
   main()
