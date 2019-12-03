import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

    ######vvv DO NOT CHANGE vvvv##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 2 # You can change this
		self.embedding_size = 30 # You should change this
	
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.optimizer = tf.keras.optimizers.Adam(.01)
		self.perplexity = 0
		self.test_perp = 0
		self.lstm_enc_out_dim = 35
		self.lstm_dec_out_dim = 35

		self.french_embedding = tf.keras.layers.Embedding(french_vocab_size,self.embedding_size,batch_input_shape=[self.batch_size, self.french_window_size])
		self.eng_embedding = tf.keras.layers.Embedding(english_vocab_size,self.embedding_size,batch_input_shape=[self.batch_size, self.english_window_size])
		# self.french_dense_1 = tf.keras.layers.Dense(activation = 'softmax',units = french_vocab_size,use_bias=True,bias_initializer='zeros')
		self.eng_dense_1 = tf.keras.layers.Dense(activation = 'softmax',units = english_vocab_size,use_bias=True,bias_initializer='zeros')
		self.encoder = tf.keras.layers.LSTM(self.lstm_enc_out_dim,return_sequences=True,return_state=True)
		self.decoder = tf.keras.layers.LSTM(self.lstm_dec_out_dim,return_sequences=True,return_state=True)


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		french_emb = self.french_embedding(encoder_input)
		encoded, state1, state2 = self.encoder(french_emb)
		# OK SO I DON't shift the eng over??
		
		# encoded = self.french_dense_1(encoded)
		eng_emb = self.eng_embedding(decoder_input)
		decoded, state_dec_1, state_dec_2 = self.decoder(eng_emb, initial_state = [state1,state2])
		out = self.eng_dense_1(decoded)
		print(out.shape)

		# TODO:
		#1) Pass your french sentence embeddings to your encoder 
		#2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		#3) Apply dense layer(s) to the decoder out to generate probabilities

		return out

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		# labels = np.multiply(labels,mask)
		# cross_ent = tf.keras.losses.sparse_categorical_crossentropy(labels,prbs,from_logits= False)
		# return (tf.reduce_sum(cross_ent))/(np.sum(mask))
		return tf.reduce_sum(mask * tf.keras.losses.sparse_categorical_crossentropy(labels,prbs))

