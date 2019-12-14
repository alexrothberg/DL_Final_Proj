import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, classical_window_size, classical_size, jazz_window_size, jazz_size):

		super(RNN_Seq2Seq, self).__init__()
		self.classical_size = classical_size
		self.jazz_size = jazz_size

		self.classical_window_size = classical_window_size
		self.jazz_window_size = jazz_window_size

		self.batch_size = 2 
		self.embedding_size = 30 
	
		self.optimizer = tf.keras.optimizers.Adam(.01)
		self.perplexity = 0
		self.test_perp = 0
		self.lstm_enc_out_dim = 35
		self.lstm_dec_out_dim = 35

		self.pitch_embedding = tf.keras.layers.Embedding(classical_size,self.embedding_size,batch_input_shape=[self.batch_size, self.classical_window_size])
		self.vel_embedding = tf.keras.layers.Embedding(jazz_size,self.embedding_size,batch_input_shape=[self.batch_size, self.jazz_window_size])
		self.dense_layer = tf.keras.layers.Dense(activation = 'softmax',units = jazz_size,use_bias=True,bias_initializer='zeros')
		self.encoder = tf.keras.layers.LSTM(self.lstm_enc_out_dim,return_sequences=True,return_state=True)
		self.decoder = tf.keras.layers.LSTM(self.lstm_dec_out_dim,return_sequences=True,return_state=True)


	@tf.function
	def call(self, encoder_input, decoder_input):

		classical_emb = self.pitch_embedding(encoder_input)
		encoded, state1, state2 = self.encoder(classical_emb)
		
		jazz_emb = self.vel_embedding(decoder_input)
		decoded, state_dec_1, state_dec_2 = self.decoder(jazz_emb, initial_state = [state1,state2])
		out = self.dense_layer(decoded)

		return out

	def accuracy_function(self, prbs, labels, mask):

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		return tf.reduce_sum(mask * tf.keras.losses.sparse_categorical_crossentropy(labels,prbs))

