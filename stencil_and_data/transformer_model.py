import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################



		self.batch_size = 100
		self.embedding_size = 60
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		self.englishEmbedding = tf.Variable(tf.random.truncated_normal([self.english_vocab_size, self.embedding_size], stddev=.01, dtype=tf.float32))
		self.frenchEmbedding = tf.Variable(tf.random.truncated_normal([self.french_vocab_size, self.embedding_size], stddev=.01, dtype=tf.float32))
		
		self.frenchPositional = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
		self.englishPositional = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)
		
		self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=False)
	
		self.denseLayer = tf.keras.layers.Dense(units=self.english_vocab_size, activation="softmax")


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		frenchEmbeddings = tf.nn.embedding_lookup(self.frenchEmbedding, encoder_input)
		englishEmbeddings = tf.nn.embedding_lookup(self.englishEmbedding, decoder_input)

		addedPositionalFrench = self.frenchPositional.call(frenchEmbeddings)

		encoderSequence = self.encoder(addedPositionalFrench)

		addedPositionalEnglish = self.englishPositional.call(englishEmbeddings)

		decoderSequence = self.decoder(addedPositionalEnglish, encoderSequence)

		probabilities = self.denseLayer(decoderSequence)
	
		return probabilities

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

		return tf.reduce_sum(mask * tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

