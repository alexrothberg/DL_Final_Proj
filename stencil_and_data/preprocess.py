import numpy as np
import tensorflow as tf
import numpy as np
import pretty_midi
import os

##########DO NOT CHANGE#####################
PAD_TOKEN = 250
STOP_TOKEN = 250
START_TOKEN = 250
UNK_TOKEN = 250
FRENCH_WINDOW_SIZE = 2000
ENGLISH_WINDOW_SIZE = 2000
EXTRA_PADDING = 0
##########DO NOT CHANGE#####################

#Load MIDI file into PrettyMIDI object
def extract_pitches_vels_durations(filename):

	midi_data = pretty_midi.PrettyMIDI(filename)

	#Extract Piano
	piano_midi = midi_data.instruments[0]

	#Extract pitches in a list
	pitches = list(map(lambda x: x.pitch, piano_midi.notes))

	#Extract velocities in a list
	vels = list(map(lambda x: x.velocity, piano_midi.notes))
	durations = list(map(lambda x: x.end - x.start, piano_midi.notes))
	starts = list(map(lambda x: x.start, piano_midi.notes))
	ends = list(map(lambda x: x.end, piano_midi.notes))
	tempo = midi_data.estimate_tempo()

	return (pitches, vels, durations, starts, ends, tempo)

#Takes in a folder that contains midi files
#returns a tuple of (list of pitches list, list of velocities list, list of durations list)
def extract_pitch_vel_duration_lists_for_folder(folder):
	pitches_list = []
	vels_list = []
	durations_list = []
	starts = []
	ends = []
	tempos = []

	song_num = 10
	counter=0
	for song in os.listdir(folder):
		print(song)

		if 'mid' in song:
			song_data = extract_pitches_vels_durations(folder + '/' + song)
			pitches_list.append(song_data[0])
			vels_list.append(song_data[1])
			durations_list.append(song_data[2])
			starts.append(song_data[3])
			ends.append(song_data[4])
			tempos.append(song_data[5])

			if(counter==song_num):
				#print("added a return for testing in preprocess line 58, delete line below")
				#return (pitches_list, vels_list, durations_list, starts, ends, tempos)
				pass
			else:
				counter+=1

	return (pitches_list, vels_list, durations_list, starts, ends, tempos)

def split_train_test(lst_of_data):

	train_index = int(0.8 * len(lst_of_data))
	return (lst_of_data[0:train_index], lst_of_data[train_index:])

def preprocessing(folder, extra_padding = 0):
	EXTRA_PADDING = extra_padding
	all_data = extract_pitch_vel_duration_lists_for_folder(folder)
	padded_pitches_data, padded_vels_data = pad_corpus(all_data[0], all_data[1])
	durations = split_train_test(all_data[2])
	starts = split_train_test(all_data[3])
	tempo = split_train_test(all_data[5])
	pitches_train, pitches_test = split_train_test(padded_pitches_data)
	vels_train, vels_test = split_train_test(padded_vels_data)

	return (vels_train, vels_test, pitches_train, pitches_test, durations, starts, tempo, 250)

def recreate_song_from_data(pitches, vels, starts, durs, tempo, name):

	recreated_song = pretty_midi.PrettyMIDI(initial_tempo = tempo)

	#Create an Instrument instance for a piano instrument:
	song_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(prorgam=song_program)

	#Iterate over note names, which will be converted to note number later:
	for i in range(0, len(vels)):
		note = pretty_midi.Note(velocity=vels[i], pitch=pitches[i], start=start[i], end=starts[i] + durs[i])
		piano.notes.append(note)

	recreated_song.instruments.append(piano)

	#Write out the MIDI data:
	recreated_song.write(name + '.mid')


def pad_corpus(french, english):
    """
    DO NOT CHANGE:

    arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param french: list of French sentences
    :param english: list of English sentences
    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English, French sentence lengths, English sentence lengths)
    """
    FRENCH_padded_sentences = []
    FRENCH_sentence_lengths = []
    for line in french:
        padded_FRENCH = [PAD_TOKEN]*EXTRA_PADDING+ line[:FRENCH_WINDOW_SIZE-1]
        padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
        FRENCH_padded_sentences.append(padded_FRENCH)

    ENGLISH_padded_sentences = []
    ENGLISH_sentence_lengths = []
    for line in english:
        padded_ENGLISH = [PAD_TOKEN]*EXTRA_PADDING+line[:ENGLISH_WINDOW_SIZE-1]
        padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
        ENGLISH_padded_sentences.append(padded_ENGLISH)

    return FRENCH_padded_sentences, ENGLISH_padded_sentences


def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

  Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text


def get_data(french_training_file, english_training_file, french_test_file, english_test_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param french_training_file: Path to the french training file.
	:param english_training_file: Path to the english training file.
	:param french_test_file: Path to the french test file.
	:param english_test_file: Path to the english test file.

	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	french vocab (Dict containg word->index mapping),
	english vocab (Dict containg word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""

	frenchTrainingSentences = read_data(french_training_file)
	englishTrainingSentences = read_data(english_training_file)

	frenchTestingSentences = read_data(french_test_file)
	englishTestingSentences = read_data(english_test_file)

	paddedFrenchTraining, paddedEnglishTraining = pad_corpus(frenchTrainingSentences, englishTrainingSentences)

	paddedFrenchTesting, paddedEnglishTesting = pad_corpus(frenchTestingSentences, englishTestingSentences)

	frenchVocab, frenchTrainIndex = build_vocab(paddedFrenchTraining)

	englishVocab, englishTrainIndex = build_vocab(paddedEnglishTraining)

	englishTrainIDs = convert_to_id(englishVocab, paddedEnglishTraining)
	englishTestIDS = convert_to_id(englishVocab, paddedEnglishTesting)

	frenchTrainIDs = convert_to_id(frenchVocab, paddedFrenchTraining)
	frenchTestIDs = convert_to_id(frenchVocab, paddedFrenchTesting)

	return englishTrainIDs, englishTestIDS, frenchTrainIDs, frenchTestIDs, englishVocab, frenchVocab, englishTrainIndex
