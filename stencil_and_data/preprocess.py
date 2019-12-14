import numpy as np
import tensorflow as tf
import numpy as np
import pretty_midi
import os

PAD_TOKEN = 250
STOP_TOKEN = 250
START_TOKEN = 250
CLASSICAL_WINDOW_SIZE = 2000
JAZZ_WINDOW_SIZE = 2000

def extract_pitches_vels_durations(filename):

	midi_data = pretty_midi.PrettyMIDI(filename)

	piano_midi = midi_data.instruments[0]

	pitches = list(map(lambda x: x.pitch, piano_midi.notes))

	vels = list(map(lambda x: x.velocity, piano_midi.notes))
	durations = list(map(lambda x: x.end - x.start, piano_midi.notes))
	starts = list(map(lambda x: x.start, piano_midi.notes))
	ends = list(map(lambda x: x.end, piano_midi.notes))
	tempo = midi_data.estimate_tempo()

	return (pitches, vels, durations, starts, ends, tempo)

def extract_pitch_vel_duration_lists_for_folder(folder):
	pitches_list = []
	vels_list = []
	durations_list = []
	starts = []
	ends = []
	tempos = []

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

	return (pitches_list, vels_list, durations_list, starts, ends, tempos)

def split_train_test(lst_of_data):

	train_index = int(0.8 * len(lst_of_data))
	return (lst_of_data[0:train_index], lst_of_data[train_index:])

def preprocessing(folder):

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

	pitches = list(map(lambda x: max(0,min(x,127)),pitches))
	vels = list(map(lambda x: max(0,min(x,127)),vels))


	song_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=song_program)

	for i in range(0, min(2000,len(vels),len(starts))):
		note = pretty_midi.Note(velocity=max(0,min(int(vels[i]),127)), pitch=int(pitches[i]), start=(starts[i]), end=(starts[i]) + durs[i])
		piano.notes.append(note)

	recreated_song.instruments.append(piano)

	recreated_song.write('gen_songs/' + str(name))


def pad_corpus(classical, jazz):


    classical_padded = []
    classical_lengths = []
    for line in classical:
        padded_classical = line[:CLASSICAL_WINDOW_SIZE-1]
        padded_classical += [STOP_TOKEN] + [PAD_TOKEN] * (CLASSICAL_WINDOW_SIZE - len(padded_classical)-1)
        classical_padded.append(padded_classical)

    jazz_padded = []
    jazz_lengths = []
    for line in jazz:
        padded_jazz = line[:JAZZ_WINDOW_SIZE-1]
        padded_jazz = [START_TOKEN] + padded_jazz + [STOP_TOKEN] + [PAD_TOKEN] * (JAZZ_WINDOW_SIZE - len(padded_jazz)-1)
        jazz_padded.append(padded_jazz)

    return classical_padded, jazz_padded


def extract_pitches_vels_durations_names(filename):

	midi_data = pretty_midi.PrettyMIDI(filename)

	piano_midi = midi_data.instruments[0]

	pitches = list(map(lambda x: x.pitch, piano_midi.notes))

	vels = list(map(lambda x: x.velocity, piano_midi.notes))
	durations = list(map(lambda x: x.end - x.start, piano_midi.notes))
	starts = list(map(lambda x: x.start, piano_midi.notes))
	ends = list(map(lambda x: x.end, piano_midi.notes))
	tempo = midi_data.estimate_tempo()

	return (pitches, vels, durations, starts, ends, tempo,filename)

def extract_pitch_vel_duration_lists_for_folder_post(folder):
	pitches_list = []
	vels_list = []
	durations_list = []
	starts = []
	ends = []
	tempos = []
	names = []

	for song in os.listdir(folder):
		print(song)

		if 'mid' in song:
			song_data = extract_pitches_vels_durations_names(folder + '/' + song)
			pitches_list.append(song_data[0])
			vels_list.append(song_data[1])
			durations_list.append(song_data[2])
			starts.append(song_data[3])
			ends.append(song_data[4])
			tempos.append(song_data[5])
			names.append(song)

	return (pitches_list, vels_list, durations_list, starts, ends, tempos,names)

def post_processing(folder):

	all_data = extract_pitch_vel_duration_lists_for_folder_post(folder)
	padded_pitches_data, padded_vels_data = pad_corpus(all_data[0], all_data[1])
	durations = (all_data[2])
	starts = (all_data[3])
	tempo = all_data[5]
	pitches_train, pitches_test = split_train_test(padded_pitches_data)
	vels_train, vels_test = split_train_test(padded_vels_data)
	names_train,names_test = split_train_test(all_data[6])



	return (vels_train, vels_test, pitches_train, pitches_test, durations, starts, tempo, 250,names_test,names_train)
