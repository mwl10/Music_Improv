import numpy as np
from train_music import prep_sequences
from music21 import instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
import pickle

# --------------------------------------------------------------------------------------------
'''
This file contains the code to output a newly generated midi file (a song) from the trained LSTM model 
all you need to input is the weights file when it prompts on the CL, otherwise it will use the default.
'''
# --------------------------------------------------------------------------------------------
""" prepare the prediction model w/ the weights loaded """
def create_model(input, n_vocab, weights): 
	model = Sequential()
	model.add(LSTM(512, input_shape=(input.shape[1], input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
	model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
	model.add(LSTM(512))
	model.add(BatchNorm())
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(BatchNorm())
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.load_weights("weights.hdf5")
	return model

# --------------------------------------------------------------------------------------------	
""" outputs the notes as strings in a list starting w/ a randomly indexed sequence from the input """
def create_notes(input, model, n_vocab, pitchnames):
	start = np.random.randint(0, len(input) -1)
	int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
	pattern = input[start]
	pred_output = []
	for note_index in range(500):
		# make it 1,length of the sequence (los),1 instead of los,1
		pred_input = np.reshape(pattern, (1,len(pattern), 1))
        # normalize input
		pred_input = pred_input / float(n_vocab)
        # predict w/ model
		pred = model.predict(pred_input, verbose=0)
        # get the most likely index of the next note
		index = np.argmax(pred)
        # convert to note form 
		result = int_to_note[index]
        # append the result to the output
		pred_output.append(result)
		pattern = np.append(pattern,index)
		pattern = pattern[1:len(pattern)]
	return pred_output

# --------------------------------------------------------------------------------------------
""" translates the notes to a midi stream & writes the midi to ./output/test_out.mid """
def create_midi(pred_output):
	# translate chords/notes
	offset = 0
	output_notes = []
	for pattern in pred_output:
    	# for chords
		if ('.' in pattern) or pattern.isdigit():
			chord_notes = pattern.split('.')
			notes = []
			for cur_note in chord_notes:
				n_note = note.Note(int(cur_note))
				n_note.storeInstrument = instrument.Bass()
				notes.append(n_note)
			n_chord = chord.Chord(notes)
			n_chord.offset = offset
			output_notes.append(n_chord)
		# for notes (mostly all w/ the bass)
		else:
			n_note = note.Note(pattern)
			n_note.offset = offset
			n_note.storedInstrument = instrument.Bass()
			output_notes.append(n_note)
		# increase offset 
		offset+=0.5
	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp='output/test_out.mid')

# --------------------------------------------------------------------------------------------
""" put it all together """
def predict(weights = 'weights.hdf5'):
	# need notes
	with open('data/notes.pickle', 'rb') as f:
		notes = pickle.load(f)

	pitchnames = sorted(set(notes))

	input, output, n_vocab = prep_sequences(notes, sequence_length = 30)
	model = create_model(input, n_vocab, weights)

	pred_output = create_notes(input, model, n_vocab, pitchnames)
	create_midi(pred_output)

# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
	if (weights := input('weights file? \notherwise press enter to use default\n')):
		predict(weights)
		print('\nmusic file written to output/test_out.mid\n')
	else:
		predict()
		print('\nmusic file written to output/test_out.mid\n')

	

