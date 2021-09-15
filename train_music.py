import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import pickle
'''
This file uses midi (song) files to train a LSTM model for music generation.
You'll need to input the midi files on the command line and give the index of the instrument
you want to make predictions with from the mid file. Otherwise, it will use defaults. 
'''
# --------------------------------------------------------------------------------------------
''' 
input:  list of file names
output: a list of strings where chords are a grouping of numerical notes, 
         separated by '.'; notes are the note names themselves 
'''
def file_to_note_str(melody_pos, files):
    notes = []
    for file in files:
        # load file into Music21 stream to get list of all the notes and chords
        try:
        	midi = converter.parse(file)
        except Exception as e:
        	print('probably bad mid/midi file: {}, exiting...\n'.format(file), e)
        	exit()
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        # melodies must be in the same position! (Here the bass lines are in index 2 of the midi)
        try:
        	notes_to_parse = parts.parts[melody_pos].recurse()
        except Exception as e:
        	print('bad index for the midi file instrument part, exiting...\n', e)
        	exit()
        for elem in notes_to_parse:
            if isinstance(elem, note.Note):
                notes.append(str(elem.pitch))
            elif isinstance(elem, chord.Chord):
                # append chords by encoding the id of every note in the chord together in a string separated by a dot
                notes.append('.'.join(str(n) for n in elem.normalOrder))

    # save the notes for prediction
    with open('data/notes.pickle', 'wb') as f:
        pickle.dump(notes, f)
    return notes

# --------------------------------------------------------------------------------------------
''' prepare the input sequences and corresponding output for the rnn '''
def prep_sequences(notes, sequence_length = 50):
    # pitchnames holds all the different notes/chords in a set
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    input= []
    output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])
    num_patterns = len(input)
    input = np.reshape(input, (num_patterns, sequence_length, 1))
    # normalize
    input = input / float(n_vocab)
    # if it errors here, your midi file instrument part is bad,
    try:
    	output = to_categorical(output)
    except Exception as e:
    	print('likely bad midi index for the melody:\n', e)
    	exit()
    print('pitchnames:\n', pitchnames)
    print('number of notes/chords\n', n_vocab)
    return input, output, n_vocab

# --------------------------------------------------------------------------------------------
''' prepare the model ''' 
def create_model(input, n_vocab):
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
    #opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model 

# --------------------------------------------------------------------------------------------
''' 
train the model based on the data and output checkpoints that you can monitor/input to make 
predictions along the way 
''' 
def train(model, input, output, epochs=50, batch_size=64):
    fp = "weight_checkpoints/music/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(fp, monitor='loss', verbose=0,save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(input, output, epochs=50, batch_size=64, callbacks=callbacks_list)

# --------------------------------------------------------------------------------------------
''' whole thang '''
def train_rnn(melody_pos = 2, files = glob.glob("bass_mids/*.MID")):
    notes = file_to_note_str(melody_pos, files)
    input, output, n_vocab = prep_sequences(notes)
    model = create_model(input, n_vocab)
    train(model, input, output)

# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# try to handle some input, and bad input...
	inp = input('provide regular expression for where midi/mid files are, i.e. "music/*.mid", \nor filepaths separated by commas; \notherwise, to set the default as "bass_mids/*.MID", press enter\n')
	if (fps := glob.glob(inp)):
		while True:
			try:
				melody_pos= int(input('provide index position of the melodies in the midi files, default is 2\n'))
				break
			except ValueError:
				print('not a valid number, try again\n')
		train_rnn(melody_pos, fps)	
	else:
		if inp == '':
			print('running default\n')
			train_rnn()
		else:
			print('bad file paths\n')




