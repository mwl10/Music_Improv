# Music_Improv

### train_music.py
This file uses midi (song) files to train a LSTM model for music generation.
You'll need to input the midi files on the command line and give the index of the instrument
you want to make predictions with from the mid file. Otherwise, it will use defaults (Joy Divison riffs).

### predict_music.py
This file contains the code to output a newly generated midi file (a song) from the trained LSTM model.
All you need to input is the weights file when it prompts on the CL, otherwise it will use a default.

