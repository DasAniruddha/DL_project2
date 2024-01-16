## Music Genre Identificaion
### This is project no.2
 In this project the task is to identify the type of music genre for a particular audio file using DL.

 The genres are as follow:
  * Blues:0
  * Classicl:1
  * Country:2
  * Disco:3
  * Hiphop:4
  * Jazz:5
  * Metal:6
  * Pop:7
  * Reggae:8
  * Rock:9

 To use the audio files first we have converted them to a format that can be fed to a deep learning algorith using 
 Audio Spectograms as such:

 ```
 def create_spectogram(filename, name):
  plt.interactive(False)
  clip, sample_rate = librosa.load(filename, sr=None)
  fig = plt.figure(figsize=[0.72,0.72])
  ax = fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
  librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
  filename = '/content/full_data/' + name + '.jpg'
  plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  del filename, name,clip, sample_rate, fig, ax, S
 ```

 And utilizing a sequential network as such:

 ```
 model = Sequential()

model.add(Conv2D(32, (3,3), padding='same',
                 input_shape=(64,64,3)))

model.add(Activation('relu'))

model.add(Conv2D(64, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25)) # initial: 0.25

model.add(Conv2D(64, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))  # initial: 0.5

model.add(Conv2D(128, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))  # initial: 0.5

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))  # initial: 0.5

model.add(Dense(10, activation='softmax'))

model.compile(optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6), # learning_rate was 0.0005
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
 ```