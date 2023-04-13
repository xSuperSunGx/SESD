import datetime
import os
import random
import threading
import time
import uuid

import librosa as librosa
import pandas as pd
import pyaudio
import requests
import speech_recognition as sr
import pyautogui
import matplotlib.pyplot as plt
import numpy as np
import openai
from google.cloud import texttospeech
import IPython.display as ipd
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm.auto import tqdm
from google.cloud import texttospeech

credentials = service_account.Credentials.from_service_account_file('hcin-382111-2b36cc7a166f.json')
openai.api_key = "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"
model = Sequential()
labelencoder = LabelEncoder()
p = pyaudio.PyAudio()
# Initialize the recognizer

def get_weather(location, name):
    #Mieders, PLZ 6142
    responsee = None
    prompt = f"Du bist ein Sprecher für eine Wettervorhersagung. Für deine Vorhersagungen benutzt du https://at.wetter.com/. Schreibe kurze, formale und prezise Antworten. Wie ist das Wetter heute in {location}? Benutze dazu das Internet. Nenne mich {name}. Bitte nur die Information wiedergeben."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()




# Define the function to start the app
def start_app(app_name):

        pyautogui.press('win')
        time.sleep(1)
        pyautogui.typewrite(app_name)
        time.sleep(1)
        pyautogui.press('enter')

# Define the function to record and identify spoken audio
def listen_and_start():
    r = sr.Recognizer()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("Available microphone sources:")

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"  [{'{:02d}'.format(i+1)}] ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    output = None
    num = int(input("Select a microphone source: "))
    print(f"Selected microphone source: {p.get_device_info_by_host_api_device_index(0, num-1).get('name')}")
    with sr.Microphone(device_index=p.get_device_info_by_host_api_device_index(0, num-1).get('index')) as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening...")
        audio = r.listen(source)

        try:
            recognized_text = r.recognize_google(audio, language="de-DE")
            print(f"You said ({str(i)}): " + recognized_text)

            if recognized_text.lower().startswith("öffne"):
                start_app(recognized_text.lower().split(" ")[1])
                return
            elif "noa" in recognized_text.lower() and "wetter" in recognized_text.lower():
                output = get_weather(getUserByName("Noah"), "Noah")
            elif "timo" in recognized_text.lower() and "wetter" in recognized_text.lower():
                output = get_weather(getUserByName("Timo"), "Timo")
            else:
                output = "Ich habe dich nicht verstanden."
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        uu = uuid.uuid4()
        synthesize_text_with_audio_profile(output, "temp/" + str(uu) + ".mp3", "medium-bluetooth-speaker-class-device", "de-DE-Neural2-B")
        time.sleep(1)
        for i in tqdm(range(100), desc="Loading", ncols=100, unit_scale=False):
            time.sleep(0.01)
        print("Chatbot: " + output)
        playsound(str(uu) + ".mp3")


def playsound(filename):
    from playsound import playsound

    filepath = os.path.join(os.getcwd(),"temp",filename)

    playsound(filepath)

def feature_extraction(file_name):
    x, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #plot(x, "Audio", "Time", "Amplitude")
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs


def plot(vector, name, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.show()

def addUser(name, loc, plz):
    param = {"name": name, "location": loc, "plz": plz}

    print(requests.post("http://localhost:5000/users", json=param).json()['message'])

def getUsers():
    ret = requests.get("http://localhost:5000/users").json()
    l = []
    for i in range(len(ret)):
        l.append(ret[i][0])
    return l




def getUserByName(name):
    ret = requests.get("http://localhost:5000/users/" + name).json()[0]

    return f"{ret[2]}, PLZ {ret[3]}"


def synthesize_text_with_audio_profile(text, output, effects_profile_id, voicee):


    client = texttospeech.TextToSpeechClient(credentials=credentials)

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(language_code="de-DE" , name=voicee)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=[effects_profile_id],
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    with open(output, "wb") as out:

        out.write(response.audio_content)
        print('Audiodaten wurden in File gespeichert: "%s"' % output)


def makefolder(name):
    #create a temporary folder if not exists
    import os
    if not os.path.exists(name):
        os.makedirs(name)


def main():
    makefolder('temp')
    # A List of Items
    items = ['', getUserByName("Noah"), getUserByName("Timo")]
    l = len(items)
    weather = []

    # Initial call to print 0% progress
    for i, item in tqdm(enumerate(items), total=l, desc="Progress", unit=" Process", unit_scale=True, unit_divisor=1, ncols=100):

        time.sleep(0.1)
        if i != 0:
            if i == 1:
                weather.append(get_weather(item, "Noah"))
            elif i == 2:
                weather.append(get_weather(item, "Timo"))
    for i in tqdm(range(100), desc="Loading", ncols=100, unit_scale=False):
        time.sleep(0.05)
    for w in weather:
        synthesize_text_with_audio_profile(w, "temp/" + str(random.randint(0,50)) + ".mp3", "medium-bluetooth-speaker-class-device", "de-DE-Neural2-B")
def classification():
    features = []
    l = getUsers()
    for j, names in tqdm(enumerate(l), desc="Loading Profiles", ncols=100, unit_scale=True, unit_divisor=1, unit=" Profiles"):
        for i in tqdm(os.listdir(names.lower()), desc=f"Loading {names}", ncols=100, unit_scale=True, unit_divisor=1, unit=" FeatureExtractions"):
            features.append([feature_extraction(f'{names.lower()}/{i}'), names])
            time.sleep(0.01)
    extracted_features_df = pd.DataFrame(features, columns=['feature', 'names'])
    extracted_features_df.head()
    X = np.array(extracted_features_df['feature']).tolist()
    y = np.array(extracted_features_df['names']).tolist()
    y = to_categorical(labelencoder.fit_transform(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    num_labels = y.shape[1]
    #Model Creation
    ###first layer
    model.add(Dense(100, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###final layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    #Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #Training the model
    num_epochs = 100
    num_batch_size = 32
    checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)

    print()
    print(np.array(X).shape)
    print(np.array(y).shape)
    print(np.array(X_train).shape)
    print(np.array(y_train).shape)
    print(np.array(X_test).shape)
    print(np.array(y_test).shape)

    print(X)
    print(y)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    start = datetime.datetime.now()

    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
    duration = datetime.datetime.now() - start
    print("Training completed in time: ", duration)

    #Check the Test Accuracy
    test_accuracy=model.evaluate(X_test,y_test,verbose=0)
    print(test_accuracy[1])

    predict_x = model.predict(X_test)
    names_x = np.argmax(predict_x, axis=1)
    print(names_x)

def predict(filename):
    # Testing some Test Audio Files
    pass
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    mfccs = mfccs.reshape(1, -1)
    x_predict = model.predict(mfccs)
    predicted_label = np.argmax(x_predict, axis=1)
    print(predicted_label)
    predicted_name = labelencoder.inverse_transform(predicted_label)
    print(predicted_name)

# Call the function to start listening
if __name__ == '__main__':


    listen_and_start()
    #main()
    #classification()
    #feature_extraction("temp/wetter5000.wav")
    #addUser("Noah", "Mieders", 6142)
    #print(get_weather(getUserByName("Noah"), "Noah"))
    #predict("temp/wetter5000.wav")
    #predict("temp/wetter500.wav")


