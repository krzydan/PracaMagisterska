import getopt
import os
import pathlib
import pickle
import sys
from datetime import datetime
from pathlib import Path
from random import randrange

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tensorflow.python.keras import backend

train_path = "C:/Users/user/PycharmProjects/PracaMagisterska" #defined as global because other set is used only once.

def mean_mfccs(x):
    '''Returns the mean of MFCCs'''
    return [np.mean(feature) for feature in librosa.feature.mfcc(x,n_mfcc=40)]

def parse_audio(x):
    '''Parses audio.'''
    return x.flatten('F')[:x.shape[0]]


def get_audios_full_soundfile(train_file_names, dataset_path):
    ''' Returns samples which are full length audios in mfccs of each file. Files are read by soundfile.'''

    samples = []
    for file_name in train_file_names:
        x, sr = sf.read(dataset_path + str(file_name))
        x = parse_audio(x)
        samples.append(mean_mfccs(x))

    return samples

def get_audios_full_librosa(train_file_names, dataset_path):
    ''' Returns samples which are full length audios in mfccs of each file. Files are read by Librosa'''
    samples = []
    max_pad_len=1320
    for file_name in train_file_names:
        x, sr = librosa.load(dataset_path + str(file_name), res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)

        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        samples.append(mfccs)

    return samples

def get_audios_part(train_file_names,seconds,dataset_path):
    ''' Function returns x seconds of each audio file in mfccs.This part of audio file starts at random moment.'''
    samples = []

    for file_name in train_file_names:
        off = randrange(30-seconds)
        x, sr = librosa.load(dataset_path + str(file_name), duration=seconds, offset=off)
        x = parse_audio(x)
        samples.append(mean_mfccs(x))

    return samples


def get_audios_more(dataset,train_file_names,N):
    ''' Function returns N * 1000 2-second samples'''
    samples = []

    for file_name in train_file_names:
        i=0
        while i < N:
            off = randrange(28)
            x, sr = librosa.load(dataset+ str(file_name), duration=2, offset=off)
            x = parse_audio(x)
            samples.append(mean_mfccs(x))
            i=i+1

    return samples

def get_samples_wav(dataset,representation,rep_val=1):
    '''Funcions returns samples from *.wav dataset.'''
    df = np.loadtxt(Path(dataset+"/genres/input.mf"), dtype=str)
    files = df[:, 0]
    data = np.char.replace(files, '/Users/sness/mirex2008', '')
    classes = df[:, 1]
    final_classes=[]
    if(rep_val>=1):
        multi = rep_val
        if representation!= "size":
            multi = 1
        for category in classes:
            i=0
            while i < multi:
                final_classes.append(category)
                i=i+1
    if representation == "duration":
        if rep_val == 30:
            return get_audios_full_soundfile(data, dataset), final_classes
        else:
            return get_audios_part(data,rep_val,dataset), final_classes
    elif representation == "":
        return get_audios_full_soundfile(data, dataset), final_classes
    elif representation == "size":
        return get_audios_more(dataset,data,rep_val), final_classes
    elif representation in ("cnn_3conv","cnn_4conv", "cnn_4convnodropout","vgg16_fine"):
        return get_audios_full_librosa(data, dataset), final_classes



def get_samples_info(dataset):
    '''Function gets names of image files and returns it and class names for GTZAN set.'''
    df2 = np.loadtxt(Path(dataset+"/genres/input.mf"), dtype=str)
    files = df2[:, 0]
    data = np.char.replace(files, '/Users/sness/mirex2008/genres', '')
    data = np.char.replace(data,'.','')
    data = np.char.replace(data,'wav','.png')
    all_classes= ["blues", "classical", "country","disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for label in all_classes:
        data = np.char.replace(data,"/"+label+"/","")
    classes = df2[:, 1]
    return data, classes

def append_ext(fn):
    return fn+".jpg"

def get_samples_images():
    # df = pd.read_csv(train_file, names=["class", "artist", "album", "track", "track_number", "file_path"], usecols=["class","file_path"],)
    df2 = np.loadtxt(Path("PracaMagisterska/genres/input.mf"), dtype=str)
    files = df2[:, 0]
    lists = []
    data = np.char.replace(files, '/Users/sness/mirex2008', '')
    classes = df2[:, 1]
    final_classes=[]
    multi =1
    for category in classes:
        i=0
        while i < multi:
            final_classes.append(category)
            i=i+1

    return files, final_classes


def get_audios_mp3(dataset_path, train_file_names):
    '''Function returns samples from mp3 set as mfccs.'''
    train_path = dataset_path+"/audio/training/"
    samples = []
    for file_name in train_file_names:
        x, sr = librosa.load(Path(train_path + str(file_name)))
        x = parse_audio(x)
        samples.append(mean_mfccs(x))

    return samples


def get_samples_mp3(dataset_path, train_file):
    '''Function returns paths of audio files from mp3 set.'''
    df = pd.read_csv(train_file, names=["class", "artist", "album", "track", "track_number", "file_path"],
                     usecols=["class", "file_path"], )
    return get_audios_mp3(dataset_path, df['file_path']), df['class']

def machineLearning(dataset,classifier,n_neighbours=3, weights='distance', metric='manhattan', usePCA=False,
                    repeats=10, kernel='rbf', degree = 3, C=10,n_components=15, representation="",rep_val=1):
    '''Function gets parameters required by specifed machine learning algorithm and prints results of training model.'''
    print(datetime.now())
    if representation == "mp3":
        XT, YT = get_samples_mp3(dataset, dataset + '/metadata/training/tracklist.csv')
    else:
        XT,YT=get_samples_wav(dataset, representation = representation,rep_val=rep_val)


    if classifier=="knn":
        clf = KNeighborsClassifier(n_neighbors=n_neighbours, weights= weights, metric=metric)
    elif classifier=="svm":
        if representation == "mp3":
            clf = SVC(C=C, kernel=kernel, degree=degree, gamma='auto', probability=False, coef0=1)
        else:
            clf = SVC(C=C, kernel=kernel, degree=degree, gamma='auto', probability=True, coef0=1)

    kappa=[]
    cv = []
    f1 = []
    auc = []
    k_fold = KFold(n_splits=10, shuffle=True)
    i=0

    #PCA
    text = ' without'
    features =[]
    if(usePCA):
        scaler = StandardScaler()
        scaler.fit(XT)
        XT_scaled = scaler.transform(XT)

        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(XT_scaled)
        features = principalComponents
        text = ' with'
    else:
        features = XT

    while (i < repeats):
        i = i + 1
        cv_mean = []
        f1_mean = []
        auc_mean = []
        kappa_mean = []
        for train_index, test_index in k_fold.split(features):
            clf.fit(np.array(features)[train_index], np.array(YT)[train_index])
            ypred = clf.predict(np.array(features)[test_index])
            cv_mean.append(accuracy_score(np.array(YT)[test_index], ypred))
            f1_mean.append((f1_score(np.array(YT)[test_index], ypred, average='macro')))
            kappa_score = cohen_kappa_score(np.array(YT)[test_index], ypred)
            if (np.math.isnan(kappa_score)!=True):
                kappa_mean.append(kappa_score)

            if (classifier=='knn' or clf.probability==True):
                yproba = clf.predict_proba(np.array(features)[test_index])
                auc_mean.append(roc_auc_score(np.array(YT)[test_index], yproba, multi_class='ovr'))
        cv.append(np.mean(cv_mean))
        f1.append(np.mean(f1_mean))
        if (classifier=='knn' or clf.probability==True):
            auc.append(np.mean(auc_mean))
        kappa.append((np.mean(kappa_mean)))

    print(classifier+text+' PCA cv mean:{:.2f} +/-{:.2f}'.format(np.mean(cv), np.std(cv)*2))
    print(classifier+text+' PCA f1 mean:{:.2f} +/-{:.2f}'.format(np.mean(f1), np.std(f1)*2))
    if (classifier=='knn' or clf.probability==True):
         print(classifier+text+' PCA AUC mean:{:.2f} +/-{:.2f}'.format(np.mean(auc), np.std(auc) * 2))
    print(classifier+text+' PCA kappa mean:{:.2f} +/-{:.2f}'.format(np.mean(kappa), np.std(kappa)*2))


    print(datetime.now())

    if(n_components==20):
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Liczba składowych')
        plt.ylabel('Wariancja (%)')
        plt.show()


def cnn_4conv(num_labels, num_rows,num_columns, num_channels, dropout=True):
    '''Returns CNN which consists of 4 Conv + 4 MaxPool + GAP.'''
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    if dropout:
        model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    if dropout:
        model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    if dropout:
        model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    if dropout:
        model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))
    return model

def cnn_3conv(num_labels, num_rows,num_columns, num_channels):
    '''Returns CNN which consists of 3 Conv + 3 MaxPool + GAP.'''
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_labels, activation='softmax'))

    return model

def cnn_MFCCs(dataset_path,cnn_type,epochs,batch_size):
    '''Funtion gets convolutional network type and studied parameters. At the end it prints measures of accuracy
    of the best model. The model is trained on MFCCs from audio files.'''
    XT, YT = get_samples_wav(dataset_path,cnn_type)
    XT = np.array(XT)

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(YT))
    num_labels = 10
    num_channels = 1
    num_rows = 40
    num_columns = 1320
    iterations = 3
    num_epochs = epochs
    num_batch_size = batch_size
    i=0
    lr = 0.001
    opt = Adam(lr=lr)

    k_fold = KFold(n_splits=10, shuffle=True)

    kappa_all = []
    accuracy_all = []
    loss = []
    f1_all = []
    auc_all = []
    pretrained =[]
    start = datetime.now()

    while(i < iterations):
        i+=1
        VALIDATION_ACCURACY = []
        VALIDATION_LOSS = []
        f1_mean = []
        auc_mean = []
        kappa_mean = []
        pretrained_mean =[]

        fold_var=1
        pathlib.Path(dataset_path+f'/saved_models/MFCCs/{i}/').mkdir(parents=True, exist_ok=True)  # create folders if not existing
        for train_index, test_index in k_fold.split(yy):
            print('\nIteration ', i)
            print('\nFold ', fold_var)
            x_train = XT[train_index]
            x_test = XT[test_index]
            y_train = yy[train_index]
            y_test = yy[test_index]


            x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
            x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

            #choosing the cnn architecture
            if cnn_type == "cnn_4conv":
                model = cnn_4conv(num_labels, num_rows, num_columns, num_channels)
            elif cnn_type == "cnn_4convnodropout":
                model = cnn_4conv(num_labels, num_rows, num_columns, num_channels,False)
            elif cnn_type == "cnn_3conv":
                model = cnn_3conv(num_labels, num_rows, num_columns, num_channels)


            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

            model.summary()

            # Calculate pre-training accuracy
            score = model.evaluate(x_test, y_test, verbose=0)
            pretrained_mean.append(score[1])

            # Save best model for current iteration
            checkpointer = ModelCheckpoint(
                filepath=dataset_path+f'/saved_models/MFCCs/{i}/model_{cnn_type}_{num_batch_size}_{num_epochs}_{lr}_{fold_var}.h5',
                monitor='val_accuracy', verbose=1, save_best_only=True)


            callbacks_list = [checkpointer]


            history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
                                validation_data=(x_test, y_test),
                                callbacks=callbacks_list)



            plot_name = dataset_path+f"/plots/MFCCs/{i}/MFCCs.{cnn_type}_{num_batch_size}_{num_epochs}_{lr}_{fold_var}.png"
            save_history_plots(history, plot_name)


            # load best model to evaluate the performance of the model
            model.load_weights(
                dataset_path+f'/saved_models/MFCCs/{i}/model_{cnn_type}_{num_batch_size}_{num_epochs}_{lr}_{fold_var}.h5')

            # Evaluating the model on testing data
            score = model.evaluate(x_test, y_test, verbose=0)

            VALIDATION_ACCURACY.append(score[1])
            VALIDATION_LOSS.append(score[0])

            pred_proba = model.predict(x_test, verbose=0)
            pred_classes = model.predict_classes(x_test, verbose=0)
            y = np.argmax(y_test, axis=1)


            accuracy = accuracy_score(y, pred_classes)
            VALIDATION_ACCURACY.append(accuracy)

            #f1_score
            f1 = f1_score(y, pred_classes, average='macro')
            f1_mean.append(f1)

            # kappa
            kappa = cohen_kappa_score(y, pred_classes)
            kappa_mean.append(kappa)

            # ROC AUC
            auc = roc_auc_score(y, pred_proba, multi_class='ovr')
            auc_mean.append(auc)

            backend.clear_session()
            fold_var += 1

        auc_all.append(auc_mean)
        f1_all.append(f1_mean)
        kappa_all.append(kappa_mean)
        accuracy_all.append(np.mean(VALIDATION_ACCURACY))
        loss.append(np.mean(VALIDATION_LOSS))
        pretrained.append(np.mean(pretrained_mean))


    print('cnn cv accuracy:{:.2f}$\pm${:.2f}'.format(np.mean(accuracy_all), np.std(accuracy_all) * 2))
    print('cnn cv loss:{:.2f}$\pm${:.2f}'.format(np.mean(loss), np.std(loss) * 2))
    print('cnn f1 mean:{:.2f}$\pm${:.2f}'.format(np.mean(f1_all), np.std(f1_all) * 2))
    print('cnn without PCA AUC mean:{:.2f}$\pm${:.2f}'.format(np.mean(auc_all), np.std(auc_all) * 2))
    print('cnn  kappa mean:{:.2f}$\pm${:.2f}'.format(np.mean(kappa_all), np.std(kappa_all) * 2))
    duration = datetime.now() - start
    print("Experiment completed in time: ", duration)
    print('cnn  pretrained mean:{:.2f}$\pm${:.2f}'.format(np.mean(pretrained), np.std(pretrained) * 2))


def cnn_images(data_path,cnn,mode,epochs,minibatch):
    '''CNN is learnt on images from librosa or plt folder (chosen mode) and saves best models.'''

    start = datetime.now()
    if mode == "images_librosa":
        imgdir = data_path+'/img_librosa'
    else:
        imgdir = data_path+"/img_plt"
    save_dir = data_path+'/saved_models/Images/'
    type = cnn+"_"+mode

    XT, YT = get_samples_info(data_path)
    data = pd.DataFrame(data=[XT,YT]).transpose()
    data.columns =["filename","class"]

    kf = KFold(n_splits=10, shuffle=True)


    num_classes = 10
    num_channels=3
    lr = 0.001
    img_size=64

    batch_size=minibatch
    num_epochs=epochs

    opt = Adam(lr=lr)
    datagen = ImageDataGenerator(rescale=1. / 255.)
    iterations = 3
    i=0
    kappa_all = []
    accurracy_all = []
    loss =[]
    f1_all = []
    auc_all = []



    while (i < iterations):
        i=i+1
        VALIDATION_ACCURACY = []
        VALIDATION_LOSS = []
        f1_mean = []
        auc_mean = []
        kappa_mean = []
        fold_var = 1  # fold index
        pathlib.Path(save_dir + f'{i}/').mkdir(parents=True, exist_ok=True)  # create folders if not existing
        for train_index, val_index in kf.split(data):
            print('\nIteration ', i)
            print('\nFold ', fold_var)
            training_data = data.iloc[train_index]
            validation_data = data.iloc[val_index]



            train_data_generator = datagen.flow_from_dataframe(training_data, directory=imgdir,
                                                               x_col="filename", y_col="class",
                                                               class_mode="categorical", shuffle=True,
                                                               batch_size=batch_size,
                                                               target_size=(img_size, img_size))
            valid_data_generator = datagen.flow_from_dataframe(validation_data, directory=imgdir,
                                                               x_col="filename", y_col="class",
                                                               class_mode="categorical", shuffle=True,
                                                               batch_size=batch_size,
                                                               target_size=(img_size, img_size))
            test_data_generator = datagen.flow_from_dataframe(validation_data, directory=imgdir,
                                                              x_col="filename", y_col="class",
                                                              class_mode="categorical", shuffle=False,
                                                              batch_size=batch_size,
                                                              target_size=(img_size, img_size))

            # create model
            if cnn == "vgg16_fine":
                model = get_instancevgg16(num_classes, img_size,img_size, num_channels, True)
            elif cnn == "vgg16_transfer":
                model = get_instancevgg16(num_classes, img_size, img_size, num_channels, False)
            elif cnn == "cnn_4conv":
                model = cnn_4conv(num_classes,img_size,img_size,num_channels)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            checkpoint = ModelCheckpoint(data_path+f"/saved_models/Images/{i}/{type}_BatchNormalization{img_size}_{batch_size}_{num_epochs}_{lr}_{fold_var}.h5",
                                         monitor='val_accuracy', verbose=1,
                                         save_best_only=True, mode='max')


            callbacks_list = [checkpoint]

            history = model.fit(train_data_generator,
                                epochs=num_epochs,
                                callbacks=callbacks_list,
                                validation_data=valid_data_generator,
                                steps_per_epoch=train_data_generator.n // batch_size,
                                validation_steps=valid_data_generator.n // batch_size)

            # plotting
            history_name = data_path+f"/history/Images/{i}/{type}_BatchNormalization{img_size}_{batch_size}_{num_epochs}_{lr}_{fold_var}.hist"
            plot_name = data_path+f"/plots/Images/{i}/{type}_BatchNormalization{img_size}_{batch_size}_{num_epochs}_{lr}_{fold_var}.png"
            save_history(history, history_name)
            #h = load_history(history_name)
            save_history_plots(history, plot_name)

            # Load the best model to evaluate the performance of the model
            model.load_weights(
                data_path+f"/saved_models/Images/{i}/{type}_BatchNormalization{img_size}_{batch_size}_{num_epochs}_{lr}_{fold_var}.h5")

            results = model.evaluate(valid_data_generator)
            results = dict(zip(model.metrics_names, results))

            VALIDATION_ACCURACY.append(results['accuracy'])
            VALIDATION_LOSS.append(results['loss'])

            pred = model.predict_generator(test_data_generator, steps=test_data_generator.n // batch_size,verbose=2)
            predicted_class_indices = np.argmax(pred, axis=1)

            # Fetch labels from train gen for testing
            labels = (test_data_generator.class_indices)
            labels = dict((v, k) for k, v in labels.items())
            predictions = [labels[k] for k in predicted_class_indices]

            test_y = validation_data["class"]

            accuracy = accuracy_score(test_y, predictions)

            f1 = f1_score(test_y, predictions, average='macro')
            f1_mean.append(f1)

            # kappa
            kappa = cohen_kappa_score(test_y, predictions)
            kappa_mean.append(kappa)

            # ROC AUC
            auc = roc_auc_score(test_y, pred, multi_class='ovr')

            backend.clear_session()

            fold_var += 1

        auc_all.append(auc_mean)
        f1_all.append(f1_mean)
        kappa_all.append(kappa_mean)
        accurracy_all.append(VALIDATION_ACCURACY)
        loss.append(VALIDATION_LOSS)


    print(VALIDATION_ACCURACY)
    print('cnn cv accuracy:{:.2f} $\pm$ {:.2f}'.format(np.mean(accurracy_all), np.std(accurracy_all) * 2))
    print('cnn cv loss:{:.2f} $\pm$ {:.2f}'.format(np.mean(loss), np.std(loss) * 2))
    print('cnn f1 mean:{:.2f} $\pm$ {:.2f}'.format(np.mean(f1_all), np.std(f1_all) * 2))
    print('cnn without PCA AUC mean:{:.2f} $\pm$ {:.2f}'.format(np.mean(auc_all), np.std(auc_all) * 2))
    print('cnn  kappa mean:{:.2f} $\pm$ {:.2f}'.format(np.mean(kappa_all), np.std(kappa_all) * 2))
    duration = datetime.now() - start
    print("Experiment completed in time: ", duration)

def save_history(history, file_name):
    '''Saving history of fitting the neural network.'''
    with open(file_name, 'wb') as f:
        pickle.dump(history, f)


def load_history(file_name):
    '''Loading history of fitting the neural network.'''
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_history_plots(history, file_name=None):
    '''Function saves accuracy and loss as plot showing changes between epochs.'''
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_accuracy" + file_extension)
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_loss" + file_extension)
    plt.close()


def generateImages(path,library):
    '''Function generates spectrograms from audio files from path using chosen library and saves them into path.'''
    plt.figure(figsize=(10, 10))
    songname = path+"/genres/blues/blues.00000.wav"
    cmap = plt.get_cmap('inferno')
    y, sr = librosa.load(songname)
    if library == "librosa":
        folder = "img_librosa"
        spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
        spectogram = librosa.power_to_db(spectogram, ref=np.max)
        librosa.display.specshow(spectogram, x_axis='time', y_axis='mel', sr=sr, hop_length=512)
    else:
        folder = "img_plt"
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
    pathlib.Path(path + f'/{folder}').mkdir(parents=True, exist_ok=True)  # create folder if not existing
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for g in genres:
        for filename in os.listdir(path+f'/genres/{g}'):
            songname = path+f'/genres/{g}/{filename}'
            y, sr = librosa.load(songname)
            if library == "librosa":
                spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
                spectogram = librosa.power_to_db(spectogram, ref=np.max)
                librosa.display.specshow(spectogram, x_axis='time', y_axis='mel', sr=sr, hop_length=512)
            else:
               plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('off');
            plt.savefig(path+f'/{folder}/{filename[:-3].replace(".", "")}.png')
            print(filename+" transformed into image")
            plt.clf()

        print(f'{g} done!')


def get_instancevgg16(num_classes,img_width, img_height, channels, finetuning):
    '''Returns intance of VGG16 convolutional neural network.'''
    feature_extractor = applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                     input_shape=(img_height, img_width , channels))

    for layer in feature_extractor.layers:
        layer.trainable = False

    x = feature_extractor.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)

    result_model = Model(inputs=feature_extractor.input, outputs=output)


    if (finetuning == True):
        for layer in result_model.layers:
            if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']:
                layer.trainable = True
        layers = [(layer.name, layer.trainable) for layer in result_model.layers]

    return result_model

def visualizeData():
    '''Visualizing data distribution as a bar graph.'''
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    counts = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    df = pd.DataFrame(counts, index=genres)
    plt.figure(figsize=(10,20))
    ax = df.plot.bar(zorder=3)
    ax.get_legend().remove()
    ax.grid(zorder=0)
    plt.tight_layout()
    plt.savefig(train_path+"/audioset.png")
    plt.close()

def startExperiment():
    '''Gets parameters from terminal and runs function proper for chosen classifier.'''
    classif = ''
    weights =''
    metric = ''
    kernel = ''
    k = -1 # parameter k for KNN
    kernel = ''
    degree = 0
    c = -1
    pca = False
    n_components = -1
    minibatch = -1
    epochs = -1
    representation = ''
    representation_value = 1
    repeats = 10 #how many times repeat experiment
    try:
        opts, args = getopt.getopt(sys.argv[2:], "hc:r:v:", ["clf=", "weights=", "metric=", "n_neighbours=", "kernel=",
                                                             "degree=", "cfun=","pca=","n_components=","minibatch=",
                                                             "epochs=","repeats=", "representation="])
    except getopt.GetoptError:
        print('__main__.py <dataset> --clf <classifier> [classifier parameters] [-r <music representation parameter type> -v <value of that parameter>]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('__main__.py <dataset> -c <classifier> [classifier parameters] [-r <music representation parameter type> -v <value of that parameter>]')
            sys.exit(2)
        elif opt == "--clf":
            classif = arg
        elif opt == '--weights':
            weights = arg
        elif opt == '--metric':
            metric = arg
        elif opt == '--n_neighbours':
            k = int(arg)
        elif opt == '--kernel':
            kernel = arg
        elif opt == '--degree':
            degree = int(arg)
        elif opt == '--cfun':
            c = int(arg)
        elif opt == '--pca':
            if arg.lower() == 'true':
                pca = True
        elif opt == '--n_components':
            n_components = int(arg)
        elif opt == '--minibatch':
            minibatch = int(arg)
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt in ('-r', "--representation"):
            representation = arg
        elif opt == '-v':
            representation_value = int(arg)
        elif opt == '--repeats':
            repeats = int(arg)
    dataset = sys.argv[1]
    if representation == "generateplt":
        generateImages(dataset,"plt")
        sys.exit(0)
    elif representation == "generatelibrosa":
        generateImages(dataset, "librosa")
        sys.exit(0)
    if representation_value < 1 and representation in ("size", "duration"):
        print('Value of -v option has to be greater or equal 1.')
        sys.exit(2)
    if classif == 'knn':
        machineLearning(dataset=dataset,classifier=classif, n_neighbours=k, weights=weights, metric=metric, usePCA=pca,
                        repeats=repeats, representation = representation,rep_val=representation_value)
    elif classif == 'svm':
        if(n_components<0):
           machineLearning(dataset=dataset, classifier=classif, kernel=kernel, degree=degree, C=c, repeats=repeats,
                            usePCA=pca, representation =representation, rep_val=representation_value)
        else:
           machineLearning(dataset=dataset,classifier=classif,kernel= kernel, degree=degree,C=c, repeats=repeats,
                           usePCA=pca,n_components=n_components, representation= representation,rep_val=representation_value)
    elif classif in("cnn_3conv", "cnn_4conv","cnn_4convnodropout","vgg16_fine","vgg16_transfer"):
        if representation in ("","MFCCs"):
            cnn_MFCCs(dataset,classif,epochs,minibatch)
        elif representation in ("images_librosa","images_plt"):
            cnn_images(dataset,classif,representation,epochs, minibatch)
    else:
        print("Classifier type unknown.")

if __name__ == '__main__':
    #visualizeData()
    startExperiment()
