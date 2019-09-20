import wave
import scipy
import pylab
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

from .wavutils import load_audio_file, save_tensor

# returns the mel spectrogram of the input audio
def get_mel_spectrogram(audio,sr):
    eps=2.220446049250313e-16

    audio = audio.reshape([1,-1])
    ms = int(0.04*sr) #40ms at 44100 Hz
    window = scipy.signal.hamming(
                                ms,
                                sym=False
                                )
    mel_basis = librosa.filters.mel(sr=sr,
                                    n_fft=2048,
                                    n_mels=96,
                                    fmin=100,
                                    fmax=1000,
                                    htk=False,
                                    norm=None
                                    )
    feature_matrix = np.empty((0,96))
    hop_length = int(sr/50)
    stft = librosa.stft(audio[0,:]+eps,
                            n_fft=2048,
                            win_length=ms,
                            hop_length=hop_length,
                            center=True,
                            window=window
                            )
    # print("stft shape : {}".format(stft.shape))
    spectrogram = np.abs(stft)**2
    mel_spectrogram = np.dot(mel_basis,spectrogram)
    mel_spectrogram = mel_spectrogram.T
    mel_spectrogram = np.log10(mel_spectrogram + eps)
    feature_matrix = np.append(feature_matrix,mel_spectrogram,axis=0)
    return feature_matrix

def genFeatures(filePath,outPath,name,suffixe,sr):
    mel_spec = get_mel_spectrogram(load_audio_file(filePath),sr)
    # print('generated feature of shape {}'.format(mel_spec.shape))
    save_tensor(mel_spec,
                outPath,
                name,
                suffixe)


# test_filepath = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000\\01000000_20.wav'
#
# genFeatures('D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000\\01000000_20.wav',
#             'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000\\test',
#             '01000000_20.wav',
#             '_test')


#### OLD ####

# filepath = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000\\01000000_20.wav'
# csvpath = 'D:\\datas\\SON\\OUAKAM_AVRIL_MAI_2018\\2018-05\\prepared_dataset\\01000000\\01000000_20.csv'
# labeldf = pd.read_csv(csvpath)
# activity = labeldf['fish_activity']
# print ('activité : ',activity)
#
# #################
# ### WAVE METHOD
# #################
#
# wav = wave.open(filepath,'r')
# (nchannels,sampwidth,framerate,nframes,comptype,compname)=wav.getparams()
# print('number of frames : ', nframes)
# print('framerate : ', framerate)
# frames = wav.readframes(-1)
# sound_info = pylab.fromstring(frames, 'int16')
# framerate = wav.getframerate()
# wav.close()
#
# # # fig, (ax1,ax2)= plt.subplots(nrows=2)
# # spec = ax1.specgram(sound_info, Fs=framerate, NFFT=2048, mode = 'psd')
# # # ax1.colorbar()
# # ax1.ylim((100,2000))
# # print('shape of pylab spectrogram : ', np.array(spec[0]).shape)
# # # pylab.ylim((100,2000))
# spec,freqs,bins,im = pylab.specgram(sound_info, Fs=framerate, NFFT=2048,mode='magnitude')
# plt.colorbar()
# print('shape of pylab spectrogram : ', np.array(spec).shape)
# # pylab.ylim((100,2000))
# print('shape of the im : ',im.get_extent())
# plt.autoscale(False)
# im.set_extent((0.0014512471655328818, 0.9592743764172336, 100, 2000))
# print('shape of the im : ',im.get_extent())
# print('shape of pylab spectrogram : ', np.array(spec).shape)
# data = ''
# data = im.format_cursor_data(im)
# plt.show()
# print(data)
# #################
# ###LIBROSA METHOD
# #################
# data, _ = librosa.core.load(filepath, sr=44100,duration=1)
# data = librosa.feature.mfcc(data, sr=44100,
#                                    n_mfcc=128)
#
# y,sr = librosa.load(filepath,sr=44100)
# dur = librosa.get_duration(y=y,sr=sr)
# print('duration : ', dur)
# print('sampling rate :',sr)
# print('shape of librosa mfcc : ', data.shape)
#
# fig=plt.figure(figsize=(8, 4))
# librosa.display.specshow(data, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.grid(True,which='both')
# plt.tight_layout()
#
# plt.show()
