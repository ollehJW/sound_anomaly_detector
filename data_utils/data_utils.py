import os
import numpy as np
import pandas as pd
import librosa
import sys
from tqdm import tqdm
from nptdms import TdmsFile
from torch.utils.data import Dataset

def prepare_data(train_path, remove_filename_list=[]):
    if 'csv' == train_path.split('.')[-1]:
        files_df = pd.read_csv(train_path)
        files = files_df['path'].to_list()
    else:
        files = []
        filelist = os.listdir(train_path)
        for i, file in enumerate(filelist):
            if file.endswith('.wav') | file.endswith('.tdms') | file.endswith('.npy'):
                if file not in remove_filename_list:
                    files.append(os.path.join(train_path, file))

    return files


def get_signal(fpath, sr, channel_name = 'CPsignal1'):
    if fpath.endswith('tdms'):
        signal = load_tdms(file_path=fpath, channel_name=channel_name)
    else:
        signal = load_wav(file_path=fpath, sr=sr)

    return signal

def load_wav(file_path,sr):
    signal, _ = librosa.load(file_path, sr)
    return signal

def load_tdms(file_path, channel_name="CPsignal1"):
    """Load Single tdms file and returns target signal

    Parameters
    ----------
    file_path : str
        a single tdms file path
    channel_name : str
        Name of interest channel. Defaults to "CPsignal1".

    Returns
    --------
    np.array
        raw singal shape of (N,)
    """
    tdms_file = TdmsFile(file=file_path)
    tdms_group = tdms_file.groups()
    signal = tdms_group[0][channel_name][:]
    return signal

def get_spec(signal,
            n_mels=128,
            frames=5,
            n_fft=1024,
            hop_length=512,
            power=2.0,
            sr = 25000):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def get_dataset(file_list,
                         msg="calc...",
                         channel='CPsignal1',
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         sr = 25000):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        signal = get_signal(file_list[idx], sr, channel)
        vector_array = get_spec(signal,
                                n_mels=n_mels,
                                frames=frames,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                power=power,
                                sr = sr)
        if idx == 0:
            dataset = vector_array
        else:
            dataset = np.concatenate((dataset, vector_array), axis=0)

    return dataset

def get_test_dataset(file_list, npy_save_dir,
                         channel='CPsignal1',
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         sr = 25000):
    """
    Extract npy from Test audio files.
    Each audio files makes npy files with same file name.

    """
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list))):
        file_name = os.path.basename(file_list[idx])
        save_name = file_name.split(".")[0] + ".npy"
        save_path = os.path.join(npy_save_dir, save_name)
        signal = get_signal(file_list[idx], sr, channel)
        
        vector_array = get_spec(signal,
                                n_mels=n_mels,
                                frames=frames,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                power=power,
                                sr = sr)

        np.save(save_path, vector_array)
        



class sound_dataset_generator(Dataset):

    """
    Make Train Dataset_generator
    Input: Total train audio feature array (Very large)
    
    """
    def __init__(self, sound_dataset):
        self.sound_dataset = sound_dataset
        
    def __len__(self):
        return len(self.sound_dataset)

    def __getitem__(self, idx):
        feature = self.sound_dataset[idx]
        return feature

class sound_dataset_generator_by_filename(Dataset):

    """
    Make Test Dataset_generator
    Input: Test audio feature array (Small)
    
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        feature = np.load(file_name)
        return feature

