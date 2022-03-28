# %% 
from data_utils.data_utils import prepare_data, get_dataset, get_test_dataset, sound_dataset_generator, sound_dataset_generator_by_filename
from model.hvae import VAE_BN, VAE_NoBN, loss_function, loss_function_test

import json
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# %%
class Sound_anomaly_detector(object):
    """ A tool to easily conduct for Sound Anomaly Detection
    
    Parameters
    -----------
    cfg_path : str
        json type config path
    
    Examples
    """
    def __init__(self, cfg_path):

        # initialize parameter groups
        self.main_config = {}
        self.directory_config = {}
        self.training_config = {}
        self.spectrogram_config = {}
        

        # load json config, parameter groups will have values
        print("--- Load Config ---")
        self._load_config(cfg_path)
        print("--- Load Config End ---")

    def _load_config(self,config):
        """Load Json type configuration and put the parameters on Memory 
        Parameters
        -----------
        cfg_path : str
            json type configuration path
        
        """
            
        with open(config, 'r') as cfg:
            self.config = json.load(cfg)
        self.main_config = self.config['main_parameters']
        self.directory_config = self.config['directory_parameters']
        self.training_config = self.config['training_parameters']
        self.spectrogram_config = self.config['spectrogram_parameters']

    def train(self):
        """Sound Anomaly Detector's training function.
        Parameters should be defined with config path (json type)
        when class instantiated.
            
        Returns
        -----------
        npy
            train dataset (numpy array)
        pt
            model weight file, 
        """

        ### 1. Make test dataset (npy)
        train_dataset_dir = os.path.join(self.directory_config['experiment_result_path'], 'train_dataset.npy')
        
        if os.path.exists(train_dataset_dir):
            train_dataset = np.load(train_dataset_dir)
        else:
            train_file_list = prepare_data(self.directory_config['train_data_dir'], remove_filename_list=self.directory_config['remove_filename_list'])
            train_dataset = get_dataset(train_file_list, channel=self.directory_config['tdms_channel_name'], n_mels=self.spectrogram_config['n_mels'], 
                                        frames=self.spectrogram_config['frames'], n_fft=self.spectrogram_config['n_fft'], hop_length=self.spectrogram_config['hop_length'],
                                        power=self.spectrogram_config['power'], sr=self.spectrogram_config['sr'])
            np.save(train_dataset_dir, train_dataset) 

        ### 2. Make Pytorch Dataloader (Train) 
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset = sound_dataset_generator(train_dataset)
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.training_config['batch_size'])
        
        ### 3. Build Model
        if self.training_config['sep_arch'] == True:
            model = VAE_BN(layers=self.training_config['layers'], embedding_dim=self.training_config['embedding_dim'])
        else:
            model = VAE_NoBN(layers=self.training_config['layers'], embedding_dim=self.training_config['embedding_dim'])
        print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
        model.cuda()

        ### 4. Set up the optimizer
        opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

        ### 5. Model Train
        min_loss = 10000000
        model.train()
        EPOCHS = self.training_config['epochs']
        for epoch in range(EPOCHS):
            scheduler.step(epoch)
            loop = tqdm(train_loader, total = len(train_loader), leave = True)
            lossfs = []
            for feature in loop:
                feature = feature.float()
                feature = feature.to(device)
                recon_feature, recon_sigma, mu, logvar = model(feature)
                loss = loss_function(self.training_config['beta'], recon_feature, recon_sigma, feature, mu, logvar)
                lossf = loss.data.item()
                lossfs.append(lossf)
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            print("Epoch: {} / {}, train average loss: {}".format(epoch, EPOCHS, np.mean(lossfs)))
            if min_loss > np.mean(lossfs):
                print("min loss updated {} to {}. New model saved!!".format(min_loss, np.mean(lossfs)))
                min_loss = np.mean(lossfs)
                torch.save(model.state_dict(), self.directory_config['model_save_dir'])
        
    def test(self):
        """Sound Anomaly Detector's inference function.
        Parameters should be defined with config path (json type)
        when class instantiated.
            
        Returns
        -----------
        npy
            test dataset (numpy array)
        pt
            model weight file, 
        """
        
        ### 1. Make test dataset (npy)
        test_dataset_dir = os.path.join(self.directory_config['experiment_result_path'], 'test')
        
        if os.path.exists(test_dataset_dir):
            print("npy files is already extracted.")
            test_npy_list = prepare_data(test_dataset_dir, remove_filename_list=self.directory_config['remove_filename_list'])
        else:
            os.makedirs(test_dataset_dir, exist_ok=True)
            test_file_list = prepare_data(self.directory_config['test_data_dir'], remove_filename_list=self.directory_config['remove_filename_list'])
            get_test_dataset(test_file_list, test_dataset_dir, channel=self.directory_config['tdms_channel_name'], n_mels=self.spectrogram_config['n_mels'], 
                                        frames=self.spectrogram_config['frames'], n_fft=self.spectrogram_config['n_fft'], hop_length=self.spectrogram_config['hop_length'],
                                        power=self.spectrogram_config['power'], sr=self.spectrogram_config['sr'])
            test_npy_list = prepare_data(test_dataset_dir, remove_filename_list=self.directory_config['remove_filename_list'])

        ### 2. Make Pytorch Dataloader (Test) 
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        test_dataset = sound_dataset_generator_by_filename(test_npy_list)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

        ### 3. Build Model
        if self.training_config['sep_arch'] == True:
            model = VAE_BN(layers=self.training_config['layers'], embedding_dim=self.training_config['embedding_dim'])
        else:
            model = VAE_NoBN(layers=self.training_config['layers'], embedding_dim=self.training_config['embedding_dim'])
        model.load_state_dict(torch.load(self.directory_config['model_load_dir']))
        model.cuda()
        model.eval()

        ### 4. Evaluate
        anomaly_score_list = []
        loop = tqdm(test_loader, total = len(test_loader), leave = True)
        with torch.no_grad():
            for feature in loop:
                feature = feature.float()
                feature = feature[0]
                feature = feature.to(device)
                recon_feature, recon_sigma, mu, logvar = model(feature)
                loss = loss_function_test(self.training_config['beta'], recon_feature, recon_sigma, feature, mu, logvar)
                """
                loss_average: Average of loss in one audio sameple
                loss_median: Median of loss in one audio sameple
                """
                loss_average = torch.mean(loss).cpu().detach()
                loss_median = torch.median(loss).cpu().detach()
                loss_list = [loss_average, loss_median]
                loss = np.array([loss_list])
                anomaly_score_list = anomaly_score_list + [loss]

        ### 5. Save a Result
        anomaly_score_list = np.vstack(anomaly_score_list)
        result1 = pd.DataFrame(anomaly_score_list)
        file_name_list = [os.path.basename(file) for file in test_npy_list]
        result = pd.DataFrame({'File': file_name_list})
        result = pd.concat([result, result1], axis = 1)
        result.columns = ['File', 'Mean', 'Median']
        result_path = os.path.join(self.directory_config['experiment_result_path'], 'test_prediction')
        os.makedirs(result_path, exist_ok=True)
        result.to_csv(result_path + '/' + self.directory_config['test_prediction_filename'] + '.csv', index = False)
            
if __name__ == "__main__":
    sound_anomaly_detector = Sound_anomaly_detector("./hvae_parameters.params")
    if sound_anomaly_detector.main_config['phase'] == 'train':
        sound_anomaly_detector.train()
    else:
        sound_anomaly_detector.test()
# %%
