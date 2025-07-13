# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:57:30 2024
train version for windows/linux

use of the acc function

@author: Jia_G
"""

import argparse
import torch
import torch.nn as nn
# import netCDF4 as nc
import os
import scipy.io as scio
import numpy as np
from Models import network2d
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, interp2d
import Lossfunction
import matplotlib.pyplot as plt
import modelpara
import pandas as pd
import shutil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class EnvMed_multi():
    def __init__(self, 
                 adresse_data_ga = '', 
                 adresse_data_MDT = '',
                 adresse_data_meridianDOV = '',
                 adresse_data_primeDOV = '',
                 adresse_data_H_model = '',
                 adresse_VGG = '',
                 adresse_data_seaheight = '',
                 adresse_data_sedimentary = '',
                 list_adresse = [],
                 # adresse_data_slope,
                 size_image = [6, 6],
                 input_nomber = []):
        
        self.data_ga = np.load(list_adresse[0])
        self.data_MDT = np.load(list_adresse[1])
        self.data_mDOV = np.load(list_adresse[2])
        self.data_pDOV = np.load(list_adresse[3])
        self.data_H = np.load(list_adresse[4])
        self.data_VGG = np.load(list_adresse[5])
        self.data_seaheight = np.load(list_adresse[6])
        self.data_sedimentary = np.load(list_adresse[7])
        self.data_grad_H = np.load(list_adresse[8])
        # self.data_GEBCO = np.load(list_adresse[9])
        
        self.list_lon = self.data_ga['lon']
        self.list_lat = self.data_ga['lat']
        self.image_ga = self.data_ga['z']
        self.image_MDT = self.data_MDT['z']
        self.image_mDOV = self.data_mDOV['z']
        self.image_pDOV = self.data_pDOV['z']
        self.image_H = self.data_H['z']
        self.image_VGG = self.data_VGG['z']
        self.image_sh = self.data_seaheight['z']
        self.image_sed = self.data_sedimentary['z']
        
        self.image_grad_H = self.data_grad_H['z']
        self.image_grad_H_theta = self.data_grad_H['grad']
        # self.image_GEBCO = self.data_GEBCO['z']
        self.size_image = size_image             
        
        self.input_nomber = input_nomber
        if len(self.input_nomber) == 13:
            self.choose = False
        else:
            self.choose = True
            self.images_nomber = len(self.input_nomber)
 
    def data_generater(self, lon, lat, intype = 'lines', normalise=''):
        size_image = self.size_image
        
        left_index = np.where(self.list_lon <= lon)[0][-1]
        south_index = np.where(self.list_lat <= lat)[0][-1]
        
        ga_image = np.zeros(size_image)
        MDT_image = np.zeros(size_image)
        mDov_image = np.zeros(size_image)
        pDov_image = np.zeros(size_image)
        H_image = np.zeros(size_image)
        VGG_image = np.zeros(size_image)
        seaheight_image = np.zeros(size_image)
        sedimentary_image = np.zeros(size_image)
        grad_H_image = np.zeros(size_image)
        grad_H_theta_image = np.zeros(size_image)
        # GBECO_image = np.zeros(size_image)
        
        x_image = np.zeros(size_image, dtype = int) # lat
        y_image = np.zeros(size_image, dtype = int) # lon
        lat_image = np.zeros(size_image)
        lon_image = np.zeros(size_image) 
        
        for x in range(size_image[0]):
            for y in range(size_image[1]):
                x_image[x, y]  = int(south_index + size_image[0] / 2 - x )
                y_image[x, y]  =  int(left_index - int(size_image[1] / 2) + y + 1 )
                
                lat_image[x, y] = self.list_lat[x_image[x, y]]
                lon_image[x, y] = self.list_lon[y_image[x, y]]
                ga_image[x, y] = self.image_ga[x_image[x, y] , y_image[x, y]]
                MDT_image[x, y] = self.image_MDT[x_image[x, y] , y_image[x, y]]
                H_image[x, y] = self.image_H[x_image[x, y] , y_image[x, y]]
                VGG_image[x, y] = self.image_VGG[x_image[x, y] , y_image[x, y]]
                seaheight_image[x, y] = self.image_sh[x_image[x, y] , y_image[x, y]]
                sedimentary_image[x, y] = self.image_sed[x_image[x, y] , y_image[x, y]]
                mDov_image[x, y] = self.image_mDOV[x_image[x, y] , y_image[x, y]]
                pDov_image[x, y] = self.image_pDOV[x_image[x, y] , y_image[x, y]]
                
                grad_H_image[x, y] = self.image_grad_H[x_image[x, y] , y_image[x, y]]
                grad_H_theta_image[x, y] = self.image_grad_H_theta[x_image[x, y] , y_image[x, y]]
                # GBECO_image[x, y] = self.image_GEBCO[x_image[x, y] , y_image[x, y]]
                
        land_sea = H_image < 0 # Is this point located on land or at sea
        land_sea = land_sea.astype('float')
        
        d_lat_image = lat_image - lat
        d_lon_image = lon_image - lon
        
        if intype == 'line':
            lines = []
            for x in range(size_image[0]):
                for y in range(size_image[1]):
                    lines.append(d_lon_image[x, y])
                    lines.append(d_lat_image[x, y])
                    lines.append(ga_image[x, y])

            lines = np.array(lines)
            lines = lines[np.newaxis, :]
            return lines
        
        elif intype == 'image':
            out = np.zeros([13, size_image[0], size_image[1]])
            out[0, :, :] = d_lon_image 
            out[1, :, :] = d_lat_image
            out[2, :, :] = H_image / 1000 # km
            out[3, :, :] = land_sea
            out[4, :, :] = grad_H_image
            out[5, :, :] = grad_H_theta_image
            out[6, :, :] = ga_image # cm/s²
            out[7, :, :] = VGG_image 
            out[8, :, :] = MDT_image / 10000 # 10km
            out[9, :, :] = pDov_image
            out[10, :, :] = mDov_image
            out[11, :, :] = seaheight_image
            out[12, :, :] = np.log(sedimentary_image) # / 1000000
            if self.choose == False:
                return out                
            else:
                images_all = out
                out = np.zeros([self.images_nomber, size_image[0], size_image[1]])
                for i in range(self.images_nomber):
                    out[i, :, :] = images_all[self.input_nomber[i], :, :]
            if normalise == 'normal':
                out = (out - out.mean()) / out.std()
            elif normalise == 'minmax':
                out = (out - out.min()) / ((out - out.max()))   
            return out

class CustomDataset_1(Dataset):
    def __init__(self,
                 adresse_truth,
                 adresse_data_image,
                 size_image, 
                 mode, 
                 type_input,
                 input_nomber=[],
                 factor = 1,
                 normalise = '',
                 **kwargs):
        super(CustomDataset_1, self).__init__(**kwargs)
        self.mode = mode
        self.EM = EnvMed_multi(list_adresse = adresse_data_image, size_image=size_image, input_nomber=input_nomber)
        self.adresse_truth = adresse_truth
        self.type_input = type_input
        assert self.adresse_truth, 'there is no fichier given'
        data_truth = np.load(adresse_truth)
        # list_nomber = data_truth['list_' + mode]    
        for key in data_truth.keys():
            if 'list_' in key:
                list_nomber = data_truth[key]
        wd = data_truth['wd']
        coordinates = data_truth['coordinates']

        self.list_wd = wd[list_nomber] / factor
        self.list_coordinates = coordinates[list_nomber]
        self.factor = factor
        self.normalise = normalise
        
    def __len__(self):
        return len(self.list_wd)
    def __getitem__(self, index):
        lon_lat = np.array(self.list_coordinates[index])
        wd = np.array(self.list_wd[index])
        
        image = self.EM.data_generater(lon_lat[1], lon_lat[0], intype='image', normalise = self.normalise)
        if 'residual' in self.type_input:
            if self.type_input == 'residual4':
                image_H = image[2, :, :]
                image_diff = image_H[int(self.EM.size_image[0] / 2 - 1) : int(self.EM.size_image[0] / 2 + 1), 
                                     int(self.EM.size_image[1] / 2 - 1) : int(self.EM.size_image[1] / 2 + 1)]
                image_diff = - image_diff * 1000 - wd # surronding - h
                wd = image_diff.reshape(4)
            elif self.type_input == 'residual1':
                image_H = image[2, :, :]
                image_diff = image_H[int(self.EM.size_image[0] / 2 - 1), 
                                     int(self.EM.size_image[1] / 2 - 1)]
                wd = - image_diff * 1000 - wd
                
            return image, wd, image_diff
        else:
            return image, wd

def commands_load(args):
    if '.mat' not in args.command:
        args.command = args.command + '.mat'
        
    try:
        setting = scio.loadmat(args.command)
    except:
        os.kill()
    
    for key in setting.keys():
        print(key)
        if '__' not in key:
            try:
                if isinstance(setting[key][0], str):
                    setattr(args, key, setting[key][0])
            except:
                continue
    if len(setting['input_nomber']) == 0:
        list_inputnomber = []
        for i in range(13):
            list_inputnomber.append(i)
        setattr(args, 'input_nomber', list_inputnomber)
    else:
        setattr(args, 'input_nomber', setting['input_nomber'][0])
            
    args.size_image = int(args.size_image)
    args.epochs = int(args.epochs)
    
    args.factor = setting['factor'][0][0]
    args.show = setting['show'][0][0]
    if 'debug' in args.task:
        args.epochs = 3
    if '4' in args.type_input:
        if '1' in args.network_nomber:
            print('wrong network and input type')
            os.kill()
    return args

#%% main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', help = 'the command used to run the progam')
    args = parser.parse_args()
    args = commands_load(args)
    current_pos = os.getcwd()
    
    current_pos = os.getcwd()
    if current_pos[0] == '/':
        current_dic = current_pos + '/'
        core_part = '/home/vipuser'
        adresse_ga = core_part + '/Afterphd/SEA/SDUST2022GRA.npz' # z(lat, lon) (9721, 21601) 0, 360; -80, 82. 1'
        adresse_VGG = core_part + '/Afterphd/SEA/curv_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80. 1'
        adresse_H_model = core_part + '/Afterphd/SEA/topo_25.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_MDT = core_part + '/Afterphd/SEA/dtu22mdt_global.npz' # z(y, x) (1361, 2881) [  0. 360.]; [-80.  90.]
        adresse_meridianDOV = core_part + '/Afterphd/SEA/north_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_primeDOV = core_part + '/Afterphd/SEA/east_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_sedimentary = core_part + '/Afterphd/SEA/GlobSed-v3.npz' # z(lat, lon) (2161, 4321) -180, 180; -80, 80
        adresse_MSS = core_part + '/Afterphd/SEA/DTU21MSS_1min.npz' # z(lat, lon) (10800, 21600) -180, 180; -90, 90
        adresse_grad_H = core_part + '/Afterphd/SEA/grad_H.1.npz'        
        # adresse_GEBCO = core_part + '/Afterphd/SEA/data1/GEBCO.npz'
        adresse_data_Phil_truth_t = core_part + '/Afterphd/train_file3.npz'
        adresse_data_Phil_truth_v = core_part + '/Afterphd/vali_file3.npz'
        adresse_data_Phil_truth_te = core_part + '/Afterphd/test_file3.npz'        

        sign_system = True
    elif current_pos[0] == 'D':
        adresse_ga = 'D:\\Afterphd\\SEA\\data1\\SDUST2022GRA.npz' # z(lat, lon) (9721, 21601) 0, 360; -80, 82. 1'
        adresse_VGG = 'D:\\Afterphd\\SEA\\data1\\curv_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80. 1'
        adresse_H_model = 'D:\\Afterphd\\SEA\\data1\\topo_25.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_MDT = 'D:\\Afterphd\\SEA\\data1\\dtu22mdt_global.npz' # z(y, x) (1361, 2881) [  0. 360.]; [-80.  90.]
        adresse_meridianDOV = 'D:\\Afterphd\\SEA\\data1\\north_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_primeDOV = 'D:\\Afterphd\\SEA\\data1\\east_32.1.npz' # z(lat, lon) (9600, 21600) -180, 180; -80, 80
        adresse_sedimentary = 'D:\\Afterphd\\SEA\\data1\\GlobSed-v3.npz' # z(lat, lon) (2161, 4321) -180, 180; -80, 80
        adresse_MSS = 'D:\\Afterphd\\SEA\\data1\\DTU21MSS_1min.npz' # z(lat, lon) (10800, 21600) -180, 180; -90, 90
        adresse_grad_H = 'D:\\Afterphd\\SEA\\data1\\grad_H.1.npz'
        # adresse_GEBCO = core_part + 'D:\\Afterphd\\SEA\\data1\\GEBCO.npz'
        adresse_data_Phil_truth_t = 'D:\\Afterphd\\ISBI2019chanllenge\\train_file2.npz'
        adresse_data_Phil_truth_v = 'D:\\Afterphd\\ISBI2019chanllenge\\vali_file2.npz'
        adresse_data_Phil_truth_te = 'D:\\Afterphd\\ISBI2019chanllenge\\test_file2.npz'
        current_dic = current_pos + '\\'
        sign_system = False

    channels_input = len(args.input_nomber)
    if args.network_nomber == '1': 
        model_train = network2d.network_2([channels_input, args.size_image, args.size_image]).to(device)
    elif args.network_nomber == '11': # more parameters
        model_train = network2d.network_2([channels_input, args.size_image, args.size_image], parameters=[16, 32, 64]).to(device)            
    elif args.network_nomber == '2': # residual 1 point
        model_train = network2d.network_2([channels_input, args.size_image, args.size_image], parameters=[32, 64, 128]).to(device)
    elif args.network_nomber == '3': # residual 4 points
        model_train = network2d.network_2([channels_input, args.size_image, args.size_image], flatten_parametesr = [512, 256, 128, 4]).to(device)
    elif args.network_nomber == '4': 
        model_train = network2d.network_2([channels_input, args.size_image, args.size_image], parameters = [64, 128, 128]).to(device)
    elif args.network_nomber == 'GBZ':
        model_train = network2d.network_GBZ([channels_input, args.size_image, args.size_image]).to(device)
         
    model_train.double()
    if args.lossfn == 'MSELoss':
        loss_fn = nn.MSELoss() # MSE Loss
    elif args.lossfn == 'L1loss':
        loss_fn = nn.L1Loss() # L1 Loss
    elif args.lossfn == 'MSEpercent':
        loss_fn = Lossfunction.MSELoss_percent
    elif args.lossfn == 'MSEcomb':
        loss_function = Lossfunction.MSElosspercent(k = 100)
        loss_fn = loss_function.lossfunction
        
    if args.lossfn == 'MSEcomb':
        acc_fns = [Lossfunction.simple_loss_noabs, nn.MSELoss(), Lossfunction.MSELoss_percent]
    else:
        acc_fns = [Lossfunction.simple_loss_noabs]
    
    list_adresse = [adresse_ga, # cm/s²
                    adresse_MDT, 
                    adresse_meridianDOV,
                    adresse_primeDOV,
                    adresse_H_model,
                    adresse_VGG,
                    adresse_MSS,
                    adresse_sedimentary,
                    adresse_grad_H
                    # adresse_GEBCO
                    ]
#%% train
    if 'train' in args.task:
        learning_rate = 1e-2
        num_epochs = args.epochs
        batch_size_train = 2048
        optimizer_train = torch.optim.Adam(model_train.parameters(), lr = learning_rate)
        # schedular = torch.optim.lr_scheduler.StepLR(optimizer_train, step_size = 10, gamma = 0.1)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_train, mode='min', factor=0.1, patience=10, verbose=True)
        
        train_Dataset = CustomDataset_1(adresse_data_Phil_truth_t, 
                                     list_adresse,
                                     [args.size_image, args.size_image],
                                     mode = 'train',
                                     type_input = args.type_input,
                                     input_nomber = args.input_nomber,
                                     factor=args.factor
                                     )
        val_Dataset = CustomDataset_1(adresse_data_Phil_truth_v, 
                                     list_adresse,
                                     [args.size_image, args.size_image],
                                     mode = 'vali',
                                     type_input = args.type_input,
                                     input_nomber = args.input_nomber,
                                     factor=args.factor
                                     )
        train_loader = DataLoader(train_Dataset, batch_size = batch_size_train, shuffle=True)
        val_loader = DataLoader(val_Dataset, batch_size = 2048, shuffle=False)
        recording_train_loss = np.zeros([1, num_epochs])
        recording_vali_loss = np.zeros([1, num_epochs])     
        nom_acc = len(acc_fns)
        
        recording_train_acc = np.zeros([nom_acc, num_epochs])
        recording_acc = np.zeros([nom_acc, num_epochs])
    
        lr_list = []
        a = time.localtime() 
        # if args.num_model
        num_model = args.num_model
        # str(a[2]) + str(a[1]) + str(a[0]) + str(a[3]) + str(a[4])
        filePath_tracking = 'model' + num_model + 'track'
        modelpara.fichier_creat(filePath_tracking)
        # lossfn = Lossfunction.simple_loss_noabs
        
        if sign_system:
            filePath_save = current_pos + filePath_tracking + '/'
        else:
            filePath_save = current_pos + filePath_tracking + '\\'
            
        loss_value = 0
        for epoch in range(num_epochs):
            model_train.train()
            loss_vale = 0
            acc_value = np.zeros([nom_acc, 1])
            # train
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, target) in pbar:
                data, target = data.to(device), target.to(device) 
                optimizer_train.zero_grad() # 清除梯度
                output = model_train(data) # 正向传播
                loss = loss_fn(output, target) # 计算损失
                loss.backward() # 反向传播
                optimizer_train.step() # 优化参数
                for i in range(nom_acc):
                    acc_fn = acc_fns[i]
                    acc = acc_fn(output, target)
                    acc_value[i, 0] += acc.mean().cpu().detach().numpy()
                avg_train_acc_value = acc_value / (batch_idx + 1)
                loss_value += loss.item()
                avg_loss = loss_value / (batch_idx + 1)
                test_line = f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc:{avg_train_acc_value[0][0]:.4f}'
                if nom_acc > 1:
                    test_line = test_line + f',{avg_train_acc_value[1][0]:.4f}, {avg_train_acc_value[2][0]:.4f}'
                pbar.set_description(test_line)
            # validate
            model_train.eval()
            val_loss_value = 0
            val_loss_simple = 0 
            pbar1 = tqdm(enumerate(val_loader), total=len(val_loader))
            with torch.no_grad():
                for batch_idx, (data, target) in pbar1:
                    data, target = data.to(device), target.to(device) 
                    output = model_train(data)
                    loss = loss_fn(output, target)
                    for i in range(nom_acc):
                        acc_fn = acc_fns[i]
                        acc = acc_fn(output, target)
                        acc_value[i, 0] += acc.mean().cpu().numpy()
                    avg_acc_value = acc_value / (batch_idx + 1)
                    val_loss_value += loss.item()
                    avg_val_loss = val_loss_value / (batch_idx + 1)
                    test_line = f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Acc:{avg_acc_value[0][0]:.4f}'
                    if nom_acc > 1:
                        test_line = test_line + f',{avg_acc_value[1][0]:.4f}, {avg_acc_value[2][0]:.4f}'
                    pbar1.set_description(test_line)
                    # pbar1.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}')
            lr_list.append(optimizer_train.state_dict()['param_groups'][0]['lr'])
            schedular.step(avg_acc_value[0][0])
            recording_train_loss[0, epoch] = avg_loss
            recording_train_acc[:, epoch] = avg_train_acc_value.flatten()
            recording_vali_loss[0, epoch] = avg_val_loss
            recording_acc[:, epoch] = avg_acc_value.flatten()
            adresse_save_inepoch = current_dic + filePath_tracking + '/train' + args.nomber + 'model' + str(epoch) + '.pth'
            torch.save(model_train.state_dict(), adresse_save_inepoch)

        adresse_save_final = current_dic + filePath_tracking + '/train' + args.nomber + 'model' + num_model + '.pth'
        torch.save(model_train.state_dict(), adresse_save_final)
        
        try:
            line_val = recording_acc[0, :]
            position_min = np.argmin(recording_vali_loss[0])
            modelpara.fichier_copy(current_dic + filePath_tracking + '/train' + args.nomber + 'model' + str(position_min) + '.pth', 
                                   current_dic + filePath_tracking + '/train' + args.nomber + 'model' + str(position_min) + 'best.pth')
        except:
            print('min try lost')
        
        np.savez(current_dic + filePath_tracking + 'recording_train' + num_model, 
                 recording_train_loss = recording_train_loss,
                 recording_vali_loss = recording_vali_loss,
                 lr_list = lr_list,
                 args = args
                 )
        
        #%% save validation result
        val_loader = DataLoader(val_Dataset, batch_size = 1, shuffle=False)
        recording_vali_loss = []    
        recording_t_loss = []
        recording_loss = []
        
        model_train.eval()
        val_loss_value = 0
        test_loss_noabs = 0
        test_loss_value = 0

        epoch = 0
        num_epochs = 1
        # model_train.eval()
        loss_real_noabs = Lossfunction.simple_loss_noabs
        loss_real = Lossfunction.simple_loss

        list_res = []
        list_tar = []
        pbar1 = tqdm(enumerate(val_loader), total=len(val_loader))
        
        with torch.no_grad():
            for batch_idx, (data, target) in pbar1:
                list_tar.append(target.numpy())
                data, target = data.to(device), target.to(device) 
                output = model_train(data)
                loss1 = loss_fn(output, target)
                loss2 = loss_real(target, output)
                loss3 = loss_real_noabs(output, target)
                val_loss_value += loss1.item()
                test_loss_value += loss2.item()
                test_loss_noabs += loss3.item()
                recording_vali_loss.append(loss1.item())
                recording_t_loss.append(loss2.item())
                recording_loss.append(loss3.item())
                
                avg_val_loss = val_loss_value / (batch_idx + 1)
                avg_test_loss = test_loss_value / (batch_idx + 1)
                avg_test_loss_noabs = test_loss_noabs / (batch_idx + 1)
                list_res.append(output.cpu().numpy())
                pbar1.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}, {avg_test_loss_noabs:.4f}')
        
        np.savez(current_dic + 'recording_vali' + num_model, 
                 avg_val_loss = avg_val_loss, 
                 avg_test_loss = avg_test_loss,
                 recording_vali_loss = recording_vali_loss,
                 recording_t_loss = recording_t_loss,
                 recording_loss = recording_loss,
                 avg_test_loss_noabs = avg_test_loss_noabs,
                 list_res = list_res,
                 list_tar = list_tar
                 )
        
        list_res = np.squeeze(list_res)
        list_tar = np.squeeze(list_tar)
        
        if 'residual' in args.type_input:
            data_truth = np.load(adresse_data_Phil_truth_v)
            for key in data_truth.keys():
                if 'list_' in key:
                    list_nomber = data_truth[key]
            wd = data_truth['wd']
            coordinates = data_truth['coordinates']
            list_wd = wd[list_nomber]
            list_coordinates = coordinates[list_nomber]
            data_H = np.load(adresse_H_model)
            image_H = data_H['z']
            list_lon = data_H['lon']
            list_lat = data_H['lat']
            depth_ref = np.zeros_like(list_wd)
            for i in range(len(list_coordinates)):
                lon_lat = np.array(list_coordinates[i])
                
                left_index = np.where(list_lon <= lon_lat[1])[0][-1]
                south_index = np.where(list_lat <= lon_lat[0])[0][-1] + 1
                if 'residual1' in args.type_input: # differences + reference height
                    ref_h = image_H[int(south_index), int(left_index)]
                    list_res[i] = list_res[i] + ref_h                    
                    list_tar[i] = list_tar[i] + ref_h                    
                    
                elif 'residual4' in args.type_input: 
                    ref_h = image_H[int(south_index)-1:int(south_index)+1, int(left_index):int(left_index)+2]                   
                    test_ref_h = np.zeros_like(ref_h)
                    test_ref_h[0, :] = ref_h[1, :]
                    test_ref_h[1, :] = ref_h[0, :]
                    ref_h = test_ref_h.reshape(4)
                    list_res[i, :] = list_res[i, :] + ref_h                    
                    list_tar[i, :] = list_tar[i, :] + ref_h                    
            
        R2 = Lossfunction.R2_cal_np(list_res, list_tar) 
        MAE = Lossfunction.MAE_np(list_res, list_tar)
        MAPE = Lossfunction.MAPE_np(list_res, list_tar)
        region_150 = np.sum(np.abs(recording_loss) < 150) / len(list_res)
        RMS = np.sqrt(np.mean(np.power(recording_loss, 2)))
        
        deeps_region = np.floor(list_tar / 1000)
        regions_deeps = np.zeros([11, 2])
        for i in range(11):
            regions_deeps[i, 0] = i
            regions_deeps[i, 1] = np.sum(deeps_region == i)
        regions_deeps_2 = np.zeros([5, 1])
        regions_deeps_2[0, 0] = np.sum(deeps_region == 0)
        regions_deeps_2[1, 0] = np.sum(deeps_region == 1)
        regions_deeps_2[2, 0] = np.sum(deeps_region == 2)
        regions_deeps_2[3, 0] = np.sum(deeps_region == 3)
        regions_deeps_2[4, 0] = np.sum(deeps_region > 3)
        
        regions_deeps_n = np.zeros_like(list_tar)
        deep_sign = np.zeros([len(list_tar), 5])
        
        for i in range(len(regions_deeps_n)):
            regions_deeps_n[i] = (deeps_region[i] if deeps_region[i] < 4 else 4)
            deep_sign[i, regions_deeps_n[i].astype(int)] = i + 1 
        
        R2_d = np.zeros([5, 1])
        MAE_d = np.zeros_like(R2_d)
        MAPE_d = np.zeros_like(R2_d)
        mean_d = np.zeros_like(R2_d)
        std_d = np.zeros_like(R2_d)
        min_d = np.zeros_like(R2_d)
        max_d = np.zeros_like(R2_d)
        RMS_d = np.zeros_like(R2_d)
        length_deeps = []
        for i in range(5):
            position_choosed = deep_sign[:, i]
            non_zero_indices = np.nonzero(position_choosed)
            new_arr = position_choosed[non_zero_indices]
            length_deeps.append(len(new_arr))
            new_arr = new_arr - 1
            new_arr = new_arr.astype(int)
            out = list_res[new_arr]
            tar = list_tar[new_arr]
            error = recording_loss[new_arr]
            R2_d[i, 0] = Lossfunction.R2_cal_np(out, tar) 
            MAE_d[i, 0] = Lossfunction.MAE_np(out, tar) 
            MAPE_d[i, 0] = Lossfunction.MAPE_np(out, tar) 
            RMS_d[i, 0] = np.sqrt(np.mean(np.power(error, 2)))
            mean_d[i, 0] = np.mean(error)
            std_d[i, 0] = np.std(error)
            min_d[i, 0] = np.min(error)
            max_d[i, 0] = np.max(error)
        data = {
            'deep' : ['all', '<1000', '1000-2000', '2000-3000', '3000-4000', '>4000']
            }
        
        list_info_dm = []
        list_info_dm.append(str(np.mean(recording_loss)))
        list_info_dstd = []
        list_info_dstd.append(str(np.std(recording_loss)))
        list_info_dmin = []
        list_info_dmin.append(str(np.min(recording_loss)))
        list_info_dmax = []
        list_info_dmax.append(str(np.max(recording_loss)))
        list_info_R2 = []
        list_info_R2.append(str(R2))
        list_info_MAE = []
        list_info_MAE.append(str(MAE))
        list_info_MAPE = []
        list_info_MAPE.append(str(MAPE))
        list_info_RMS = []
        list_info_RMS.append(str(RMS))

        if 'residual' not in args.type_input:
            data = {
                'Colume1' : ['deep', 'all', '<1000', '1000-2000', '2000-3000', '3000-4000', '>4000']
                }
            for i in range(5):
                list_info_dm.append(str(mean_d[i, 0]))
                list_info_dstd.append(str(std_d[i, 0]))
                list_info_dmin.append(str(min_d[i, 0]))
                list_info_dmax.append(str(max_d[i, 0]))
                list_info_R2.append(str(R2_d[i, 0]))
                list_info_MAE.append(str(MAE_d[i, 0]))
                list_info_MAPE.append(str(MAPE_d[i, 0]))
                list_info_RMS.append(str(RMS_d[i, 0]))
        else:
            data = {
                'Colume1' : ['deep', 'all']
                }
        
        data['distance m'] = list_info_dm
        data['distance std'] = list_info_dstd
        data['distance min'] = list_info_dmin
        data['distance max'] = list_info_dmax
        data['R2'] = list_info_R2
        data['RMS'] = list_info_RMS
        data['MAE'] = list_info_MAE
        data['MAPE'] = list_info_MAPE
        df = pd.DataFrame(data)
        file_path = current_dic + filePath_tracking + '/output.csv'
        df.to_csv(file_path, index=False)
    #%% validation
    if 'vali' in args.task:
        num_model = 'model' + args.num_model + 'track/train0model' + args.best + '.pth'
        model_train.double()    
        model_state_dict = torch.load(current_dic + num_model)
        model_train.load_state_dict(model_state_dict)
        # torch.load(model_train.state_dict(), '/home/vipuser/Afterphd/ISBI2019chanllenge/' + num_model)
        val_Dataset = CustomDataset_1(adresse_data_Phil_truth_v, 
                                     list_adresse,
                                     [args.size_image, args.size_image],
                                     mode = 'vali',
                                     type_input = args.type_input,
                                     input_nomber = args.input_nomber,
                                     factor=args.factor
                                     )
        
        model_train.eval()
        val_loader = DataLoader(val_Dataset, batch_size = 1, shuffle=False)

        loss_fn = nn.MSELoss() # MSE Loss
        loss_real = Lossfunction.simple_loss
        loss_real_noabs = Lossfunction.simple_loss_noabs
        
        recording_vali_loss = []    
        recording_t_loss = []
        recording_loss = []
        
        val_loss_value = 0
        test_loss_value = 0
        test_loss_noabs = 0
        
        list_res = []
        list_tar = []
        pbar1 = tqdm(enumerate(val_loader), total=len(val_loader))

        with torch.no_grad():
            for batch_idx, (data, target) in pbar1:
                list_tar.append(target.numpy())
                data, target = data.to(device), target.to(device) 
                output = model_train(data)
                loss1 = loss_fn(output, target)
                loss2 = loss_real(target, output)
                loss3 = loss_real_noabs(output, target)
                val_loss_value += loss1.item()
                test_loss_value += loss2.item()
                test_loss_noabs += loss3.item()
                recording_vali_loss.append(loss1.item())
                recording_t_loss.append(loss2.item())
                recording_loss.append(loss3.item())
                
                avg_val_loss = val_loss_value / (batch_idx + 1)
                avg_test_loss = test_loss_value / (batch_idx + 1)
                avg_test_loss_noabs = test_loss_noabs / (batch_idx + 1)
                list_res.append(output.cpu().numpy())
                pbar1.set_description(f'validation, Loss: {avg_val_loss:.4f}, {avg_test_loss_noabs:.4f}')
        np.savez('recording_vali' + args.num_model, 
                 avg_val_loss = avg_val_loss, 
                 avg_test_loss = avg_test_loss,
                 recording_vali_loss = recording_vali_loss,
                 recording_t_loss = recording_t_loss,
                 recording_loss = recording_loss,
                 avg_test_loss_noabs = avg_test_loss_noabs,
                 list_res = list_res,
                 list_tar = list_tar
                 )
        
        list_res = np.squeeze(list_res)
        list_tar = np.squeeze(list_tar)
        
        if 'residual' in args.type_input:
            data_truth = np.load(adresse_data_Phil_truth_v)
            for key in data_truth.keys():
                if 'list_' in key:
                    list_nomber = data_truth[key]
            wd = data_truth['wd']
            coordinates = data_truth['coordinates']
            list_wd = wd[list_nomber]
            list_coordinates = coordinates[list_nomber]
            data_H = np.load(adresse_H_model)
            image_H = data_H['z']
            list_lon = data_H['lon']
            list_lat = data_H['lat']
            depth_ref = np.zeros_like(list_wd)
            for i in range(len(list_coordinates)):
                lon_lat = np.array(list_coordinates[i])
                
                left_index = np.where(list_lon <= lon_lat[1])[0][-1]
                south_index = np.where(list_lat <= lon_lat[0])[0][-1] + 1
                if 'residual1' in args.type_input: # differences + reference height
                    ref_h = image_H[int(south_index), int(left_index)]
                    list_res[i] = list_res[i] + ref_h                    
                    list_tar[i] = list_tar[i] + ref_h                    
                    
                elif 'residual4' in args.type_input: 
                    ref_h = image_H[int(south_index)-1:int(south_index)+1, int(left_index):int(left_index)+2]                   
                    test_ref_h = np.zeros_like(ref_h)
                    test_ref_h[0, :] = ref_h[1, :]
                    test_ref_h[1, :] = ref_h[0, :]
                    ref_h = test_ref_h.reshape(4)
                    list_res[i, :] = list_res[i, :] + ref_h                    
                    list_tar[i, :] = list_tar[i, :] + ref_h                    
            
        R2 = Lossfunction.R2_cal_np(list_res, list_tar) 
        MAE = Lossfunction.MAE_np(list_res, list_tar)
        MAPE = Lossfunction.MAPE_np(list_res, list_tar)
        region_150 = np.sum(np.abs(recording_loss) < 150) / len(list_res)
        RMS = np.sqrt(np.mean(np.power(recording_loss, 2)))
        
        deeps_region = np.floor(list_tar / 1000)
        regions_deeps = np.zeros([11, 2])
        for i in range(11):
            regions_deeps[i, 0] = i
            regions_deeps[i, 1] = np.sum(deeps_region == i)
        regions_deeps_2 = np.zeros([5, 1])
        regions_deeps_2[0, 0] = np.sum(deeps_region == 0)
        regions_deeps_2[1, 0] = np.sum(deeps_region == 1)
        regions_deeps_2[2, 0] = np.sum(deeps_region == 2)
        regions_deeps_2[3, 0] = np.sum(deeps_region == 3)
        regions_deeps_2[4, 0] = np.sum(deeps_region > 3)
        
        regions_deeps_n = np.zeros_like(list_tar)
        deep_sign = np.zeros([len(list_tar), 5])
        
        for i in range(len(regions_deeps_n)):
            regions_deeps_n[i] = (deeps_region[i] if deeps_region[i] < 3 else 3)
            deep_sign[i, regions_deeps_n[i].astype(int)] = i + 1 
        
        R2_d = np.zeros([5, 1])
        MAE_d = np.zeros_like(R2_d)
        MAPE_d = np.zeros_like(R2_d)
        mean_d = np.zeros_like(R2_d)
        std_d = np.zeros_like(R2_d)
        min_d = np.zeros_like(R2_d)
        max_d = np.zeros_like(R2_d)
        RMS_d = np.zeros_like(R2_d)
        length_deeps = []
        for i in range(5):
            position_choosed = deep_sign[:, i]
            non_zero_indices = np.nonzero(position_choosed)
            new_arr = position_choosed[non_zero_indices]
            length_deeps.append(len(new_arr))
            new_arr = new_arr - 1
            new_arr = new_arr.astype(int)
            out = list_res[new_arr]
            tar = list_tar[new_arr]
            error = recording_loss[new_arr]
            R2_d[i, 0] = Lossfunction.R2_cal_np(out, tar) 
            MAE_d[i, 0] = Lossfunction.MAE_np(out, tar) 
            MAPE_d[i, 0] = Lossfunction.MAPE_np(out, tar) 
            RMS_d[i, 0] = np.sqrt(np.mean(np.power(error, 2)))
            mean_d[i, 0] = np.mean(error)
            std_d[i, 0] = np.std(error)
            min_d[i, 0] = np.min(error)
            max_d[i, 0] = np.max(error)
        data = {
            'deep' : ['all', '<1000', '1000-2000', '2000-3000', '3000-4000','>4000']
            }
        
        list_info_dm = []
        list_info_dm.append(str(np.mean(recording_loss)))
        list_info_dstd = []
        list_info_dstd.append(str(np.std(recording_loss)))
        list_info_dmin = []
        list_info_dmin.append(str(np.min(recording_loss)))
        list_info_dmax = []
        list_info_dmax.append(str(np.max(recording_loss)))
        list_info_R2 = []
        list_info_R2.append(str(R2))
        list_info_MAE = []
        list_info_MAE.append(str(MAE))
        list_info_MAPE = []
        list_info_MAPE.append(str(MAPE))
        list_info_RMS = []
        list_info_RMS.append(str(RMS))

        if 'residual' not in args.type_input:
            data = {
                'Colume1' : ['deep', 'all', '<1000', '1000-2000', '2000-3000', '3000-4000', '>4000']
                }
            for i in range(5):
                list_info_dm.append(str(mean_d[i, 0]))
                list_info_dstd.append(str(std_d[i, 0]))
                # list_info_d1.append(str(mean_d[i, 0]) + '+-' + str(std_d[i, 0]))
                list_info_dmin.append(str(min_d[i, 0]))
                list_info_dmax.append(str(max_d[i, 0]))
                # list_info_d2.append(str(min_d[i, 0]) + ', ' + str(max_d[i, 0]))
                list_info_R2.append(str(R2_d[i, 0]))
                list_info_MAE.append(str(MAE_d[i, 0]))
                list_info_MAPE.append(str(MAPE_d[i, 0]))
                list_info_RMS.append(str(RMS_d[i, 0]))
        else:
            data = {
                'Colume1' : ['deep', 'all']
                }
        
        data['distance m'] = list_info_dm
        data['distance std'] = list_info_dstd
        data['distance min'] = list_info_dmin
        data['distance max'] = list_info_dmax
        data['R2'] = list_info_R2
        data['RMS'] = list_info_RMS
        data['MAE'] = list_info_MAE
        data['MAPE'] = list_info_MAPE
        df = pd.DataFrame(data)
        file_path = current_dic + filePath_tracking + '/output.csv'
        df.to_csv(file_path, index=False)
        
    if 'test' in args.task: 
        if sign_system:
            num_model = 'model' + args.num_model + 'track/train0model' + args.best + '.pth'
        else:
            num_model = 'model' + args.num_model + 'track\\train0model' + args.best + '.pth'
        model_train.double()
        model_state_dict = torch.load(current_dic + num_model)
        model_train.load_state_dict(model_state_dict)
        val_Dataset = CustomDataset_1('test_file_model.npz', 
                                     list_adresse,
                                     [args.size_image, args.size_image],
                                     mode = 'vali',
                                     type_input = args.type_input,
                                     input_nomber = args.input_nomber,
                                     factor=args.factor
                                     )
        val_loader = DataLoader(val_Dataset, batch_size = 2048, shuffle=False)
        pbar1 = tqdm(enumerate(val_loader), total=len(val_loader))
        model_train.eval()
        epoch = 0
        num_epochs = 1
        model_train.eval()
        list_res = []
        with torch.no_grad():
            for batch_idx, (data, target) in pbar1:
                data, target = data.to(device), target.to(device) 
                output = model_train(data)
                output1 = output.cpu().numpy()
                output1 = output1.transpose()
                for no_i in range(len(output)):
                    list_res.append(output1[0, no_i])
                
                pbar1.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        
        np.savez('recording_test' + args.num_model + 'check' + args.nomber, 
                 list_res = list_res
                 )
        try:
            prediction = list_res.reshape([35 * 60 + 1, 30 * 60 + 1])
            prediction = prediction.astype(int)
            np.savez('recording_test' + args.num_model + 'check' + args.nomber, 
                     prediction = prediction
                     )
        except:
            print('wrong')
        
