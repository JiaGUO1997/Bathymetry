# -*- coding: utf-8 -*-
"""
Created on Sat May 31 21:27:01 2025

@author: Utilisateur
"""


import scipy
import scipy.io as scio
# import netCDF4 as nc
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator, interp2d
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy

fichier_data = 'D:\\Afterphd\\SEA\\PhilippineSea_120E_150E_0N_35N\\'

assert fichier_data, 'there is no fichier given'
for i, j, mat_list in os.walk(fichier_data):
    print('read the adress list')

#%% remove not unique points
try:
    test_data = np.load('unique_data.npz')
    unique_wd = test_data['unique_wd']
    unique_coordinates = test_data['unique_coordinates']
except:

    len_datas = len(mat_list)
    nomber_all = 0
    nomber_data = np.zeros([len_datas, 1])
    size_all = 5183703
    longitude = np.zeros([size_all, 1])
    latitude = np.zeros([size_all, 1])
    water_depth = np.zeros([size_all, 1])
    position_ll = np.zeros([size_all, 2])
    no_size = 0
    for no_data in range(len_datas):
        print(f'data {no_data + 1}/{len_datas}')
        file_path = fichier_data + mat_list[no_data]
        with open(file_path, 'r') as file:
            # Read the contents of the file
            file_contents = file.read()
        file.close()
        lines = file_contents.split('\n')
        lines.remove('')
        len_line = len(lines)
        no_line_data = 0
        for no_line in range(len_line):
            # no_line_data += 1
            line = lines[no_line]
            line_seperated = line.split('\t')
            longitude[no_size] = float(line_seperated[0])
            latitude[no_size] = float(line_seperated[1])
            water_depth[no_size] = float(line_seperated[2])
            position_ll[no_size, 0] = latitude[no_size]
            position_ll[no_size, 1] = longitude[no_size]
            
            nomber_data[no_data] += 1
            no_size += 1
            
    # 重复
    unique_coordinates, unique_indices, counts = np.unique(position_ll, axis=0, return_index=True, return_counts=True)
    unique_wd = water_depth[unique_indices]
    
    duplicate_coordinates = unique_coordinates[counts > 1] # 重复坐标
    duplicate_indices = unique_indices[counts > 1] # 重复位置
    
    # print("重复的坐标点及其对应的第三列数值位置：")
    # for coord, index in zip(duplicate_coordinates, duplicate_indices):
    #     print("坐标点:", coord)
    #     print("对应数值位置:", index)
    #     print("对应数值:", water_depth[index])
    sum_duplicate = 0
    not_duplicate_coordinates = position_ll[duplicate_indices,:]
    
    for no_duplicate in range(len(duplicate_indices)):    
        coordinate = duplicate_coordinates[no_duplicate, :]
        # index = duplicate_coordinates[no_duplicate, :]
        indices = np.where((position_ll[:, 0] == coordinate[0]) & (position_ll[:, 1] == coordinate[1]))
        duplicate_wd = water_depth[indices]
        # sum_duplicate += len(indices)
        wd_estimated = np.mean(duplicate_wd)
        unique_indices_estimated_wd = np.where((unique_coordinates[:, 0] == coordinate[0]) & (unique_coordinates[:, 1] == coordinate[1]))
        unique_wd[unique_indices_estimated_wd] = wd_estimated
            
        
    np.savez('unique_data', unique_coordinates = unique_coordinates,
            # unique_indices_estimated_wd,
            unique_wd = unique_wd)

print(len(unique_wd))
print('f')

#%%
try:
    data_correction = np.load('correction_stage3.npz')
    print('correction points load finished')
    off_wd = data_correction['off_wd']
    off_coordinates = data_correction['off_coordinates']
except:
    adresse_H_model = 'D:\\Afterphd\\data\\topo_25.1.nc' # z(lat, lon) 
    
    assert adresse_H_model, 'the given model of H does not exist'
    data = nc.Dataset(adresse_H_model)
    
    list_lat = data.variables['lat'][:]
    list_lon = data.variables['lon'][:]
    H_model = data.variables['z'][:]
    
    lat_min, lat_max = -5, 40
    lon_min, lon_max = 115, 155
    # 找出落在这个范围内的索引
    lat_indices = np.where((list_lat >= lat_min) & (list_lat <= lat_max))[0]
    lon_indices = np.where((list_lon >= lon_min) & (list_lon <= lon_max))[0]
    # 提取对应纬度、经度的子数组
    sub_lat = list_lat[lat_indices]
    sub_lon = list_lon[lon_indices]
    
    # 提取对应区域的高程数据（注意lat在第一个维度）
    sub_elevation = H_model[np.min(lat_indices):np.max(lat_indices)+1,
                            np.min(lon_indices):np.max(lon_indices)+1]
    x = np.arange(0, len(sub_lat)) * 1/60 + sub_lat[0]
    y = np.arange(0, len(sub_lon)) * 1/60 + sub_lon[0]
    
    interpH = RegularGridInterpolator([x, y], sub_elevation, method='cubic', bounds_error = False, fill_value = None)
    
    list_water_d_model = interpH(unique_coordinates)   
    list_water_d_model = list_water_d_model[:, np.newaxis]
    differences_w_d = unique_wd + list_water_d_model
    
    # raise Exception('111')
    # list_water_d_model = interpH(unique_coordinates)   
    # list_water_d_model = list_water_d_model[:, np.newaxis]
    # differences_w_d = unique_wd + list_water_d_model
    
    mean_d = np.mean(differences_w_d)
    std_d = np.std(differences_w_d)
    np.max(differences_w_d)
    np.min(differences_w_d)
    
    distance_region = np.array([mean_d - std_d * 3, mean_d + std_d * 3])
    
    list_little = differences_w_d < distance_region[0] 
    list_big = differences_w_d > distance_region[1]
    list_notwanted = list_little + list_big   
    
    list_wanted = []
    for no_nw in range(len(list_notwanted)):
        nw = list_notwanted[no_nw]
        if not nw:
            list_wanted.append(no_nw)
            
    off_coordinates = unique_coordinates[list_wanted, :]
    off_wd = unique_wd[list_wanted, :]
    
    #%%
    list_water_d_model_1 = interpH(off_coordinates)   
    list_water_d_model_1 = list_water_d_model_1[:, np.newaxis]
    differences_w_d_1 = off_wd + list_water_d_model_1
    
    mean_d_1 = np.mean(differences_w_d_1)
    std_d_1 = np.std(differences_w_d_1)
    
    distance_region = np.array([mean_d_1 - std_d_1 * 3, mean_d_1 + std_d_1 * 3])
    
    list_little = differences_w_d_1 < distance_region[0] 
    list_big = differences_w_d_1 > distance_region[1]
    list_notwanted_topo = list_little + list_big  
    list_wanted_topo = []
    for no_nw in range(len(list_notwanted_topo)):
        nw = list_notwanted_topo[no_nw]
        if not nw:
            list_wanted_topo.append(no_nw) 
    
    off_coordinates_2 = off_coordinates[list_wanted_topo, :]
    off_wd_2 = off_wd[list_wanted_topo, :]
    
    #%%
    list_water_d_model_2 = list_water_d_model_1[list_wanted_topo, :]
    differences_w_d_2 = off_wd_2 + list_water_d_model_2
    
    mean_d_2 = np.mean(differences_w_d_2)
    std_d_2 = np.std(differences_w_d_2)
    
    distance_region = np.array([mean_d_2 - std_d_2 * 3, mean_d_2 + std_d_2 * 3])
    
    list_little = differences_w_d_2 < distance_region[0] 
    list_big = differences_w_d_2 > distance_region[1]
    list_notwanted_topo = list_little + list_big  
    list_wanted_topo = []
    for no_nw in range(len(list_notwanted_topo)):
        nw = list_notwanted_topo[no_nw]
        if not nw:
            list_wanted_topo.append(no_nw) 
            
    off_coordinates_3 = off_coordinates_2[list_wanted_topo, :]
    off_wd_3 = off_wd_2[list_wanted_topo, :]
    
    list_water_d_model_3 = list_water_d_model_2[list_wanted_topo, :]
    differences_w_d_3 = off_wd_3 + list_water_d_model_3
    
    import Lossfunction
    print(Lossfunction.MAPE_np(-list_water_d_model_3, off_wd_3))
    print(Lossfunction.MAE_np(-list_water_d_model_3, off_wd_3))
    print(Lossfunction.R2_cal_np(-list_water_d_model_3, off_wd_3))
    np.savez('correction_stage3', 
             off_wd = off_wd_3,
             off_coordinates = off_coordinates_3,
             list_water_d_model_3 = list_water_d_model_3,
             readme = "This file contains the corrected results obtained after performing "
                 "three iterations of outlier removal based on the 3-sigma rule. "
                 "In each iteration, values falling outside the mean ± 3×standard deviation "
                 "were identified as outliers and excluded. "
                 "The arrays 'off_wd' and 'off_coordinates' represent the refined dataset "
                 "after this iterative cleaning process."
             )

lat_region = [0, 5, 10, 15, 20, 25, 30, 35]
lon_region = [120, 125, 130, 135, 140, 145, 150]
all_region = np.zeros([7, 6, 3]) # 以5度划分区域，每个区域有多少个点
nomber_region = np.zeros_like(off_coordinates) - 1 # 每个点对应的区域 
try: 
    region_information_ofdata3 = np.load('region_information_ofdata3.npz', allow_pickle=True)
    all_region = region_information_ofdata3['all_region']
    nomber_region = region_information_ofdata3['nomber_region']
    nomber_region_n = region_information_ofdata3['nomber_region_n']
except:
    for no_x in range(len(lat_region) - 1):
        print(no_x)
        all_region[no_x, :, 0] = lat_region[no_x]
        if no_x == 6:
            lat_n_s = off_coordinates[:, 0] <= (no_x+1) * 5
        else:
            lat_n_s = off_coordinates[:, 0] < (no_x+1) * 5
        # if no_x == 0:
        lat_n_b = off_coordinates[:, 0] >= no_x * 5
        # else:
            # lat_n_b = off_coordinates[:, 0] > no_x * 5
            
        lat_n = (lat_n_s.astype(int) & lat_n_b.astype(int))
        coordinates_region = off_coordinates[lat_n == 1, :]
        nomber_region[lat_n == 1, 0] = no_x * 5
        order_lat_n = np.where(lat_n)
        
        for no_y in range(len(lon_region)-1):
            print(no_y)
            all_region[:, no_y, 1] = lon_region[no_y]
            if no_y == 5:
                lon_n_s = coordinates_region[:, 1] <= (no_y+1) * 5 + 120
            else:
                lon_n_s = coordinates_region[:, 1] < (no_y+1) * 5 + 120
            lon_n_b = coordinates_region[:, 1] >= no_y * 5 + 120
            lon_n = (lon_n_s.astype(int) & lon_n_b.astype(int))
            all_region[no_x, no_y, 2] = np.sum(lon_n)
            
            order_lon_n = order_lat_n[0][lon_n == 1]
            nomber_data_ori = [lat_n == 1]
            nomber_region[order_lon_n, 1] = no_y * 5 + 120
    
    nomber_region_n = np.zeros([len(off_coordinates), 1])   
    for i in range(len(nomber_region_n)):
        nomber_region_n[i] = (nomber_region[i, 0]/5 ) * 6 + (nomber_region[i, 1]-120)/5 
    
    np.savez('region_information_ofdata3', all_region = all_region,
              nomber_region = nomber_region, nomber_region_n = nomber_region_n)

divide_region = np.zeros_like(all_region)
divide_region[:, :, 0] = np.floor(all_region[:, :, 2] * 0.6)
divide_region[:, :, 1] = np.floor(all_region[:, :, 2] * 0.2)
divide_region[:, :, 2] = all_region[:, :, 2] - divide_region[:, :, 0] - divide_region[:, :, 1]
divide_region = divide_region.astype(int)

order_all = []
nomber_region_n_temp = []
for i in range(len(nomber_region_n)):
    nomber_region_n_temp.append(int(nomber_region_n[i]))
    order_all.append(i)
dict_region_order = {}
for no_r in range(42):
    dict_region_order[str(no_r)] = []
for i in range(len(nomber_region_n_temp)):
    dict_region_order[str(nomber_region_n_temp[i])].append(order_all[i])

for no_r in range(42):
    print(len(dict_region_order[str(no_r)]))
    
for no_r in range(42):
    x_lat = int(np.floor(no_r / 6))
    x_lon = int(no_r % 6)
    # extract the real order of the coordiante points    
    list_all = dict_region_order[str(no_r)]
    
    # do the divide
    nom_ficher_list = np.arange(int(all_region[x_lat, x_lon, 2]))
    nom_ficher_train = np.random.choice(int(all_region[x_lat, x_lon, 2]), int(divide_region[x_lat, x_lon, 0]), replace = False)
    # remove the choosen one for train, because the selected one is the nomber of the choosen point from the list_all,
    # so np.delete could just remove the corresponding position
    nom_ficher_list_vali_test = np.delete(nom_ficher_list, nom_ficher_train)
    
    nom_ficher_vali = np.random.choice(int(all_region[x_lat, x_lon, 2]) - int(divide_region[x_lat, x_lon, 0]), int(divide_region[x_lat, x_lon, 1]),replace = False)
    nom_ficher_test = np.delete(nom_ficher_list_vali_test, nom_ficher_vali)
    
    dict_region_order[str(no_r) + 'train'] = []
    for i in range(int(divide_region[x_lat, x_lon, 0])):
        dict_region_order[str(no_r) + 'train'].append(list_all[nom_ficher_train[i]])
    dict_region_order[str(no_r) + 'vali'] = []
    for i in range(int(divide_region[x_lat, x_lon, 1])):
        dict_region_order[str(no_r) + 'vali'].append(list_all[nom_ficher_list_vali_test[nom_ficher_vali[i]]])
    dict_region_order[str(no_r) + 'test'] = []
    for i in range(int(divide_region[x_lat, x_lon, 2])):
        dict_region_order[str(no_r) + 'test'].append(list_all[nom_ficher_test[i]])

dict_region_order['wd'] = off_wd
dict_region_order['coordinates'] = off_coordinates
dict_region_order['divide_region'] = divide_region
dict_region_order['nomber_region_n'] = nomber_region_n

list_train = []
for no_r in range(42):
    list_train += dict_region_order[str(no_r) + 'train']
list_vali = []
for no_r in range(42):
    list_vali += dict_region_order[str(no_r) + 'vali']
list_test = []
for no_r in range(42):
    list_test += dict_region_order[str(no_r) + 'test']
dict_region_order['list_train'] = list_train
dict_region_order['list_vali'] = list_vali
dict_region_order['list_test'] = list_test
scio.savemat('divide_info3.mat', dict_region_order)
np.savez('train_file3', list_train = list_train,
          wd = off_wd, coordinates = off_coordinates)
np.savez('vali_file3', list_vali = list_vali,
          wd = off_wd, coordinates = off_coordinates)
np.savez('test_file3', list_test = list_test,
          wd = off_wd, coordinates = off_coordinates)


#%% check
# dict_region_order = scio.loadmat('divide_info3.mat')
# train_file3 = np.load('train_file3.npz', allow_pickle=True)
# list_train = train_file3['list_train']
# vali_file3 = np.load('vali_file3.npz', allow_pickle=True)
# list_vali = vali_file3['list_vali']
# test_file3 = np.load('test_file3.npz', allow_pickle=True)
# list_test = test_file3['list_test']
