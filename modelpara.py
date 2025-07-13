# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:26:18 2024

@author: Jia_G
"""
import os
import shutil
def fichier_creat(name):
    """
    maske sure that the fichier named as name is existe, creat it if isn't
    """
    folder = os.path.exists(name)
    if not folder:
        os.makedirs(name)
        print('folder' + str(name) + 'created')
    else:
        print('folder' + str(name) + 'existed')

def fichier_copy(src_file, dst_file):
    shutil.copy(src_file, dst_file)  # 复制文件
    new_file = os.path.join(os.path.dirname(dst_file), dst_file)  # 新的文件名
    # os.rename(dst_file, new_file)  # 重命名文件
    