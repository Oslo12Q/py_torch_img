#!/usr/bin/python
#-*- coding: UTF-8 -*- 
#coding=utf-8

# @author Oslo
# @version 2019-07-28.


import os
import subprocess
from django.conf import settings

'''
# 参数 动作/00
# 获取每个动作的 value 值
# return  False/None、 True/value
'''
def command_action(action):
    try:
        if action == '00':
            return True,'Black_Hair'
        elif action == '01':
            return True,'Blond_Hair'
        elif action == '02':
            return True,'Brown_Hair'
        elif action == '03':
            return True,'Gray_Hair'
        elif action == '10':
            return True,'Yong'
        elif action == '11':
            return True,'old'
        elif action == '20':
            return True,'Male'
        elif action == '21':
            return True,'Female'
        else:
            return False,None
    except Exception as err:
        return False,None

'''
# 参数： mac/appid  mac地址/小程序用户id
# 根据参数创建输入图片目录
# return 路径
'''
def input_path(mac_app_id):
    file_path = settings.INPUT_PATH + mac_app_id
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    return file_path

'''
# 参数： mac/appid  mac地址/小程序用户id
# 根据参数创建输出图片目录
# return 路径
'''
def output_path(mac_app_id):
    file_path = settings.OUTPUT_PATH + mac_app_id
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    return file_path

'''
# 参数图片名称,图片保存的路径
# 根据图片名称检索new图片
# return path
'''
def detect_ready(fid,output_path):
    for extension in settings.IMG_OUTPUT_EXTENSIONS:
        path = os.path.join(output_path, '{}{}'.format(fid, extension))
        if os.path.exists(path):
            return path

'''
# 命令行执行脚本
'''
def Cmd(input_dir,imagename,model_save_dir,selected_attrs,result_dir):
    cmd = u'python3 /data/py_torch/py_torch_img/py_torch_img/stargan_master/test.py {input_dir} {imagename} {model_save_dir} {selected_attrs} {result_dir}'.format(input_dir=input_dir, imagename=imagename,model_save_dir=model_save_dir,selected_attrs=selected_attrs,result_dir=result_dir)
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    for i in out.splitlines():
        print (i)