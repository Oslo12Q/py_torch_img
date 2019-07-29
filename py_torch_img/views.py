#!/usr/bin/python
#-*- coding: UTF-8 -*- 
#coding=utf-8

# @author Oslo
# @version 2019-07-28.


import os
import logging
import traceback
import threading
import time
import json
import base64
import random
import datetime
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

from .config import *
from py_torch_img.stargan_master.main_arr import main_arr


'''
# 根url
'''
def home(request):
    return HttpResponse("<h1>建设中</h1>") 

# json时间处理
class DateEncoder(json.JSONEncoder):
    def default(self, obj):  
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')   
        elif isinstance(obj,date):
            return obj.strftime('%Y-%m-%d')
        else:    
            return json.JSONEncoder.default(self, obj)

# 通用函数,返回json数据
def get_json_response(request, json_rsp):
    return HttpResponse(json.dumps(json_rsp,cls=DateEncoder), content_type='application/json')

'''
# 上传图片api
# 请求方式： POST
# 参数： im_id/图片key, mac_app_id/mac地址、小程序用户id, command/动作
'''
def input_img(request):
    try:
        if request.method != 'POST': 
            return get_json_response(request, dict(suc_id=0, ret_cd=405, ret_ts=int(time.time()),errorMsg = 'Method not allowed',im_id='',successResult=''))
        img_file = request.FILES.get("im_id", None)
        if not img_file:
            return get_json_response(request, dict(suc_id=0, ret_cd=104, ret_ts=int(time.time()),errorMsg = 'The parameter format is not correct',im_id='', successResult=''))
        mac_app_id = request.POST.get('mac_app_id',None)
        if not mac_app_id:
            return get_json_response(request, dict(suc_id=0, ret_cd=104, ret_ts=int(time.time()),errorMsg = 'Please submit the upload mac_app_id',im_id='', successResult=''))
        command = request.POST.get('command',None)
        if not command:
            return get_json_response(request, dict(suc_id=0, ret_cd=104, ret_ts=int(time.time()),errorMsg = 'Please submit the upload command',im_id='', successResult=''))
        flag,command_value = command_action(command) #获取每个动作的value值
        if flag is False:
            return get_json_response(request, dict(suc_id=0, ret_cd=104, ret_ts=int(time.time()),errorMsg = 'This action does not exist',im_id='', successResult=''))
        
        input_paths = input_path(mac_app_id) # 创建输入目录
        print (input_paths)
        out_paths = output_path(mac_app_id) # 创建输出目录
        print (out_paths)

        file_obj_base = base64.b64encode(img_file.read()) #读取文件内容，转换为base64编码   
        img_name = '{}_{}.jpg'.format(int(time.time()),random.randint(1000, 9999),)
        original_image_dest = input_paths +'/'+'{}'.format(img_name)
        
        file_objects = base64.b64decode(file_obj_base) #base64转化为图片
        original_image = open(original_image_dest, 'wb+')
        original_image.write(file_objects)
        original_image.close()
        print (input_paths,img_name,settings.MODEL_PATH,[command_value],out_paths)
        results = main_arr(input_paths,img_name,settings.MODEL_PATH,[command_value],out_paths)
        def wait_ready(img_name,out_paths):
            for i in range(10):
                ret = detect_ready(img_name,out_paths)
                if ret:
                    return ret
                time.sleep(1)
            return ''
        # 检索生成的new图片
        detect_path = wait_ready(img_name,out_paths)
        print (detect_path)
        return get_json_response(request, dict(suc_id=0, ret_cd=200, ret_ts=int(time.time()),errorMsg = '',im_id=img_name,successResult=''))

    except Exception as err:
        logging.error(err)
        logging.error(traceback.format_exc())
        return get_json_response(request, dict(suc_id=0, ret_cd=500, ret_ts=int(time.time()),errorMsg = 'Server internal error',im_id='',successResult=''))



'''
# 404页面
'''
def page_not_found(request):
    return HttpResponse("<h1>404</h1>") 

