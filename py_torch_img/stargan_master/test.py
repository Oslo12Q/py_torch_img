#!/usr/bin/python
#-*- coding: UTF-8 -*- 
#coding=utf-8

# @author Oslo
# @version 2019-07-28.

import os
from .main_arr import main_arr

if __name__ == '__main__':
    input_dir = os.args[0]
    print ('584844')
    print (input_dir)
    imagename = os.args[1]
    model_save_dir = os.args[2]
    selected_attrs = os.args[3]
    result_dir = os.args[4]

    main_arr(input_dir,imagename,model_save_dir,selected_attrs,result_dir)
