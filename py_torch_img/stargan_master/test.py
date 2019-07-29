#!/usr/bin/python
#-*- coding: UTF-8 -*- 
#coding=utf-8

# @author Oslo
# @version 2019-07-28.

import sys
from main_arr import main_arr

if __name__ == '__main__':
    input_dir = sys.argv[0]
    print ('584844')
    print (input_dir)
    imagename = sys.argv[1]
    model_save_dir = sys.argv[2]
    selected_attrs = sys.argv[3]
    result_dir = sys.argv[4]

    main_arr(input_dir,imagename,model_save_dir,selected_attrs,result_dir)
