#!/usr/bin/python
#-*- coding: UTF-8 -*- 
#coding=utf-8

# @author Oslo
# @version 2019-07-28.


from django.conf.urls import include, url
from django.contrib import admin
from . import views

urlpatterns = [
    # Examples:
    url(r'^$',views.home, name='home'),
    url(r'^api/$',views.input_img, name = 'input_img'),# 上传图片url
    #url(r'^admin/', include(admin.site.urls)),
]
from django.conf import settings
# 图片url
if settings.DEBUG is False:
    urlpatterns += [
        url(r'^images/(?P<path>.*)$','django.views.static.serve',{'document_root': settings.MEDIA_ROOT}),
    ]
# 404
handler404 = views.page_not_found