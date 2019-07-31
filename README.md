# py_torch_img

# 运行环境
```
PYTHON 3.5.X
```

# 相关路径修改
```
open  settings.py 
修改相关路径~
```

# 快速部署
```
cd py_torch_img
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:80
```

# API 地址
```
yuming/api/
```

# 请求方式
```
POST
```

# 快速Nginx部署 ubuntu/系统
```
#更新操作系统软件库
sudo apt-get update

#安装gcc g++的依赖库
sudo apt-get install build-essential
sudo apt-get install libtool

#安装 pcre依赖库
sudo apt-get install libpcre3 libpcre3-dev

#安装 pcre依赖库
sudo apt-get install zlib1g-dev

#安装 ssl依赖库
sudo apt-get install openssl

#安装uwsgi,如果是虚拟环境~，请在虚拟环境安装
sudo pip3 install uwsgi

# 安装完之后（如果是虚拟环境进入虚拟环境项目目录,manage.py 同级）
sudo vim mysite.xml

编辑或者复制进去，
<uwsgi>    
   <socket>127.0.0.1:8997</socket><!-- 内部端口，自定义 --> 
   <chdir>/data/py_torch/py_torch_img/</chdir><!-- 项目路径 -->            
   <module>py_torch_img.wsgi</module> 
   <processes>4</processes> <!-- 进程数 -->     
   <daemonize>uwsgi.log</daemonize><!-- 日志文件 -->
</uwsgi>

#安装nginx，并配置

/usr/local 目录执行下面命令下载nginx源码：
wget http://nginx.org/download/nginx-1.13.7.tar.gz

然后解压它：
tar -zxvf nginx-1.13.7.tar.gz

cd 进入解压后的nginx-1.13.7文件夹，依次执行以下命令：
sudo ./configure 
sudo make 
sudo make install

nginx一般默认安装好的路径为/usr/local/nginx

进入/usr/local/nginx/conf/目录，然后备份一下nginx.conf文件，以防意外。

sudo cp nginx.conf nginx.conf.bak
然后打开nginx.conf文件，把里面的内容全部删除，输入下面代码：

events {
    worker_connections  1024;
}
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    server {
        listen       80;
        server_name  www.django.cn;
        charset utf-8;
        location / {
           include uwsgi_params;
           uwsgi_pass 127.0.0.1:8997;
           uwsgi_param UWSGI_SCRIPT mysite.wsgi;
           uwsgi_param UWSGI_CHDIR /data/py_torch/py_torch_img/; #项目路径
           
        }
        #location /static/ {
        #alias /data/wwwroot/mysite/static/; #静态资源路径
        #}
    }
}

进入/usr/local/nginx/sbin/目录

执行下面命令先检查配置文件是否有错：
./nginx -t

没有错就执行以下命令：
./nginx

执行下面命令(在项目根目录，manage.py下，*如果项目在虚拟环境要在虚拟环境操作)：
uwsgi -x mysite.xml

以上步骤都没有出错的话。

进入/usr/local/nginx/sbin/目录

执行：

./nginx -s reload

```