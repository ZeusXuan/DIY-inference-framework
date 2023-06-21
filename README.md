# ONE:安装

## 使用镜像
●	docker pull ubuntu拉取镜像
●	docker container run -it -v E:\projecthub\reasoning-framework:/reasoning-framework --name=reasoning-framework /bin/bash 实现本地挂载

## 安装Armadillo
●	apt update
●	apt install cmake libopenblas-dev liblapack-devlibarpack2-dev libsuperlu-dev
●	对armadillo进行解压进入该目录
●	mkdir build
●	cd build
●	cmake ..
●	make -j8
●	make install 编译后进行安装
●	头文件安装在：/usr/include/
●	库文件安装在：/usr/lib/x86_64-linux-gnu

##  安装Glog日志库和GTest测试库
●	https://github.com/google/googletest
●	https://github.com/google/glog
●	与安装Armadillo步骤相同
●	库文件在： /usr/local/lib/
●	头文件：/usr/local/include/

##  测试
测试Armadillo的安装:
mkdir bulid
cd build
cmake ..
make -j8




