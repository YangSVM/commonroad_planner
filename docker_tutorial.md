# learning docker
1. 安装：https://docs.docker.com/get-started/. 点击链接到linux的下载地址。https://docs.docker.com/engine/install/。 再选择Ubuntu的下载地址，https://docs.docker.com/engine/install/ubuntu/，使用第一种安装方法：Install using the repository。其中Install Docker Engine 中的2.不用执行，因为不需要安装特定版本的docker.
2. 安装完后，进入下一个教程，使得docker能够在non-root用户下使用。https://docs.docker.com/engine/install/linux-postinstall/。
    ```bash
      sudo groupadd docker
      sudo usermod -aG docker $USER
      newgrp docker 
      docker run hello-world
    ```
    完成后**重启电脑**。此时新开一个终端，直接运行`docker run hello-world`也不会报错
3. Start the tutorial [https://docs.docker.com/get-started/#start-the-tutorial]
4. 基本概念
   1. container(容器)。容器相当于创建了新的独立与其他所有进程的环境。利用了linux已有的内核的命名空间和cgroups等特性完成。
   2. container image(容器镜像)。容器使用独立的文件系统。需要将程序需要的文件、代码、依赖包、甚至环境变量、a default command to run, and other metadata等，放置于image中。
5. 尝试 python 版本的practice[https://docs.docker.com/language/python/]
   1. build image
      1. 新建一个conda环境。按照流程安装flask，并编写app.py，能够成功运行web程序。
      2. 学会编写 Dockerfile
         1. 定义：Dockerfile是`docker build`命令读取并执行该文件，组装相应的文件构成容器镜像。
         2. 详见Dockerfile。定义格式->添加并安装 requirements.txt 中的python包->添加源文件->添加默认命令
      3. 创建image：`docker build --tag python-docker .`
      4. 查看已有image: `docker images`
   2. Run your image as a container
      1. `docker run -p 5000:5000 python-docker` -p [host port]:[container port]. docker网络也完全隔离。应用需要进行端口映射，才能正常使用
      2. `docker run -dp 5000:5000 python-docker` -d: 表示能够后台运行
      3. `docker ps` 可以查看正在后台运行的docker程序
6. 其他docker常用命令


# running docker
1. 镜像包示例[https://gitlab.lrz.de/tum-cps/commonroad-docker-submission]
2. build
   ```bash
   docker build --tag yangyibin98/dummy-image:latest .

   ```
3. test
   ```bash
   docker images
   ```
   应该输出有 yangyibin98/dummy-image:latest 为名称的包。而不是None。
4. run
   ```bash
    docker run \
      --rm \
      -v $(pwd)/scenarios:/commonroad/scenarios \
      -v $(pwd)/solutions:/commonroad/solutions \
      yangyibin98/tongji-tsinghua-cr-planner:v1.0

   ```
5. docker上传。
   1. 首先在docker hub[https://hub.docker.com/signup]注册账号。
   2. 上传镜像
      ```bash
      docker login
      # 重命名技巧
      # docker tag yangyibin/dummy-image:latest yangyibin98/dummy-image:v1
      docker push yangyibin98/dummy-image:v1
      ```
      第二句是将docker 镜像添加新名字。docker push仅能push 以自己账号开头的image。
      其中 yangyibin98 需改为 docker hub注册的账号
   3. 