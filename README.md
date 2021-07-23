# commonroad_planner
common road TUM竞赛


## 安装步骤
- bash脚本
```bash
# 安装conda新环境
conda create -n cr37 python=3.7
pip install commonroad-io
# drivability-checker
git clone https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker.git
# 先安装依赖包
sudo apt install libomp5
sudo apt install libomp-dev
# 注意thrid-party目录中可能因为网络原因导致空目录。此时需要手动安装。
bash build.sh -e /home/thicv/miniconda3/envs/cr37 -v 3.7 --cgal --serializer -i -j 2

# 安装interactive-scenario
pip install sumocr

# minimal example生成视频需要安装ffmpeg
conda install -c conda-forge ffmpeg

# 安装route-planner
pip install 
```
-  下载比赛包地址
https://nextcloud.in.tum.de/index.php/s/9Lzy6a3a3P7gHzT
-  ``
- bash 