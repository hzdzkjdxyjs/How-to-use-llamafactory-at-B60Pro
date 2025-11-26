# How-to-use-llamafactory-at-B60Pro

---

## 第0步：确定配置
 - 我遇到的很多ISV厂商客户都有反馈说无法运行B60显卡，主要是他们的主板并不支持B60显卡，B60显卡是很特殊的架构，如果是蓝戟的卡那是单芯，如果是铭瑄的卡是很独特的双芯架构，你需要确定你的主板支持x8x8pcie通道分离
 - 当你组装好服务器后，铭瑄的卡你需要在bios处设置开启ResizeBar以及将IIOPCIE启动为auto或x8x8
 - 蓝戟的卡仅需要在bios处设置开启ResizeBar即可，蓝戟的卡不需要确定支不支持x8x8因为他不需要通道分离

 Enable Re-Size BAR Support and PCIe Gen5 X8X8 as below:
<img width="1158" height="632" alt="image" src="https://github.com/user-attachments/assets/ea594ad5-a698-45d3-8998-ff2be6e983ea" />

<img width="1086" height="650" alt="image" src="https://github.com/user-attachments/assets/d764d296-ea73-4dc7-b046-75b20380ea87" />

<img width="1173" height="729" alt="image" src="https://github.com/user-attachments/assets/762aa6a7-9b31-4462-88b3-b89f0130b03b" />

<img width="1271" height="846" alt="image" src="https://github.com/user-attachments/assets/6beff878-e51f-4b47-9746-7ad7856e1efa" />

<img width="1044" height="705" alt="image" src="https://github.com/user-attachments/assets/0a761e9a-3c81-45d2-9d79-e1c9a5e41e03" />

<img width="1169" height="754" alt="image" src="https://github.com/user-attachments/assets/7b60018e-1dd8-4237-857b-a1dbb6f075c7" />

---


## 第1步：安装系统和基础环境配置环境

 - 您需要在服务器安装ubuntu系统25.04，并确认您的服务器版本的内核是25.04的原生内核
 - 接下来您需要安装驱动，安装驱动教程可以参考这个教程中的1.1部分Install Bare Metal Environment：https://github.com/intel/llm-scaler/blob/main/vllm/README.md/#1-getting-started-and-usagexit
 - 还要提到一点的是，b60的xpu-smi的监控不设置的话只支持root权限，需要设置
```bash
sudo gpasswd -a ${USER} render
sudo newgrp render
sudo reboot
````

---

## 第2步：安装项目和驱动

 - 您需要安装anaconda环境并安装相关依赖
 - 国内环境请设置国内pip镜像
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
````
 - 下载项目
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
````
 - 创建环境
```bash
conda create -n b60 python=3.10 -y
conda activate b60
````
 - 安装依赖
```bash
cd LLaMA-Factory
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
pip install -e ".[metrics]" --no-build-isolation
````
 - 检查torch安装情况，和我一样就是安装完成了
```bash
(B60) root@b60:~/ultralytics# python
Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.version.xpu)
20250101
>>> print(torch.xpu.is_available())
True
>>> print(torch.xpu.get_device_name(0))
Intel(R) Graphics [0xe211]
````
 - 检查安装情况
```bash
(b60) root@b60:~/LLaMA-Factory# llamafactory-cli env

- `llamafactory` version: 0.9.4.dev0
- Platform: Linux-6.14.0-1006-intel-x86_64-with-glibc2.41
- Python version: 3.10.19
- PyTorch version: 2.8.0+xpu
- Transformers version: 4.57.1
- Datasets version: 4.0.0
- Accelerate version: 1.11.0
- PEFT version: 0.17.1
- TRL version: 0.9.6
- Git commit: 591fc9ed025100c40a2687431a7d83a19978e42d
- Default data directory: detected
````
