1. 开启并更新wsl服务

2. 安装操作系统ubuntu

3. ubuntu安装docker配置

   [【WSL】Ubuntu 22.04 安装配置docker_wsl ubuntu docker-CSDN博客](https://blog.csdn.net/qq_37387199/article/details/129100486)

4. 配置ubuntu系统的代理

   [window子系统 wsl2 ubuntu子系统配置代理_windows子系统ubuntu设置代理-CSDN博客](https://blog.csdn.net/qq_41614928/article/details/111941978#:~:text=本文介绍了如何让Ubuntu子系统利用Windows的IPV4地址设置http_proxy和https_proxy，实现通过Windows代理来加速软件下载和安装。,首先查看Windows和Ubuntu子系统的IP地址，然后在Ubuntu中配置相应的代理设置，最后验证代理配置是否成功。 注意，WSL2的IP地址可能在重启后改变，并且需确保Clash代理服务的局域网许可已开启。)

5. 配置docker拉取镜像的代理

​		[如何优雅的给 Docker 配置网络代理 - CharyGao - 博客园](https://www.cnblogs.com/Chary/p/18096678)

```shell
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
Environment="HTTP_PROXY=http://172.28.80.1:7890/"
Environment="HTTPS_PROXY=http://172.28.80.1:7890/"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```



[使用docker-compose安装Milvus向量数据库及Attu可视化连接工具_docker部署milvus attu-CSDN博客](https://blog.csdn.net/weimeilayer/article/details/144351466)



### 在本机访问ubuntu的服务

![image-20250208150802471](./assets/image-20250208150802471.png)

- 在ubuntu中打开ipconfig。
- 图中ip即为本机访问ubuntu所需要的ip。

ps:

[GitHub - doccano/doccano: Open source annotation tool for machine learning practitioners.](https://github.com/doccano/doccano)



### 写在nVidia驱动包

sudo apt-get --purge remove "*nvidia*"

### 在ubuntu系统安装nvidia-container-toolkit

再很多镜像中需要使用主机显卡驱动，但镜像没有安装驱动，如pytorch

PyTorch 的 Docker 镜像主要包含以下内容：

- PyTorch 框架及其依赖（如 Python、CUDA 库、cuDNN 等）。
- 可选的其他工具（如 Conda、Jupyter、TorchVision、TorchAudio 等）

### 2. **显卡驱动的作用**

显卡驱动（如 NVIDIA 驱动）是操作系统与 GPU 硬件之间的桥梁，负责管理 GPU 的计算和图形渲染功能。PyTorch 依赖显卡驱动来访问 GPU 资源。

### 3. **为什么 PyTorch 镜像不包含显卡驱动？**

- **操作系统依赖**：显卡驱动需要与操作系统的内核版本匹配，通常需要在主机系统上安装。
- **硬件依赖**：不同的 GPU 型号需要不同的驱动程序，无法在镜像中统一提供。
- **安装复杂性**：显卡驱动的安装和配置通常需要管理员权限，不适合在容器中直接安装。

### 4. **如何在 Docker 中使用 GPU？**

要在 Docker 中使用 GPU，需要满足以下条件：

1. 安装显卡驱动：
   - 在主机系统上安装与 GPU 型号匹配的显卡驱动（如 NVIDIA 驱动）。
   - 验证驱动是否安装成功：运行 `nvidia-smi` 命令，查看 GPU 信息。
2. 安装 NVIDIA Container Toolkit：
   - 安装 NVIDIA Container Toolkit，使 Docker 能够访问 GPU。

安装方法：

```python
# 添加 NVIDIA 包仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装 NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit# nvidia-docker2
# 重启 Docker
sudo systemctl restart docker
```

**运行 Docker 容器时启用 GPU**：

- 使用 `--gpus` 参数指定 GPU 资源。

- 示例：

  ```bash
              bash复制
              docker run --gpus all pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime nvidia-smi
  ```

### 5. **PyTorch 镜像中的 CUDA 和 cuDNN**

PyTorch 镜像中包含了 **CUDA 运行时库** 和 **cuDNN**，这些是 PyTorch 运行 GPU 计算所需的软件库。但它们并不替代显卡驱动，显卡驱动仍然需要在主机系统上单独安装。



### 6. **总结**

- PyTorch 的 Docker 镜像 **不包含显卡驱动**。
- 要在 Docker 中使用 GPU，必须在主机系统上安装显卡驱动，并配置 NVIDIA Container Toolkit。
- PyTorch 镜像中包含了 CUDA 和 cuDNN 库，这些库与显卡驱动配合工作，实现 GPU 加速。

如果你不确定如何安装显卡驱动或配置 GPU 支持，可以参考 NVIDIA 官方文档或 PyTorch 官方指南。



*官方文档*

[Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)