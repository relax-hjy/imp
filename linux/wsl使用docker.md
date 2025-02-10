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