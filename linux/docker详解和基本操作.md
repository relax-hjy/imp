## 基本流程

#### 使用容器中的服务

1.查询已有的镜像,运行中的容器，挂起的容器

docker images # 

docker ps

docker ps -a

2.根据情况启动容器和服务

- 如果容器已经运行，无需操作

- 如果服务没有在运行也没有挂起，那么通过镜像启动容器和服务

  ```shell
  docker run -d -p 5672:5672 -p 15672:15672 --name rabbitmq 80bd4b95a49d   #docker run -d 后台运行  -p 端口映射  --name 容器别名 最后是容器id
  ```

- 如果服务的容器挂起，那么启动容器即可

  ```shell
  docker start 
  ```

3.查询启动中的容器，关注容器名，容器id,容器映射的端口。

​	docker ps