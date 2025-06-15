[Git基本使用教程（一）：入门及第一次基本完整操作_git的使用-CSDN博客](https://blog.csdn.net/qq_35206244/article/details/97698815)

[Git 详细安装教程（详解 Git 安装过程的每一个步骤）_git安装-CSDN博客](https://blog.csdn.net/mukes/article/details/115693833)





## git常用功能介绍

设置key并且测试连接，由successfully即为成功

```bash
git config --global user.name "注册名"

git config --global user.email "注册邮箱"

ssh-keygen -t rsa -C "自己的邮箱" #C:/User/用户/.ssh下，id_rsa为私钥，id_rsa.pub为公钥
ssh -T git@github.com
```

git 网络设置

```bash
git config --global http.proxy "http://127.0.0.1:7890"
```

### 1. 克隆远程仓库（如果尚未克隆）

如果还没有本地仓库，首先需要克隆远程仓库：

```shell
git clone <远程仓库URL>
```

这会将远程仓库的内容复制到本地。



### 2. 同步远程仓库到本地

#### 方法一：使用 `git fetch` 和 `git merge`

1. **获取远程更新**
   使用 `git fetch` 获取远程仓库的最新更改，但不会自动合并到本地分支：

   ```shell
   git fetch origin
   ```

2. **查看远程分支状态**
   可以通过以下命令查看远程分支的状态：

   ```shell
   git status
   ```

3. **合并远程分支到本地分支**
   使用 `git merge` 将远程分支的更改合并到当前本地分支：

   ```
   git merge origin/<分支名>
   ```

   例如，合并 `origin/main` 到本地 `main` 分支：

   ```shell
   git merge origin/main
   ```

#### 方法二：使用 `git pull`

`git pull` 是 `git fetch` 和 `git merge` 的组合命令，可以直接拉取并合并远程更改：

```bash
git pull origin <分支名>
```

例如，拉取并合并 `origin/main` 分支：

```bash
git pull origin main
```

### 3. 处理冲突

如果远程仓库和本地仓库有冲突，Git 会提示你解决冲突。你需要手动编辑冲突文件，然后标记冲突已解决：

1. 编辑冲突文件，解决冲突。

2. 将解决后的文件添加到暂存区：

   ```bash
   git add <文件名>
   ```

3. 提交更改：

   ```bash
   git commit
   ```

### 4. 推送本地更改到远程仓库（可选）

如果你在本地做了更改并希望同步到远程仓库，可以使用 `git push`：

```bash
git push origin <分支名>
```

例如，推送本地 `main` 分支到远程仓库：

```bash
git push origin main
```

### 总结

- 使用 `git fetch` 获取远程更新，然后手动合并。
- 使用 `git pull` 直接拉取并合并远程更改。
- 解决冲突后，推送本地更改到远程仓库。
- `origin` 是 Git 中默认的远程仓库别名，指向你克隆的远程仓库。

通过这些步骤，你可以轻松将远程仓库同步到本地。



## 一般流程

克隆仓库提交修改

连上平台的账号

```
git clone 仓库

修改内容

然后在该仓库下执行下面内容：

提交所有变化
git add ./*  



提交并写注释
git commit -m ""   


同步到远程仓库
git push -u origin main
如果报错，可能是没有关联仓库，需要添加关联再同步到远程。
git remote add origin 
```

