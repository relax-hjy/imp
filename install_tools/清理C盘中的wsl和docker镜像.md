[ 轻松搬迁！教你如何将WSL从C盘迁移到其他盘区，释放存储空间！](https://zhuanlan.zhihu.com/p/621873601)    

[如何查看WSL默认安装位置以及挪动其到指定安装路径](https://blog.csdn.net/tassadar/article/details/142407262)   

两份资料的核心内容其实完全一致，都是教你把默认装在 C 盘的 WSL 发行版“搬家”到别的分区，以节省 C 盘空间。下面把关键信息合并成 4 句话的极简步骤，方便你一眼看懂：

1. **先关机**：在 PowerShell / CMD 里执行 `wsl --shutdown` 确认状态为 Stopped。  
2. **导出/注销/导入**（一条线完成）：  
   • 导出：`wsl --export <发行版名> D:\目标目录\backup.tar`  
   • 注销：`wsl --unregister <发行版名>`  
   • 导入：`wsl --import <发行版名> D:\目标目录 D:\目标目录\backup.tar`  
3. **恢复原先用户名**：`Ubuntu2204 config --default-user <原用户名>`（发行版名数字间无连字符）。  
4. **验证**：`wsl -d <发行版名>` 启动，一切正常即可删除 C 盘旧目录 `%LOCALAPPDATA%\Packages\CanonicalGroupLimited...`。

一句话总结：用 `export` 把系统打包带走，再用 `import` 在新位置恢复，最后把默认用户设回原来的自己即可。