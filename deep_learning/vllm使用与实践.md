vllm安装

首先需要 python环境 和 cuda（非torch的runtime，vllm需要用cuda编译）

大概率需要安装cuda，因为需要nvcc来编译，用pytorch的runtime来跑代码。

依赖pytorch



本机cuda版本<=pytorch cuda版本==vllm cuda版本 <=驱动版本



-i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu128



首先下载cuda，不行就降低版本。

vllm cuda版本在GitHub下载

pip install vllm时指名 torch源 ，cuda版本torch会一同下载好。



vllm 0.9.0以上都需要 torch 2.7.0以上



pytorch 2.7.1显示不再支持  7.0架构的V100





pytorch 2.6.0cu126   vllm0.8.5post1会报错导入不了vllm  

这个错误通常是由于 vLLM 和 PyTorch 版本不兼容导致的。符号 `_ZN5torch3jit17parseSchemaOrNameERKSsb` 是 PyTorch JIT 模块的一部分，未找到该符号表明 vLLM 编译时使用的 PyTorch 版本与当前环境中安装的版本不一致。



pytorch 2.6.0cu124  和 vllm0.8.5post1 兼容



[GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)



[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive/)



[vLLM 安装记录 （含踩坑xformers）_failed to build xformers-CSDN博客](https://blog.csdn.net/zt1091574181/article/details/145551163)

[2025 最新 DeepSeek-R1-Distill-Qwen-7B vLLM 部署全攻略：从环境搭建到性能测试(V100-32GB)-CSDN博客](https://blog.csdn.net/MnivL/article/details/145471466)











[vLLM环境安装与运行实例【最新版（0.6.4.post1）】_vllm 安装-CSDN博客](https://blog.csdn.net/yd778473278/article/details/144077743)



[fairseq-0.12.2多机训练环境搭建_fairseq python3.9-CSDN博客](https://blog.csdn.net/yd778473278/article/details/134372632?fromshare=blogdetail&sharetype=blogdetail&sharerId=134372632&sharerefer=PC&sharesource=yd778473278&sharefrom=from_link)





[vLLM CPU和GPU模式署和推理 Qwen2 等大语言模型详细教程 - 老牛啊 - 博客园](https://www.cnblogs.com/obullxl/p/18353447/NTopic2024081101)



[(99+ 封私信 / 81 条消息) vLLM使用指北 - 知乎](https://zhuanlan.zhihu.com/p/685621164)



GGUF

https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/?tl=zh-hans



P40成功案例

https://blog.csdn.net/qq_42755230/article/details/144184284

https://www.cnblogs.com/boydfd/p/18606571





```
apt list -a <package_name>
```

```
apt list --installed <package_name>
```





### 总结与最佳实践

| 你的需求                     | 推荐命令                      | 示例                            |
| :--------------------------- | :---------------------------- | :------------------------------ |
| **查看某个包的所有可用版本** | `apt list -a <包名>`          | `apt list -a nginx`             |
| **查看某个包已安装的版本**   | `apt list --installed <包名>` | `apt list --installed nginx`    |
| **详细分析版本来源和优先级** | `apt policy <包名>`           | `apt policy nginx` **(最常用)** |
| **查看所有已安装的软件包**   | `dpkg -l`                     | `dpkg -l | grep python`         |
| **查看 apt 工具自己的版本**  | `apt -v`                      | `apt -v`                        |



#### 启动embedding模型

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /data/dev/models_saved/deployed_embedding \
--port 31000 \
--trust-remote-code \
--max-model-len 512 \
--served-model-name qwen3-embedding \
--gpu-memory-utilization 0.5 \
--task  embed \
--dtype half \
--tensor-parallel-size 4  \
--max-num-batched-tokens 8192 \
--max-num-seqs 16
```

| 参数                       | 推荐值        | 作用                                                         | 备注                         |
| -------------------------- | ------------- | ------------------------------------------------------------ | ---------------------------- |
| `--gpu-memory-utilization` | 0.95          | 把 32 GB 显存吃满，避免浪费，控制 GPU 内存的使用比例，取值 0～1（默认 0.9）。如果要并发量更大，可以调高，比如 `0.95`。 | 常用                         |
| `--enable-chunked-prefill` |               | 把长 prompt 拆块，显著降低 TTFT                              | v0.6+ 版本支持               |
| `--max-num-batched-tokens` | 4096          | 控制服务端一次 GPU 批处理的 token 上限，影响显存占用和吞吐量。 | 常用                         |
| `--max-num-seqs`           | 8～16         | 同时处理的请求条数                                           | 常用，与上一条联动           |
| `--enforce-eager`          | false（默认） | 用 CUDA Graph 减少 kernel 启动开销                           | 只有显存吃紧时才改成 true    |
| `--dtype`                  | float16       | 半精度推理                                                   | V100 支持 Tensor Core        |
| `--quantization`           | gptq / awq    | 加载 4bit/8bit 量化模型                                      | 显存直接减半，v100应该不支持 |
| max-num-seqs               |               |                                                              |                              |
| max-model-len              |               | **避免分配不必要的KV Cache显存**                             |                              |

curl调用

```shell
curl http://localhost:12000/v1/embeddings   -H "Content-Type: application/json"   -d '{
    "input": "你好，世界",
    "model": "qwen3-embedding"
  }'
  
curl http://localhost:12000/v1/embeddings   -H "Content-Type: application/json"   -d '{
    "input": ["你好，世界","你好，打工人"],
    "model": "qwen3-embedding"
  }'
```



#### 启动chat模型

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

models = client.models.list()
model = models.data[0].id
prompt = "描述一下北京的秋天"

completion = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=256,        # 关键：给足生成长度
    temperature=0.7,
    stop=None              # 也可以根据需要设置 stop 序列
)
res = completion.choices[0].text.strip()
print(f"Prompt: {prompt}\nResponse: {res}")
```

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

models = client.models.list()
model = models.data[0].id
messages = [
    {"role": "user", "content": "描述一下北京的秋天"}
]

chat_completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=256,
    temperature=0.7
)
res = chat_completion.choices[0].message.content.strip()
print(f"Prompt: {messages[0]['content']}\nResponse: {res}")
```

```sh
python3 -m vllm.entrypoints.openai.api_server \
--model /data/dev/models_pre/Qwen3-30B-A3B-Instruct-2507 \
--host 0.0.0.0 \
--port 8080 \
--max-num-seqs 2 \
--max-model-len 512 \
--tensor-parallel-size 4 \
--max-num-batched-tokens 1024 \
--gpu-memory-utilization 0.90  \
--dtype float32


python3 -m vllm.entrypoints.openai.api_server \
--model /data/dev/models_pre/gpt-oss-20B \
--host 0.0.0.0 \
--port 8080 \
--max-num-seqs 4 \
--max-model-len 512 \
--tensor-parallel-size 4 \
--max-num-batched-tokens 1024 \
--gpu-memory-utilization 0.95  \
--quantization awq \
--dtype float32
```

```
编译自己的moe编码
python -m vllm.model_executor.layers.fused_moe.tuning \
  --model /data/dev/models_pre/Qwen3-30B-A3B-Instruct-2507 \
  --tp 4 \
  --dtype float32 \
  --output /root/anaconda3/envs/vllm_271cu18/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/
```

python3 -m vllm.entrypoints.openai.api_server \
> --model /data/dev/models_pre/Qwen3-30B-A3B-Instruct-2507 \
> --host 0.0.0.0 \
> --port 8080 \
> --max-num-seqs 16 \
> --max-model-len 512 \
> --tensor-parallel-size 4 \
> --max-num-batched-tokens 2048 \
> --gpu-memory-utilization 0.95  \
> --dtype float32

python3 -m vllm.entrypoints.openai.api_server --model /data/dev/models_pre/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8080 --dtype auto --max-num-seqs 16 --max-model-len 1024 --tensor-parallel-size 1 --max-num-batched-tokens 2048   --dtype float16

参数列表解释

| 参数                       | 推荐值        | 作用                                                         | 备注                         |
| -------------------------- | ------------- | ------------------------------------------------------------ | ---------------------------- |
| `--gpu-memory-utilization` | 0.95          | 把 32 GB 显存吃满，避免浪费，控制 GPU 内存的使用比例，取值 0～1（默认 0.9）。如果要并发量更大，可以调高，比如 `0.95`。 | 常用                         |
| `--enable-chunked-prefill` |               | 把长 prompt 拆块，显著降低 TTFT                              | v0.6+ 版本支持               |
| `--max-num-batched-tokens` | 4096          | 控制服务端一次 GPU 批处理的 token 上限，影响显存占用和吞吐量。 | 常用                         |
| `--max-num-seqs`           | 8～16         | 同时处理的请求条数                                           | 常用，与上一条联动           |
| `--enforce-eager`          | false（默认） | 用 CUDA Graph 减少 kernel 启动开销                           | 只有显存吃紧时才改成 true    |
| `--dtype`                  | float16       | 半精度推理                                                   | V100 支持 Tensor Core        |
| `--quantization`           | gptq / awq    | 加载 4bit/8bit 量化模型                                      | 显存直接减半，v100应该不支持 |

服务器收到了多个客户端请求后：

- vLLM 会合并请求，打一个 GPU batch。
- 这个 batch 必须同时满足：
  - **序列数 ≤ --max-num-seqs**
  - **token 总数 ≤ --max-num-batched-tokens**

# 仍在源码根目录 vllm/ 下
python benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model models_pre/Qwen3-4B-Instruct-2507  \
  --host 127.0.0.1 --port 8080 \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --request-rate 50 \
  --max-concurrency 8 \
  --num-prompts 1 \
  --random-input-len 256 \
  --random-output-len 128 








>     nproc_per_node=8
>     NPROC_PER_NODE=$nproc_per_node \
>     swift sft \
>         --model models_pre/Qwen3-Embedding-0.6B \
>         --task_type embedding \
>         --model_type qwen3_emb \
>         --train_type full \
>         --dataset extr_wuliao/data_train_qwen_embedding.json \
>         --split_dataset_ratio 0.05 \
>         --eval_strategy steps \
>         --output_dir models_saved \
>         --eval_steps 400 \
>         --num_train_epochs 2 \
>         --save_steps 400 \
>         --per_device_train_batch_size 4 \
>         --per_device_eval_batch_size 4 \
>         --gradient_accumulation_steps 4 \
>         --learning_rate 6e-6 \
>         --loss_type infonce \
>         --label_names labels \
>         --dataloader_drop_last true \
>         --deepspeed zero3 \
>         --max_length 512

nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model models_pre/Qwen3-Embedding-4B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb:positive \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir delete \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3



模型调优方向

| 训练轮数        | 提高到 3~5，或更高                   |
| --------------- | ------------------------------------ |
| warmup & 学习率 | 加入 warmup_ratio 和 scheduler 类型  |
| batch 调优      | 增大 accumulation，或 batch size     |
| 模型规模        | 若资源允可，尝试更大 Qwen3 emb 模型  |
| 损失函数调研    | 不止 InfoNCE，可尝试 supcon、triplet |
| 数据质量        | 清洗、平衡、多样化                   |
| 日志监控        | 加 TensorBoard / wandb，提高可见性   |





### 1. 指令（Instruction）

- **定义**：CPU真正执行的“命令”。CPU 架构手册里明确定义的某个抽象操作，手册中用自然语言定义方便开发者理解。
- **本质**：是**逻辑意义上的操作**，还不是二进制。
- 例子：将寄存器 R1 与 R2 相加，结果写回 R1。
  对应汇编助记符：`ADD R1, R1, R2`。 

### 2. 机器码（Machine Code）

- **定义**：指令的**二进制表示形式**，是CPU真正能识别并执行的格式。
- **本质**：就是**一串二进制数字**，比如 `11001010 00011011`。
- 例子：`11010000 11001001`

### 3. 汇编语言（Assembly Language）

- **定义**：机器码的**人类可读形式**，用助记符（如 `ADD`, `MOV`）代替二进制。

- **本质**：是**文本形式的机器码**，需要汇编器翻译成机器码。

- 例子：

  ```asm
  MOV R1, #5
  MOV R2, #3
  ADD R1, R2
  ```



#### **三者的关系**

cpu因为其架构设计，有了可执行指令集合，并对外开放机器码集合作为媒介。

当人要指挥cpu工作时通过对cpu发出指令实现。在最底层，发出指令主要有 机器码 和汇编语言这两种方式。

| 层级     | 形式       | 人类可读 | CPU直接执行 | 与机器码关系         |
| -------- | ---------- | -------- | ----------- | -------------------- |
| 语义指令 | 抽象操作   | ✅        | ❌           | 一一对应（ISA 层面） |
| 汇编语言 | 助记符文本 | ✅        | ❌           | 多对一（含伪指令）   |
| 机器码   | 二进制串   | ❌        | ✅           | 自身                 |

#### 机器码的独特性

在最终执行时，**只有机器码**进入 CPU；汇编语言只是中间表示；指令是抽象语义。



#### 一条指令对应一条机器码吗？

**指令和机器码是一一对应的**



#### 一条汇编语句在执行的时候一定对应一条机器码吗？

**汇编语言与机器码并非一一对应，即使去掉语法糖，伪指令也会导致多条汇编文本对应同一段机器码。**

一条汇编语句（assembly statement）在汇编阶段可能被翻译成：

1. 0 条机器码
   ‑ 纯汇编指令，如宏、标签、段声明、条件编译指令等，不产生任何机器码。

2. 1 条机器码
   ‑ 对应一条 CPU 指令（如 RISC-V 的 `add x5,x6,x7`）。

3. 多条机器码
   ‑ 伪指令或宏展开：

   ```
   li  x5, 0x12345678      # RISC-V 伪指令
   ```

   会被汇编器展开成，两条机器码。

   ```
   lui x5, 0x12345
   ori x5, x5, 0x678
   ```

因此，**汇编语句与机器码之间没有“必定 1:1”的关系**。



高级语言和编译器

### 1. 定义（一句话版）

表格

复制

| 概念                 | 一句话定义                                                   | 本质                                 |
| :------------------- | :----------------------------------------------------------- | :----------------------------------- |
| 高级语言 (HL)        | 面向人类抽象思维的**程序描述语言**（C/C++、Python、Java…）   | 纯文本，与机器无关                   |
| 编译器 (Compiler)    | 把**整个**高级语言源程序**一次性**翻译成**另一种语言**（通常是机器码或字节码）的程序 | 离线、批式、生成可独立执行的目标文件 |
| 解释器 (Interpreter) | **逐条**读入源程序/字节码并**立即执行**的程序                | 在线、流式、不生成独立可执行文件     |

------

### 2. 关系图（带数据流向）



复制

```
          ┌---------------┐
          │   高级语言    │
          │   源文件 (*.c │
          │   *.py *.java)│
          └------┬--------┘
                 │
    ┌------------┴------------┐
    ▼                         ▼
┌------------┐           ┌---------------┐
│  编译器    │           │   解释器      │
│ (gcc, javac│           │ (python.exe   │
│  rustc…)   │           │  node, JVM*)  │
└----┬-------┘           └------┬--------┘
     │ 机器码/字节码            │ 逐条执行
     ▼                         ▼
┌------------┐           ┌---------------┐
│ 可执行文件 │           │ 运行时结果    │
│  (*.exe *.so│           │  (stdout…)    │
└------------┘           └---------------┘
```

*注：JVM 既是字节码解释器，也带 JIT 编译器，属于混合实现。*

------

### 3. 经典问题 FAQ

表格

复制

| 问题                                       | 标准答案 / 常见误区澄清                                      |
| :----------------------------------------- | :----------------------------------------------------------- |
| **“Python 是解释型语言吗？”**              | 官方实现 CPython 是解释器；但 Python **语言规范本身**并未限定实现方式，也有 PyPy（JIT 编译器）、Nuitka（全编译器）。 |
| **“Java 到底是编译还是解释？”**            | **两段式**：源码 → `javac` 编译为字节码 → JVM **先解释**后**JIT 编译**，热点代码最终变成机器码。 |
| **“C# 和 Java 有何区别？”**                | C# 走 **MSIL + JIT** 的同样两段式，但 CLR 允许 **AOT 全编译**（ReadyToRun / NativeAOT），策略更灵活。 |
| **“解释器一定比编译器慢吗？”**             | 传统解释器确实慢；但现代解释器 + JIT 往往可把热点路径编译成机器码，**实际性能接近甚至超过静态编译**。 |
| **“一次编写到处运行是谁的功劳？”**         | **字节码 + 虚拟机**。语言规范 + 标准库 + 虚拟机屏蔽了 OS/硬件差异，不是单靠“解释”或“编译”。 |
| **“解释器能调试，编译器不能调试？”**       | 错。调试信息（DWARF、PDB、SourceMap）由编译器生成；解释器只是能**边解释边暴露源码行号**而已。 |
| **“编译器生成的机器码一定更快吗？”**       | **不一定**。静态编译无法获得运行时剖面信息；JIT 编译器可根据实际分支、热点做**投机优化**，有时会反超。 |
| **“为什么有些语言既有编译器又有解释器？”** | **开发效率 vs 运行效率**的折中：解释器启动快、跨平台；编译器生成高效本地码。现代语言通常提供 **混合工具链**（Rust：debug 用 `cargo run`，release 用 `cargo build --release`）。 |

------

### 4. 小结（背下来即可）

1. 高级语言 = 人类抽象
2. 编译器 = 离线翻译器，产出独立目标码
3. 解释器 = 在线执行器，不产独立目标码
4. 现代实现往往是“**两段式甚至三段式**”（源码→中间码→JIT/AOT→机器码）
5. **语言本身**并不先天绑定“编译”或“解释”，取决于**官方/第三方实现**。





####  gpu

SASS 和 cubin 的关系可以简单理解为：

- **SASS**（**Streaming ASSembly**）是 **GPU 上真正运行的机器码指令集**，也就是 GPU 的汇编语言。

- **cubin**（**CUDA Binary**）是一个 **包含 SASS 指令的二进制文件**，它是最终部署到 GPU 上运行的可执行文件格式

  。

具体来说：

- 在 CUDA 编译流程中，源代码（如 `.cu` 文件）首先被编译成 PTX（中间表示），然后再由 `ptxas` 工具将 PTX 编译成特定架构的 **SASS** 指令，最终生成 **cubin** 文件。

- 你可以用 `cuobjdump --dump-sass` 命令将 cubin 文件反汇编，查看其中包含的 SASS 指令

  。

- 反过来，也有一些工具（如 CuAssembler）可以将修改后的 SASS 汇编代码重新打包成 cubin 文件，实现底层优化。

总结一句话：**cubin 是一个容器，里面装着 SASS 指令**；SASS 是 cubin 文件里真正用于在 GPU 上执行的机器码。





可以理解是针对gpu的一段程序

可以这么理解，但再精确一点会更清楚：

- **SASS** 是针对 GPU 的 **机器码指令序列**，就是 GPU 真正执行的“程序”本身。
- **cubin** 是这段程序被打包后的 **二进制文件**，就像一个 `.exe` 文件，里面包含了 SASS 代码以及其他元数据（如符号表、调试信息、架构信息等）。

所以，**SASS ≈ GPU 程序本身**，**cubin ≈ 打包后的 GPU 可执行文件**。