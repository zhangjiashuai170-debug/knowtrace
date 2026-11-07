# 本地语义检索

v0.3增加`semantic`和`semantic_hybrid`两个可选模式。原来的`hybrid`仍是BM25＋字符TF-IDF，并继续作为默认。新模式改变候选排序；原有覆盖率、数字、否定和未知标识符准入继续控制回答引用，所以有时能找到相关候选，仍会保守拒答。

## 模型与本机准备

模型为MIT许可的[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)，固定revision`614241f622f53c4eeff9890bdc4f31cfecc418b3`。官方ONNX INT8权重及tokenizer共135,429,554字节，约129.16MiB；来源、大小和SHA固定在模型准备程序及运行时中。原模型作者与许可证归属原仓库，模型文件不随本项目源码包分发。

Windows在首次安装完成后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_semantic.ps1 -ModelDirectory '<本地模型目录的绝对路径>'
```

这个明确的准备步骤才安装可选依赖并下载文件。普通启动、状态读取、问题检索和评测均不会自动下载。`-CheckOnly`不安装或下载。其他系统可安装`requirements-semantic-lock.txt`中的适用版本，再运行`python scripts/prepare_semantic_model.py --destination <绝对路径> --activate`。

非秘密的模型目录保存在项目`local-settings.json`，也可用`KNOWTRACE_EMBEDDING_DIR`显式指定。此本机路径设置不进入源码包；API不接受浏览器提供模型路径。源码包不含上传文件、向量数据库或模型权重。

## 推理与缓存

- CPUExecutionProvider，最多4个计算线程，每段单独推理；不使用显卡或远程向量服务，关闭ORT遥测。实际复核发现动态INT8批量编码会让同段向量受其他样本与padding影响，因此固定单段编码，并把此策略加入模型指纹。
- 所有语言的查询加`query: `、文档加`passage: `；最长512 token，按attention mask平均池化，再做L2归一化，输出384维。
- 从本地tokenizer读取`<pad>`编号，不使用模型配置中的假定值。文件大小及SHA通过后才加载运行时。
- 先按知识库权限查询分块，再加载或计算这些分块的向量，在本机做精确余弦排序。此版本没有ANN引擎或吞吐扩展实验。
- 向量缓存在同一SQLite中；模型指纹包含模型文件、前缀、池化、归一化和截断参数，缓存还校验文本SHA与维度。删除分块会级联删除向量；查询文本不写入向量缓存。
- `semantic_hybrid`用固定k=60的RRF融合BM25、字符TF-IDF和密集向量名次，没有按本轮结果调整权重。

缺模型时语义请求返回409；编码或缓存错误返回固定说明的503，词法模式仍可用。不会把静默回退后的结果标成语义结果。

## 评测与边界

固定20篇公开文档、100题来源核对草稿，开发/测试各50。测试分组早已被查看，属于回归比较。语义召回可能帮助同义或跨语言问题，但并非每个语料都优于词法方法。报告同时展示召回、实际来源采用与误拒答，不把余弦分数当作事实支持概率。

复现使用隔离目录，避免混入用户上传：

```powershell
.venv/Scripts/python.exe scripts/compare_semantic.py --data-dir artifacts/semantic-benchmark/data --output-dir artifacts/semantic-comparison
```

该工具只计算本地检索指标，不调用DeepSeek，也不使用大模型裁判。首次搜索可能包含文档向量构建；报告把search耗时与随后ask耗时分开，均不是并发吞吐。

DeepSeek进程中的密钥只驻留内存。升级运行中服务需要通过本机隐藏输入流程提供新进程用的密钥，不能从旧进程取回。先在隔离演示验证，再受控切换；新的合成资料预检失败时保留旧服务。

方法来源：[E5模型卡](https://huggingface.co/intfloat/multilingual-e5-small)、[Sentence Transformers ONNX说明](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx)、[ORT CPU量化说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。本机实际加载和编码已通过，精度与性能仍以本项目实际报告为准。
