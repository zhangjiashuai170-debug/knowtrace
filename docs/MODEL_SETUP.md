# 可选模型配置

默认离线。适配器支持两类协议：Ollama `/api/chat` 与 Chat Completions 兼容 `/chat/completions`。2026-11-07已用DeepSeek官方服务完成一次合成资料真实联调；开发测试仍采用模拟HTTP响应，其他服务和广泛中文质量尚未验证。

接口字段核对来源：[OpenAI官方Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[Ollama官方Chat](https://docs.ollama.com/api/chat)。兼容服务的支持字段会有差异；通用分支只发送基础非流式文本请求，不声称支持全部供应商和推理模型。

## 本机连接 DeepSeek

本轮选择官方服务 `https://api.deepseek.com` 与 `deepseek-flash`，以[DeepSeek官方接入文档](https://api-docs.deepseek.com/quick_start/agent_integrations/)列出的名称为准（2026-11-07核对）。官方域名的根路径或 `/v1` 使用JSON输出、关闭thinking、1024输出token上限；其他兼容地址保持通用请求。参考[JSON输出](https://api-docs.deepseek.com/guides/json_mode/)与[thinking控制](https://api-docs.deepseek.com/guides/thinking_mode/)。

在项目目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/connect_deepseek.ps1
```

在本机窗口输入API Key并回车，字符隐藏。输入窗口随后退出，后台连接进程用临时合成文档发起一次受控模型问答，并检查无依据拒答与跨库403。这一流程不使用现有上传文档，不自动重试。检查通过后才受控结束本项目8766端口的旧演示并启动模型版；预检失败保留旧演示，端口属于其他程序时中止切换。

密钥仅由该后台进程使用，不写配置文件、命令行、报告或源码；进程结束后需要重新输入。本机安全状态在 `artifacts/deepseek-connection/status.json`，输入窗口状态单独保存在 `prompt-status.json`，合成资料检查报告在同目录 `preflight.json`。`serving` 表示本机服务已就绪，预检成功属于历史记录，不能证明下一次请求一定成功。网页默认仍选择离线摘录；需要生成时切换到“模型回答”。

同一时间只允许一个密钥输入窗口和一个连接进程。重复运行会退出，不覆盖正在运行的服务状态，也不追加模型请求。状态文件存在本身不证明进程仍在运行；以本机服务和实际请求结果核对。

加 `-CheckOnly` 只检查安装条件与固定参数，不询问密钥、不启动服务或调用模型。没有实际预检结果时，不把模拟测试写成真实模型联调成功。

已有本项目后台连接运行时，用 `scripts/connect_deepseek.ps1 -Upgrade` 输入新进程使用的密钥。升级先做一次合成资料预检，失败保留原服务；成功后才核对精确归属并替换进程。新进程按本机语义配置加载可选向量模型。只记录固定错误分类和HTTP状态，不保存原始服务端错误或凭据；不自动重试。

## 使用已有本地Ollama模型

本项目不自动安装Ollama或下载模型。若你已经有可用模型，在同一个PowerShell中配置后启动：

也可以使用参数化启动器（它只设置本次子进程环境，不保存配置）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_model.ps1 -Provider ollama -Model '<已安装的模型>'
```

```powershell
$env:KNOWTRACE_MODEL_PROVIDER = 'ollama'
$env:KNOWTRACE_MODEL_BASE_URL = 'http://127.0.0.1:11434'
$env:KNOWTRACE_MODEL_NAME = '<你已安装的模型名称>'
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

## 使用兼容服务

服务根地址应包含 `/v1`（若供应商如此定义）；应用在后面拼接 `/chat/completions`。远程地址必须HTTPS。

本机隐藏输入密钥的启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_model.ps1 -Provider openai_compatible -BaseUrl 'https://<服务地址>/v1' -Model '<模型名称>'
```

密钥不经过命令行参数，不写文件，仅交给本次服务进程；启动器退出后不保留。可以加 `-CheckOnly` 先检查参数，这不会启动服务、询问密钥或调用模型。已有8766服务时先正常结束它，或指定不同的 `-Port`，启动器不会覆盖其它进程。

```powershell
$env:KNOWTRACE_MODEL_PROVIDER = 'openai_compatible'
$env:KNOWTRACE_MODEL_BASE_URL = 'https://<服务地址>/v1'
$env:KNOWTRACE_MODEL_NAME = '<你有权限使用的模型名称>'
# 在本机为这个进程提供KNOWTRACE_MODEL_API_KEY；不要把密钥贴到聊天或提交源码。
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

配置不写入应用数据库。`.env.example`只是变量名参考，应用不会自动读取.env文件。`KNOWTRACE_MODEL_TIMEOUT_SECONDS`默认45秒，允许1至120秒。

## 使用与核对

1. 重启应用后登录，状态应显示“已配置，尚未实际请求”。这不代表服务连通。
2. 默认仍是“离线摘录”。明确选择“模型回答”才会请求推理。
3. 检索证据门槛未通过时，本地直接拒答，不发送模型请求。
4. 合规响应后显示引用格式核对状态，并记录最近成功时间。历史成功不证明下一次可用。
5. 超时、错误响应、缺引用或未知引用均明确失败，不把原始输出包装成正常答案。

发送给模型的是用户问题与当前权限中选出的证据片段；使用远程服务时，这些内容会离开本机。只有显式模型模式会触发该路径。模型配置属于服务管理员，不能从浏览器任意指定URL或提交密钥。

恢复离线：设置 `$env:KNOWTRACE_MODEL_PROVIDER = 'disabled'` 后重启。关闭模式时配置加载器不读取其他模型变量。

## 受控真实联调工具

`python scripts/check_model_live.py` 默认只登录本机演示会话并读取配置状态，报告 `configuration_status_read` 只表示读取成功；真实推理状态为 `not_run`。

服务路线选定后，可在明确提供了模型环境配置的进程中加 `--run-live`：工具使用临时合成知识库检查有依据问答、无依据拒答和跨库403。它不从运行中服务复制密钥，也不读取密码保险箱。默认启动器只配置子进程，单独运行联调工具时需要为它显式提供相同配置。

报告仅做引用格式/ID检查，保留实际派发次数、延迟与未知状态，不自动重试，也不把未调用写成联调成功。尚未配置服务时不要用模拟测试结果代替真实联调结果。
