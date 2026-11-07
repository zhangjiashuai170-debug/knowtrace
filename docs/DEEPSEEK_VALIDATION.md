# DeepSeek 接入准备与验收

2026-11-07。本记录覆盖模型适配、Windows本机密钥输入、一次真实合成资料联调和受控切换流程。用户在本机隐藏输入密钥后，真实检查于Asia/Tokyo 22:46完成。

## 已完成

- 官方服务为 `https://api.deepseek.com`，模型 `deepseek-flash`。官方根路径或`/v1`请求加入JSON输出、关闭thinking和1024输出token上限；其他兼容端点保持通用协议。
- 结构化回答、已授权引用ID和正文引用一致性校验；响应为空、损坏、被截断或出现未知引用时拒绝采用。引用格式通过不代表语义已经验证。
- 本机PowerShell隐藏输入密钥，仅给本次后台连接进程。连接进程加载后移除其密钥环境变量，配置对象在内存中供后续生成使用；辅助进程不继承该密钥。
- 连接进程在请求前取得单实例锁，输入窗口使用独立状态文件；重复启动不覆盖运行状态或追加预检请求。
- 只用临时合成文档进行一次有依据问答，同时核对无依据本地拒答和跨库403。真实调用未发生时不能标记联调通过。
- 预检通过后才受控停止本项目原演示；未知端口占用保留现场。本机服务实际完成启动后才记录`serving`。
- 前端区分配置状态、合成资料联调历史和当前应用请求历史；默认回答方式仍是离线摘录。

## 本轮实际检查

- 全套Python测试：211 passed，9 subtests passed。模型与连接流程测试使用模拟网络；系统进程停止分支使用伪造进程清单和模拟Stop-Process，没有停止真实进程。
- 一个Starlette TestClient弃用警告，不影响本轮通过；后续升级依赖时处理。
- `node --check web/app.js`通过。
- Windows PowerShell执行两个新启动/停止脚本的`-CheckOnly`通过；停止检查精确识别本项目原演示，没有执行停止。
- 两个新PowerShell脚本均为UTF-8 BOM。
- 连接代码经独立只读审查，未发现阻塞问题。

## 真实结果的依据

本轮实际结果：

- DeepSeek模型`deepseek-flash`，`protocol_checks_passed`，实际模型推理已观察到，派发1次、自动重试0次。
- 有依据问题实际返回`KnowTraceProbe stores project snapshots for 14 days [S1]。`，HTTP200、格式/授权文档ID检查通过。本轮由助手将此单句与合成原文“14 days”核对一致；未经独立人工复核，未声称完成通用语义评测。
- 无依据问题本地拒答，模型派发0次；跨库问题HTTP403，模型派发0次。
- 本机服务完成启动，`/api/model`返回`configured=true`、`live_verified=true`，成功范围为`isolated_synthetic_preflight`，时间`2026-11-07T13:46:20.951638+00:00`。
- 本轮未使用主数据目录中的上传文件进行外发测试。可公开的合成资料结果摘要见[deepseek-live-smoke.json](deepseek-live-smoke.json)。

本机运行记录在`artifacts/deepseek-connection/`，不包含在源码包。`prompt-status.json`只记录输入步骤；`status.json`记录连接生命周期；`preflight.json`记录合成资料的真实响应、实际派发次数及引用格式检查。

只有预检报告为`protocol_checks_passed`、`model_inference_observed=true`且派发一次，并且本机服务实际就绪，才能报告真实联调完成。历史成功不证明持续在线，也不构成中文生成质量或语义引用准确率评测。

官方依据：[接入方式](https://api-docs.deepseek.com/quick_start/agent_integrations/)、[JSON输出](https://api-docs.deepseek.com/guides/json_mode/)、[thinking模式](https://api-docs.deepseek.com/guides/thinking_mode/)。
