# 真实模型联调前的本机检查

2026-11-07检查：16GB内存、i5-11400H、RTX3050 Laptop 4GB显存（检查时约3GB空闲）；C盘约6GB、D盘约18GB可用。未检测到Ollama命令或本机11434服务。此记录不是推理性能测试。

## 本机路线建议

先用D盘便携Ollama和纯文本 [Qwen3:1.7b](https://ollama.com/library/qwen3:1.7b) 做实际连通与引用格式检查：官方模型页列出约1.4GB、Q4_K_M、Apache-2.0。之后如确有质量需要，再比较 [Qwen3:4b](https://ollama.com/library/qwen3:4b)（约2.5GB）。硬件适配是判断，尚不保证全GPU运行、速度或中文回答质量。

研究时核对的 [Ollama v0.40.15](https://github.com/ollama/ollama/releases/tag/v0.40.15) Windows便携包约1.61GB。模型加运行包合计约3GB下载，解压与运行还需要额外空间；计划把运行程序和 `OLLAMA_MODELS` 放D盘，不占用紧张的C盘。尚未下载或安装。

官方要求及GPU支持：[Windows说明](https://docs.ollama.com/windows)、[GPU说明](https://docs.ollama.com/gpu)。本机驱动已检测为591.74；具体可运行程度仍须实际短请求验证。

## 请求参数核对

[官方Thinking说明](https://docs.ollama.com/capabilities/thinking)支持顶层 `think:false` 关闭Qwen3长推理；[结构化输出](https://docs.ollama.com/capabilities/structured-outputs)支持JSON Schema。实际接入时再针对该服务设置短上下文、输出上限与关闭长推理，保留本地引用核验，不把JSON服从性当语义正确。

## 等待的选择

本机模型与已有API服务是两种不同部署路径。服务尚未选择，本轮仅准备启动器与受控联调工具；真实推理未执行，当前KnowTrace仍默认离线。
