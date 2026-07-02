# Changelog

All notable changes to AI Texture to PBR will be documented in this file.

## [1.1.2] - 2026-06-18

### Added
- 本地 ComfyUI 面板新增「主模型」选择器：可扫描 `ComfyUI/models/diffusion_models`（及旧版 `unet`）中已下载的生图主模型并替换工作流中的主模型。
- 偏好设置 > 模型管理新增「刷新本地模型列表」按钮，扫描本地 ComfyUI 模型目录并缓存到偏好设置。
- `ComfyUIClient._execute_workflow` 支持 `require_images=False`，为后续纯文本工作流预留。

### Fixed
- ComfyUI 检测同时支持便携版（`python_embeded/python.exe`）与桌面版/git 版（系统 Python）。
- 仅填写 ComfyUI URL 且 URL 可达时，面板不再提示「需要下载安装」。
- 无缝贴图 `ADVANCED` 模式改为 in-place 缝合，输出尺寸不变且边缘连续；`BASIC` 模式强制边界相等。
- 「测试连接」按钮改为真正验证 ComfyUI 可用性：未运行时会自动启动本地实例并等待连接成功，失败才报错。
- 桌面版 ComfyUI 支持自动启动：优先查找 venv/`.venv` 中的 Python，找不到时回退系统 Python。
- 修复 Blender Image 与保存的 PNG 上下镜像问题：写入 Blender pixels 前对 PIL 图像做垂直翻转，预览/3D 材质与磁盘文件方向一致。
- 放宽 ComfyUI 连接/启动探测的超时与错误处理，减少启动初期因响应慢导致的 ReadTimeout 误报。
- 修复输出参数区显示逻辑：种子只在 LOCAL_COMFYUI 后端下显示，API 模式下隐藏；「本地离线 PBR 烘焙」开关只要检测到本地 ComfyUI 就绪即显示，API 模式下也可选择跳过 CHORD。
- 「AI 优化提示词」无 API 时回退到本地基于材质配置的规则扩展，不再直接报错。
- 「反推提示词」无 API 时提示用户配置 API Provider 或本地 Ollama。

## [1.1.1] - 2026-06-18

### Changed
- 本地 PBR 法线生成默认参数调整为 `strength=1.5, detail=0.4`，更接近 Image Editor Master GPU 输出风格。
- 法线生成 `invert` 语义与 Image Editor Master 兼容：勾选后实际反转法线 R/X 方向。

### Added
- N 面板新增「本地法线参数」区块，可调整法线强度、细节、反转绿色通道（本地 PBR 模式下生效）。

## [1.0.0] - 2026-06-18

### Added
- Initial public release
- PBR material generation from text prompts (txt2img → CHORD)
- Reference image mode (img2img → PBR extraction)
- Industrial material configuration (19 categories, 70+ subcategories)
- Multi-backend support: Local ComfyUI, OpenAI-compatible API, Gemini
- One-click ComfyUI auto-installer with mirror download support
- Model manager with download progress
- Channel Packing (RGBA texture channel layout)
- Preview panel with enlarge, file explorer, results history
- Ctrl+V clipboard paste for reference images
- Alt+E in-place prompt editor
- AI prompt optimization via LLM
- Reference image captioning via vision models
- Configurable API providers with Fetch Models

### Dependencies
- Requires Blender 3.6.0+
- Optional: ComfyUI (auto-installed), 7-Zip or py7zr
- Python: Pillow, NumPy, websocket-client, requests (Blender bundled)
