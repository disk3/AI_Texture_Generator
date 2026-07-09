# AI Texture to PBR

**一键从提示词或参考图生成高质量 PBR 材质贴图，并自动创建 Blender 材质。**

## 功能亮点

- 🖼️ **图生材质** — 上传参考图，自动生成无缝可平铺 PBR 贴图（Base Color / Normal / Roughness / Metallic / Height）
- ✍️ **文生材质** — 输入中文描述，AI 生成对应的材质纹理
- 🎛️ **工业材质配置** — 内置 19 大类 / 70+ 子类材质预设，自动构造专业提示词
- 🔄 **多后端支持** — 本地 ComfyUI (ZImage + CHORD) / OpenAI-compatible API / Gemini
- 📦 **Channel Packing** — 支持将 Roughness / Metallic / AO / Height 打包到单张 RGBA 贴图
- 🔌 **一键安装 ComfyUI** — 内置下载器和模型管理器，无需手动配置

## 系统要求

| 项目 | 最低要求 |
|------|----------|
| Blender | 3.6.0 或更高 |
| 操作系统 | Windows 10+（macOS / Linux 部分功能支持） |
| Python | Blender 内嵌 Python（3.11+） |
| 磁盘空间 | ~25 GB（使用本地 ComfyUI 模式时） |
| GPU | NVIDIA GPU (4GB+ VRAM) 推荐用于本地模式 |

## 快速开始

### 方式一：使用 API 后端（最简单）

1. 安装插件：Blender → Edit → Preferences → Add-ons → Install → 选择 ZIP
2. 启用 **AI Texture to PBR**
3. 如果 N-Panel 提示基础组件缺失，点击 **一键修复基础组件**；如果偏好设置里提示"未安装 keyring"，点击 **一键修复 API 组件**
4. 修复完成后，重新启用插件或重启 Blender（一键修复成功后会尝试立即加载，如 UI 未刷新则重启）
5. 在 Preferences 中配置 API Provider（OpenAI / ModelScope / Gemini / 自定义兼容端点）
6. 点击面板上的 **Generate PBR Maps**

### 方式二：使用本地 ComfyUI

1. 安装插件同上
2. 在 3D View 右侧 N-Panel → **AI Texture** 面板，Provider 选择 **Local ComfyUI**
3. 点击 **获取 ComfyUI** 打开官方下载页面，下载并解压到本地；或点击 **测试连接** 自动检测/启动已配置路径的 ComfyUI
4. 若缺少 `websocket-client` 等 ComfyUI 增强组件，点击 **一键修复 ComfyUI 组件**
5. 在 Preferences → Model Manager 中下载所需模型
6. 返回面板，选择材质配置，点击 **Generate PBR Maps**

### 使用参考图

1. 在 **AI Texture** 面板的"使用参考图"区：
   - 点击"选择文件"上传，或直接 **Ctrl+V** 粘贴剪贴板图片
2. （可选）点击"反推提示词"自动分析材质特征
3. 调整"重绘幅度"和"纹理保持"参数
4. 点击 **Generate PBR Maps**

## 材质配置说明

面板中的 Material Config 支持以下维度：

- **分类**: 混凝土、金属、木材、石材、陶瓷、织物等 19 类
- **子分类**: 每类下的具体变体（如裸露骨料、波纹板、直纹木等）
- **颜色**: 20+ 预定义颜色
- **表面处理**: 光滑、粗糙、抛光、拉丝、哑光等
- **特征**: 防滑、防水、磨砂、压花等
- **视角**: 俯视 / 侧视 / 正视
- **用途**: 地面、墙面、屋顶、装饰等

修改材质配置会自动更新 Prompt，也可手动编辑或使用 **AI 优化提示词**。

## API Provider 配置

支持以下协议：

### OpenAI / 兼容端点
- Base URL: `https://api.openai.com/v1`（或第三方代理地址）
- 支持 model: `gpt-image-2`, `dall-e-3`, 及其他 `/v1/images/generations` 兼容模型

### ModelScope
- Base URL: `https://api-inference.modelscope.cn/v1`
- 内置模型: Z-Image Turbo, Qwen-Image, FLUX.2-klein 等

### Gemini
- Base URL: `https://generativelanguage.googleapis.com/v1beta`
- 支持: `gemini-2.5-flash-image` 等
- ⚠️ **安全提醒**：Gemini API 官方要求通过 URL 查询参数 `?key=` 传递 API Key（非 HTTP Header）。这意味着 Key 可能被中间代理、CDN 或服务端日志记录。建议使用专属的低权限 Key，勿与高价值服务共用同一 Key。

点击每个 Provider 的 **Fetch Models** 按钮自动拉取可用模型列表。

### API Key 安全存储

- 推荐安装 `keyring`（通过偏好设置里的 **一键修复 API 组件**）。安装后，API Key 会写入系统密钥环，Blender 偏好设置中仅显示占位符 `********`，不会随 `userpref.blend` 明文保存。
- 未安装 `keyring` 时，API Key 仍以明文存储在 Blender 用户配置中，插件会显示红色警告提醒。

## 组件与一键修复

插件会尽量使用 Blender 自带能力和 Python 标准库运行。增强组件按需安装：

| 组件 | 用途 | 修复入口 |
|------|------|----------|
| `numpy` | 本地 PBR 算法必需 | **一键修复基础组件** |
| `Pillow` | 剪贴板、图片解码增强 | **一键修复 API 组件** / **一键修复 ComfyUI 组件** |
| `requests` | HTTP 客户端增强 | **一键修复 API 组件** / **一键修复 ComfyUI 组件** |
| `keyring` | API Key 安全存储 | **一键修复 API 组件** |
| `websocket-client` | ComfyUI 实时进度 | **一键修复 ComfyUI 组件** |
| `py7zr` | ComfyUI 便携版 7z 解压 | **一键修复安装器组件** |

缺失时插件不会自动安装，只在对应面板显示 **一键修复...** 按钮，由用户手动触发，避免注册阶段阻塞或 headless 环境报错。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+V` | 从剪贴板粘贴参考图 |
| `Ctrl+ENTER` | 在文本编辑器中编辑 Prompt |

## 常见问题

### ComfyUI 安装失败？
- 确保网络可访问 GitHub 和 hf-mirror.com
- 可在 Preferences 中关闭"使用国内镜像下载"
- 也可手动安装 ComfyUI，然后在 Preferences 中设置路径

### 生成失败怎么办？
- 检查 ComfyUI 是否运行（访问 http://127.0.0.1:8188）
- 查看所需模型是否已下载（Preferences → Model Manager）
- 确认 UV 已正确展开（选中网格 → UV → Smart UV Project）

### 图片效果不理想？
- 尝试调整重绘幅度（参考图模式）
- 优化 Prompt（使用 AI 优化按钮，或手动编写更详细的描述）
- 切换后端（不同后端效果有差异）

### API Key 输入后星号变短？
这是正常现象。安装 `keyring` 后，插件会把真实密钥写入系统密钥环，Blender 输入框里只保留 8 位占位符 `********`，不会暴露真实密钥长度，也不影响实际生成。

## 许可证

本插件采用**开源非商业用途**许可。个人、学术、教育及非营利组织可自由使用和修改。商业用途需单独授权。详见 [LICENSE](LICENSE)。

第三方模型和软件各有其许可证，请使用者自行遵守。

## 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [CHORD](https://github.com/ubisoft/ComfyUI-Chord) — Ubisoft La Forge
- [Z-Image Turbo](https://modelscope.cn/models/Tongyi-MAI/ZImage-Turbo)
- [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
