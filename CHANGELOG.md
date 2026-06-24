# Changelog

All notable changes to AI Texture to PBR will be documented in this file.

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
