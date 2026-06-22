# Changelog

All notable changes to AI Texture to PBR will be documented in this file.

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
