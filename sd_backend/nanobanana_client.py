"""Google Gemini API 客户端（复用 NanoBanana 偏好槽位）。

支持 Gemini 2.0 Flash 图像生成（generateContent API）。
文档: https://ai.google.dev/gemini-api/docs

旧版 NanoBanana /images/generations 接口已清理，现在只保留 Gemini 路径。
"""

import base64
import io
import time
from typing import List

from .abstract_client import AbstractSDClient, GenerationConfig, GenerationResult, with_reference_mode_hint
from ..utils.logger import get_logger

log = get_logger(__name__)


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from ..utils import simple_requests
        return simple_requests


class NanobananaClient(AbstractSDClient):
    """Google Gemini API 客户端。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 120,
        model: str = "gemini-2.0-flash-exp-image-generation",
    ):
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.timeout = timeout
        self.model = model
        self._progress_cb = None
        self._health_cache = None
        self._health_cache_time = 0.0

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        if "googleapis.com" in base and not base.endswith("/v1beta"):
            return f"{base}/v1beta"
        return base

    def check_health(self) -> bool:
        requests = _requests_module()

        # 30 秒缓存，避免每次生成前都发请求耗尽配额
        now = time.time()
        if self._health_cache is not None and (now - self._health_cache_time) < 30:
            return self._health_cache

        if not self.api_key:
            self._health_cache = False
            self._health_cache_time = now
            return False

        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            resp = requests.get(url, timeout=10)
            result = resp.status_code == 200
            self._health_cache = result
            self._health_cache_time = now
            return result
        except Exception:
            log.debug("Gemini health check failed")
            self._health_cache = False
            self._health_cache_time = now
            return False

    def set_progress_callback(self, callback):
        self._progress_cb = callback

    def txt2img(self, config: GenerationConfig) -> GenerationResult:
        return self._generate(config)

    def img2img(self, config: GenerationConfig) -> GenerationResult:
        return self._generate(config)

    def get_models(self) -> List[str]:
        return [self.model]

    def _generate(self, config: GenerationConfig) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError(
                "Gemini API Key 未配置，请在偏好设置中填写 API Key。"
            )

        return self._generate_gemini(config)

    def _generate_gemini(self, config: GenerationConfig) -> GenerationResult:
        """调用 Gemini generateContent API 生成图像（含 429 重试）。

        局部导入 Pillow/requests，避免插件启用时依赖。

        SECURITY: Gemini API Key 通过 URL query param (?key=) 传递，
        可能被中间代理、CDN 或服务端日志记录。建议使用专属低权限 Key。
        """
        requests = _requests_module()

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        prompt = with_reference_mode_hint(config.prompt, config)

        parts = []
        if config.init_image is not None:
            from ..utils.image_utils import encode_numpy_to_png_bytes
            import numpy as np
            try:
                from PIL import Image
                pil_available = True
            except ImportError:
                pil_available = False

            init_img = config.init_image
            if pil_available and isinstance(init_img, Image.Image):
                buf = io.BytesIO()
                init_img.convert("RGB").save(buf, format="PNG")
                img_bytes = buf.getvalue()
            elif isinstance(init_img, np.ndarray):
                if init_img.ndim != 3 or init_img.shape[-1] not in (3, 4):
                    raise TypeError(f"Gemini 参考图 numpy 数组形状不支持: {init_img.shape}")
                img_bytes = encode_numpy_to_png_bytes(init_img[..., :3])
            else:
                raise TypeError("Gemini 参考图必须是 PIL Image 或 numpy 数组")

            img_b64 = base64.b64encode(img_bytes).decode()
            parts.append({"inlineData": {"mimeType": "image/png", "data": img_b64}})

        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            },
        }

        headers = {"Content-Type": "application/json"}

        if self._progress_cb:
            self._progress_cb(0.3, "正在发送请求到 Gemini...")

        # 429 自动重试（指数退避）
        max_retries = 3
        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError:
                if resp is not None and resp.status_code == 429 and attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    if self._progress_cb:
                        self._progress_cb(0.35, f"速率限制 (429)，{wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                raise
        data = resp.json()

        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            # 免费 tier limit:0 的友好提示
            if "limit: 0" in msg or "quota" in msg.lower():
                msg += (
                    "\n\n[提示] 此模型在 Google AI Studio 免费层级下配额为 0。"
                    "请在 Google AI Studio 中绑定结算信息以激活免费额度，"
                    "或切换至 GPT Image-2 / 本地 ComfyUI 后端。"
                )
            raise RuntimeError(f"Gemini API 错误: {msg}")

        if self._progress_cb:
            self._progress_cb(0.8, "正在下载结果...")

        from ..utils.image_utils import load_image_bytes_to_numpy

        images = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_b64 = part["inlineData"]["data"]
                    img_bytes = base64.b64decode(img_b64)
                    images.append(load_image_bytes_to_numpy(img_bytes))

        if not images:
            for candidate in data.get("candidates", []):
                if candidate.get("finishReason") == "SAFETY":
                    raise RuntimeError(
                        "Gemini 因安全过滤阻止了本次生成，请尝试修改提示词。"
                    )
            raise RuntimeError("Gemini 未返回任何图像。")

        if self._progress_cb:
            self._progress_cb(1.0, "完成")

        return GenerationResult(
            images=images,
            seed=-1,
            info={},
            metadata={
                "backend": "gemini",
                "model": self.model,
                "timestamp": time.time(),
            },
        )

