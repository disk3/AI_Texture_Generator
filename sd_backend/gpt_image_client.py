"""OpenAI GPT-Image API 客户端。

支持官方 OpenAI 端点以及第三方 OpenAI 兼容端点（通过 base_url）。
文档: https://platform.openai.com/docs/guides/image-generation
"""

import base64
import io
import time
import requests
from typing import List
from PIL import Image

from .abstract_client import AbstractSDClient, GenerationConfig, GenerationResult, with_reference_mode_hint
from ..utils.logger import get_logger

log = get_logger(__name__)


class GPTImageClient(AbstractSDClient):
    """OpenAI GPT-Image API 客户端，支持自定义 base_url。"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-image-2",
        base_url: str = "",
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._progress_cb = None
        self.base_url = self._normalize_base_url(base_url)

    @staticmethod
    def _normalize_base_url(base_url: str = "") -> str:
        """Normalize common OpenAI-compatible base URLs to the /v1 API root."""
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def check_health(self) -> bool:
        return bool(self.api_key)

    def set_progress_callback(self, callback):
        self._progress_cb = callback

    def _poll_modelscope_task(self, task_id: str) -> list:
        """轮询 ModelScope 异步任务直到完成，返回 PIL Image 列表。"""
        task_url = f"{self.base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-ModelScope-Task-Type": "image_generation",
        }
        max_polls = 60  # 最多轮询 60 次
        poll_interval = 2  # 每次间隔 2 秒

        for _ in range(max_polls):
            if self._progress_cb:
                self._progress_cb(0.5, f"ModelScope task {task_id[:8]}... polling")

            resp = requests.get(task_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("task_status", "")
            if status == "SUCCEED":
                output_images = data.get("output_images", [])
                if not output_images:
                    raise RuntimeError(f"ModelScope 任务成功但未返回图像。响应: {data}")
                images = []
                for img_url in output_images:
                    img_resp = requests.get(img_url, timeout=30)
                    img_resp.raise_for_status()
                    images.append(Image.open(io.BytesIO(img_resp.content)))
                return images
            elif status in ("FAILED", "CANCELLED", "EXPIRED"):
                raise RuntimeError(f"ModelScope 任务 {status}: {data.get('message', data)}")

            time.sleep(poll_interval)

        raise RuntimeError(f"ModelScope 任务轮询超时（{max_polls * poll_interval} 秒）")

    def txt2img(self, config: GenerationConfig) -> GenerationResult:
        return self._generate(config)

    def img2img(self, config: GenerationConfig) -> GenerationResult:
        return self._generate(config)

    def get_models(self) -> List[str]:
        return [self.model]

    def _generate(self, config: GenerationConfig) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("API Key 未配置，请在偏好设置中填写 API Key。")

        url = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = with_reference_mode_hint(config.prompt, config)

        # GPT-Image-2 目前支持特定尺寸
        size_map = {
            (1024, 1024): "1024x1024",
            (1024, 1536): "1024x1536",
            (1536, 1024): "1536x1024",
        }
        size = size_map.get((config.width, config.height), "1024x1024")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "quality": "high",
            "n": config.batch_size,
        }

        # img2img: 使用参考图
        files = {}
        reference_request_mode = ""
        img_bytes = b""
        data_url = ""
        b64 = ""
        is_modelscope = "modelscope" in self.base_url.lower()
        is_third_party = "api.openai.com" not in self.base_url.lower()
        is_edit_model = "edit" in self.model.lower()
        if config.init_image is not None:
            buffer = io.BytesIO()
            # RGB 即可，避免 alpha 通道对部分 API 造成干扰
            config.init_image.convert("RGB").save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            b64 = base64.b64encode(img_bytes).decode()
            data_url = f"data:image/png;base64,{b64}"

            if is_modelscope:
                # ModelScope Qwen-Image-Edit 系列需要 images (base64 数组)；
                # Qwen-Image 文生图系列使用 image_url (data URL 数组)。
                if is_edit_model:
                    # ModelScope 文档示例为 [[base64], [base64]]，优先使用二维数组
                    payload["images"] = [[b64]]
                    reference_request_mode = "modelscope_edit_images"
                else:
                    payload["image_url"] = [data_url]
                    reference_request_mode = "modelscope_generation_image_url"
            else:
                url = f"{self.base_url}/images/edits"
                files["image"] = ("ref.png", img_bytes, "image/png")
                # 保留 size 参数，强制控制输出尺寸（避免第三方代理默认返回 2K）
                payload.pop("n", None)
                reference_request_mode = "edit_file"

        if self._progress_cb:
            self._progress_cb(0.3, "正在发送请求到图像 API...")

        # ModelScope 需要异步模式 header
        post_headers = dict(headers)
        if "modelscope" in self.base_url.lower():
            post_headers["X-ModelScope-Async-Mode"] = "true"

        resp = requests.post(
            url,
            headers=post_headers,
            json=payload if not files else None,
            data=payload if files else None,
            files=files if files else None,
            timeout=self.timeout,
        )

        # 第三方代理兼容：400 时尝试去掉 quality/n 重试
        if resp.status_code == 400 and is_third_party and not files:
            import json as _json
            log.debug("GPT proxy 400. Request body: %s", _json.dumps(payload))
            log.debug("GPT proxy 400. Response: %s", resp.text[:500])
            if self._progress_cb:
                self._progress_cb(0.32, "正在以兼容模式重试（第三方代理）...")
            fallback = {
                "model": self.model,
                "prompt": prompt,
                "size": size,
            }
            if "image_url" in payload:
                fallback["image_url"] = payload["image_url"]
            resp = requests.post(
                url,
                headers=post_headers,
                json=fallback,
                timeout=self.timeout,
            )
            if resp.status_code == 400:
                log.debug("GPT proxy 400 (retry). Request body: %s", _json.dumps(fallback))
                log.debug("GPT proxy 400 (retry). Response: %s", resp.text[:500])

        # ModelScope 参考图字段兼容：Edit 模型用 images，文生图用 image_url；
        # 失败时互换尝试一次。
        if (
            resp.status_code in (400, 404, 405, 422)
            and is_modelscope
            and config.init_image is not None
        ):
            import json as _json
            log.debug("ModelScope reference request failed (%d). Response: %s", resp.status_code, resp.text[:500])
            if reference_request_mode == "modelscope_edit_images":
                if self._progress_cb:
                    self._progress_cb(0.34, "正在通过 image_url 重试（ModelScope）...")
                # 二维数组失败时，先尝试一维 base64 数组，再尝试 image_url data URL
                for alt_key, alt_value in [("images", [b64]), ("image_url", [data_url])]:
                    alt_payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "size": size,
                        alt_key: alt_value,
                    }
                    resp = requests.post(
                        url,
                        headers=post_headers,
                        json=alt_payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code < 400:
                        break
                    log.debug("%s retry body: %s", alt_key, _json.dumps(alt_payload)[:500])
                    log.debug("%s retry response: %s", alt_key, resp.text[:500])
            elif reference_request_mode == "modelscope_generation_image_url":
                if self._progress_cb:
                    self._progress_cb(0.34, "正在通过 images 重试（ModelScope）...")
                alt_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "size": size,
                    "images": [b64],
                }
                resp = requests.post(
                    url,
                    headers=post_headers,
                    json=alt_payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    log.debug("images retry body: %s", _json.dumps(alt_payload)[:500])
                    log.debug("images retry response: %s", resp.text[:500])

        # 第三方端点对参考图 API 的兼容性差异很大：
        # edits multipart 失败时尝试 generations + image_url，反过来也尝试一次 edits。
        if (
            resp.status_code in (400, 404, 405, 422)
            and is_third_party
            and config.init_image is not None
            and not is_modelscope
        ):
            import json as _json
            log.debug("Reference request failed (%d). Response: %s", resp.status_code, resp.text[:500])
            if reference_request_mode == "edit_file":
                if self._progress_cb:
                    self._progress_cb(0.34, "正在通过 image_url 重试参考图...")
                alt_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "size": size,
                    "image_url": [data_url],
                }
                resp = requests.post(
                    f"{self.base_url}/images/generations",
                    headers=post_headers,
                    json=alt_payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    log.debug("image_url retry body: %s", _json.dumps(alt_payload)[:500])
                    log.debug("image_url retry response: %s", resp.text[:500])
            elif reference_request_mode == "generation_image_url":
                if self._progress_cb:
                    self._progress_cb(0.34, "正在通过 edits 端点重试参考图...")
                alt_payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "size": size,
                    "quality": "high",
                }
                resp = requests.post(
                    f"{self.base_url}/images/edits",
                    headers=post_headers,
                    data=alt_payload,
                    files={"image": ("ref.png", img_bytes, "image/png")},
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    log.debug("edits retry response: %s", resp.text[:500])

        resp.raise_for_status()
        data = resp.json()

        # ModelScope 异步任务模式：返回 task_id，需要轮询 /tasks/{task_id}
        if "task_id" in data:
            images = self._poll_modelscope_task(data["task_id"])
            if images:
                return GenerationResult(
                    images=images,
                    seed=-1,
                    info={},
                    metadata={"backend": "modelscope", "model": self.model, "timestamp": time.time()},
                )

        if self._progress_cb:
            self._progress_cb(0.8, "正在下载结果...")

        images = []

        # 1. OpenAI 官方格式：data[].url 或 data[].b64_json
        for item in data.get("data", []):
            img_url = item.get("url")
            if img_url:
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                images.append(Image.open(io.BytesIO(img_resp.content)))
            b64 = item.get("b64_json")
            if b64:
                images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

        # 2. 某些代理的 images[] 格式
        if not images:
            for item in data.get("images", []):
                img_url = item.get("url") or item.get("image_url")
                if img_url:
                    img_resp = requests.get(img_url, timeout=30)
                    img_resp.raise_for_status()
                    images.append(Image.open(io.BytesIO(img_resp.content)))
                b64 = item.get("b64_json") or item.get("base64")
                if b64:
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64))))

        # 3. 某些代理直接返回 image_url 字符串
        if not images:
            img_url = data.get("image_url")
            if img_url:
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                images.append(Image.open(io.BytesIO(img_resp.content)))

        if not images:
            raw = str(data)[:500]
            raise RuntimeError(f"OpenAI-compatible image API returned no images. Raw response: {raw}")

        if self._progress_cb:
            self._progress_cb(1.0, "完成")

        return GenerationResult(
            images=images,
            seed=-1,
            info={},
            metadata={"backend": "openai-compatible", "model": self.model, "timestamp": time.time()},
        )
