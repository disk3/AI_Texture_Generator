import hashlib

from ..utils.logger import get_logger

log = get_logger(__name__)


class ConnectionPool:
    def __init__(self):
        self._clients = {}

    def get_client(self, backend_type: str, url: str, **kwargs):
        # 延迟导入具体客户端：缺少 requests/websocket 时仍可加载本地 PBR 功能
        if backend_type == 'COMFYUI':
            from .comfyui_client import ComfyUIClient
            client_cls = ComfyUIClient
        elif backend_type == 'GPT_IMAGE':
            from .gpt_image_client import GPTImageClient
            client_cls = GPTImageClient
        elif backend_type == 'NANOBANANA':
            from .nanobanana_client import NanobananaClient
            client_cls = NanobananaClient
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        key = f"{backend_type}@{url}"
        # API 客户端的 base_url 决定实际连接目标，必须纳入 key
        if backend_type in ('GPT_IMAGE', 'NANOBANANA'):
            base_url = kwargs.get("base_url", "")
            if base_url:
                key += f"@{base_url}"
            model = kwargs.get("model", "")
            if model:
                key += f"#{model}"
            api_key = kwargs.get("api_key", "")
            if api_key:
                key += f"@key:{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]}"

        if key not in self._clients:
            if backend_type == 'COMFYUI':
                self._clients[key] = client_cls(base_url=url, **kwargs)
            elif backend_type == 'GPT_IMAGE':
                self._clients[key] = client_cls(
                    api_key=kwargs.get("api_key", ""),
                    model=kwargs.get("model", "gpt-image-2"),
                    base_url=kwargs.get("base_url", ""),
                )
            elif backend_type == 'NANOBANANA':
                self._clients[key] = client_cls(
                    api_key=kwargs.get("api_key", ""),
                    base_url=kwargs.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                    model=kwargs.get("model", "gemini-2.5-flash-image"),
                )
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")
        return self._clients[key]

    def check_health(self, backend_type: str, url: str, **kwargs) -> bool:
        try:
            client = self.get_client(backend_type, url, **kwargs)
            return client.check_health()
        except Exception:
            log.debug("Health check failed for %s @ %s", backend_type, url)
            return False
