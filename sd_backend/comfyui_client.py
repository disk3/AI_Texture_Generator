import json
import os
import time
import uuid
import random
from typing import Dict, Any
import io
import base64

from .abstract_client import AbstractSDClient, GenerationConfig, GenerationResult
from ..utils.logger import get_logger

log = get_logger(__name__)


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from ..utils import simple_requests
        return simple_requests


class ComfyUIClient(AbstractSDClient):
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 600,
                 zimage_workflow_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.timeout = timeout
        self._progress_cb = None
        self._zimage_workflow_path = zimage_workflow_path

    def check_health(self, auto_launch_path: str = "") -> bool:
        """检查 ComfyUI 是否在线，若不在线且提供了安装路径则自动启动。"""
        # 优先尝试复用 ComfyUI 便携版中的 Pillow/requests/websocket
        if auto_launch_path:
            try:
                from .comfyui_env_resolver import inject_comfyui_packages
                inject_comfyui_packages(auto_launch_path)
            except Exception as e:
                log.debug("Could not inject ComfyUI packages: %s", e)

        requests = _requests_module()
        try:
            # 启动初期 /system_stats 可能响应较慢，使用更宽容的 10 秒超时
            r = requests.get(f"{self.base_url}/system_stats", timeout=(3, 10))
            if r.status_code == 200:
                return True
        except requests.exceptions.ReadTimeout:
            log.debug("ComfyUI at %s connected but response read timed out", self.base_url)
        except Exception:
            log.debug("ComfyUI not responding at %s (may not be running)", self.base_url)

        # 未响应且提供了安装路径 → 尝试自动启动
        if auto_launch_path and os.path.isdir(auto_launch_path):
            from .comfyui_launcher import launch_comfyui, wait_for_comfyui

            log.info("ComfyUI not running, launching from %s ...", auto_launch_path)
            if launch_comfyui(auto_launch_path):
                log.info("Waiting for ComfyUI to start ...")
                if wait_for_comfyui(self.base_url, timeout=120, comfyui_path=auto_launch_path):
                    log.info("ComfyUI is ready.")
                    return True
                else:
                    log.warning("ComfyUI failed to start within 120s.")
            else:
                log.warning("Failed to launch ComfyUI.")
        return False

    def set_progress_callback(self, callback):
        self._progress_cb = callback

    def interrupt(self) -> bool:
        """Best-effort cancellation for the currently running ComfyUI prompt."""
        requests = _requests_module()
        try:
            resp = requests.post(f"{self.base_url}/interrupt", timeout=5)
            return resp.status_code < 400
        except Exception:
            log.debug("ComfyUI interrupt failed")
            return False

    def execute_workflow_json(self, config: GenerationConfig, workflow_path: str = "") -> GenerationResult:
        """Load a static ComfyUI API workflow JSON, apply overrides, and execute.

        Used for the Zimage PBR workflow: ComfyUI generates a single seamless
        diffuse texture, and Blender algorithmically extracts normal/roughness/
        metallic maps from it.
        """
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_w = int(config.width)
        target_h = int(config.height)
        use_upscaler = target_w > 1024 or target_h > 1024

        if workflow_path and os.path.isfile(workflow_path):
            path = workflow_path
        elif self._zimage_workflow_path and os.path.isfile(self._zimage_workflow_path):
            path = self._zimage_workflow_path
        else:
            # 512/1024 走无放大流程；2048/4096 走 SeedVR2
            default_name = (
                "zimage_pbr_api.json" if use_upscaler else "zimage_pbr_lowres_api.json"
            )
            path = os.path.join(addon_dir, "workflows", default_name)

        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        overrides = self._build_zimage_overrides(config)
        actual_seed = overrides.get("7", {}).get("seed", -1)
        for node_id, params in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)

        # 应用用户从面板选择的本地模型覆盖
        for node_id, params in getattr(config, "model_overrides", {}).items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)
                log.debug("Applied model override node %s: %s", node_id, params)

        # Debug: log actual prompt injected into ComfyUI
        node4_text = workflow.get("4", {}).get("inputs", {}).get("text", "")
        log.debug("Node 4 prompt length: %d", len(node4_text))
        log.debug("Node 4 prompt preview: %s", node4_text[:300])

        # CHORD ResizeAndPadImage 尺寸（Node 11）：按用户输出尺寸自适应
        if "11" in workflow:
            workflow["11"]["inputs"]["target_width"] = target_w
            workflow["11"]["inputs"]["target_height"] = target_h

        # SeedVR2 分辨率匹配目标长边
        if use_upscaler and "75" in workflow:
            workflow["75"]["inputs"]["resolution"] = max(target_w, target_h)

        # img2img: 注入 LoadImage + VAEEncode，把参考图编码后接入 KSampler
        if config.init_image is not None:
            uploaded_name = self._upload_image(config.init_image, "ref")
            denoise = getattr(config, "denoising_strength", 0.65)
            # Use high node IDs to avoid collision
            workflow["100"] = {
                "inputs": {"image": uploaded_name},
                "class_type": "LoadImage",
            }
            workflow["101"] = {
                "inputs": {
                    "pixels": ["100", 0],
                    "vae": ["8", 0],
                },
                "class_type": "VAEEncode",
            }
            if "7" in workflow:
                workflow["7"]["inputs"]["latent_image"] = ["101", 0]
                workflow["7"]["inputs"]["denoise"] = denoise

            # CHORD 模式下：把 Flux-Fill/SeedVR2 分支里的参考图节点也换成用户上传的图
            if "58" in workflow and workflow["58"].get("class_type") == "ETN_LoadImageBase64":
                import numpy as np
                from ..utils.image_utils import encode_numpy_to_png_bytes
                init_img = config.init_image
                if hasattr(init_img, 'shape'):
                    arr = np.asarray(init_img)
                    # 保留 alpha 通道，确保 ETN_LoadImageBase64 能输出 mask
                    ref_bytes = encode_numpy_to_png_bytes(arr)
                elif hasattr(init_img, 'save'):
                    ref_buffer = io.BytesIO()
                    init_img.save(ref_buffer, format="PNG")
                    ref_bytes = ref_buffer.getvalue()
                else:
                    raise TypeError("CHORD 参考图必须是 PIL Image 或 numpy 数组")
                ref_b64 = base64.b64encode(ref_bytes).decode()
                workflow["58"]["inputs"]["image"] = ref_b64

        result = self._execute_workflow(workflow, output_mode="by_prefix")
        result.seed = actual_seed
        return result

    def execute_chord_workflow(
        self,
        diffuse_image: Any,
        width: int = 2048,
        height: int = 2048,
        model_name: str = "",
    ) -> GenerationResult:
        """上传 diffuse 贴图并执行 CHORD-only 工作流，返回高质量 PBR maps。

        Args:
            diffuse_image: API 生成的 diffuse 贴图 (PIL Image 或 numpy 数组)
            width: 输出贴图宽度（覆盖 workflow 中 Node 11 的 target_width）
            height: 输出贴图高度（覆盖 workflow 中 Node 11 的 target_height）
            model_name: 可选。覆盖 CHORD 主模型文件名

        Returns:
            GenerationResult with pbr_maps containing basecolor/normal/roughness/metalness/height
        """
        # 1. 上传 diffuse 到 ComfyUI
        uploaded_name = self._upload_image(diffuse_image, "diffuse")

        # 2. 加载 CHORD-only 工作流
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(addon_dir, "workflows", "chord_only_api.json")

        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # 3. 替换 LoadImage 节点为上传的图片
        workflow["100"]["inputs"]["image"] = uploaded_name

        # 4. 覆盖输出尺寸（避免 workflow 里硬编码 2048）
        if "11" in workflow:
            workflow["11"]["inputs"]["target_width"] = width
            workflow["11"]["inputs"]["target_height"] = height

        # 5. 应用用户选择的 CHORD 模型
        if model_name and "12" in workflow:
            workflow["12"]["inputs"]["ckpt_name"] = model_name
            log.debug("CHORD-only workflow model override: %s", model_name)

        # 6. 执行工作流
        return self._execute_workflow(workflow, output_mode="by_prefix")

    def _build_zimage_overrides(self, config: GenerationConfig) -> Dict[str, Dict]:
        """Map GenerationConfig to Zimage workflow node overrides."""
        seed = config.seed if config.seed >= 0 else random.randint(0, 2147483647)
        prompt = self._tileable_prompt(config.prompt)
        return {
            "4": {"text": prompt},
            # Z-Image Turbo 固定 1024² 生成（最佳精度/显存平衡点）
            # 最终输出尺寸由 _import_texture_results 做 resize
            "6": {"width": 1024, "height": 1024},
            "7": {"seed": seed},
        }

    @staticmethod
    def _tileable_prompt(prompt: str) -> str:
        """Add workflow-specific tileability constraints without overwriting user intent."""
        prompt = (prompt or "").strip()
        hint = (
            "四向无缝可平铺，左右边缘和上下边缘纹理连续，"
            "不要边框、不要中心主体、不要透视畸变，"
            "平光照明，无阴影，无高光，无反光，无镜面反射"
        )
        if not prompt:
            return hint
        if any(key in prompt for key in ("四向无缝", "四向平铺", "边缘连续", "tileable")):
            return prompt
        return f"{prompt}。{hint}"

    @staticmethod
    def _map_prefix_to_map_type(prefix: str) -> str:
        """Map SaveImage filename_prefix to standard PBR map type key."""
        text = os.path.basename(str(prefix or "")).strip().lower()
        if not text:
            return ""
        text = text.replace("-", "_").replace(" ", "_")
        mapping = {
            "basecolor": "basecolor",
            "base_color": "basecolor",
            "albedo": "basecolor",
            "diffuse": "diffuse",
            "color": "diffuse",
            "texture_image": "diffuse",
            "normal": "normal",
            "roughness": "roughness",
            "rough": "roughness",
            "height": "height",
            "displacement": "height",
            "metalness": "metalness",
            "metallic": "metalness",
            "metal": "metalness",
            "放大": "diffuse",
        }
        if text in mapping:
            return mapping[text]
        for key, map_type in mapping.items():
            if text.startswith(f"{key}_") or text.startswith(f"{key}.") or f"_{key}_" in text:
                return map_type
        return ""

    def _upload_image(self, image: Any, name: str = "input") -> str:
        requests = _requests_module()
        import numpy as np
        from ..utils.image_utils import encode_numpy_to_png_bytes

        if hasattr(image, 'shape'):
            arr = np.asarray(image)
            img_bytes = encode_numpy_to_png_bytes(arr)
        elif hasattr(image, 'save'):
            # PIL Image fallback
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
        else:
            raise TypeError("image must be PIL Image or numpy array")

        files = {"image": (f"{name}.png", img_bytes, "image/png")}
        data = {"type": "input", "subfolder": ""}
        resp = requests.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json().get("name", f"{name}.png")

    def _execute_workflow(self, workflow: dict, output_mode: str = "flat", require_images: bool = True) -> GenerationResult:
        requests = _requests_module()
        client_id = str(uuid.uuid4())
        ws = None

        try:
            try:
                import websocket
                ws = websocket.create_connection(f"{self.ws_url}?clientId={client_id}", timeout=self.timeout)
            except ImportError:
                log.debug("websocket-client not installed; falling back to HTTP polling")
            except Exception as e:
                log.debug("WebSocket connection failed, falling back to HTTP polling: %s", e)

            prompt_data = {"prompt": workflow, "client_id": client_id}
            resp = requests.post(f"{self.base_url}/prompt", json=prompt_data, timeout=30)
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
            log.debug("ComfyUI prompt_id: %s", prompt_id)

            if self._progress_cb:
                self._progress_cb(0.2, "工作流已提交到 ComfyUI...")

            history = None
            if ws is not None:
                while True:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        msg_data = json.loads(msg)
                        msg_type = msg_data.get("type")
                        if msg_type == "progress":
                            value = msg_data.get("data", {}).get("value", 0)
                            max_val = msg_data.get("data", {}).get("max", 1)
                            progress = 0.2 + (value / max_val) * 0.6
                            if self._progress_cb:
                                self._progress_cb(progress, f"Sampling step {value}/{max_val}")
                        elif msg_type == "executing":
                            if msg_data.get("data", {}).get("node") is None:
                                break
                    elif isinstance(msg, bytes):
                        pass
            else:
                deadline = time.time() + self.timeout
                last_report = 0.0
                while time.time() < deadline:
                    history_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
                    if history_resp.status_code == 200:
                        candidate = history_resp.json()
                        entry = candidate.get(prompt_id, {})
                        if entry.get("outputs") or entry.get("status", {}).get("status_str") == "error":
                            history = candidate
                            break
                    now = time.time()
                    if self._progress_cb and now - last_report > 5:
                        elapsed = max(0.0, self.timeout - (deadline - now))
                        progress = 0.2 + min(elapsed / max(self.timeout, 1), 1.0) * 0.65
                        self._progress_cb(progress, "ComfyUI 正在执行工作流...")
                        last_report = now
                    time.sleep(1)
                if history is None:
                    raise TimeoutError(f"ComfyUI 工作流轮询超时（{self.timeout} 秒）")

            if self._progress_cb:
                self._progress_cb(0.9, "正在获取结果图像...")

            if history is None:
                history_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
                history_resp.raise_for_status()
                history = history_resp.json()
            outputs = history.get(prompt_id, {}).get("outputs", {})
            log.debug("History outputs nodes: %s", list(outputs.keys()))

            images = []
            pbr_maps = {}
            output_debug = []
            for node_id, node_output in outputs.items():
                # Determine prefix from the workflow definition
                prefix = ""
                if node_id in workflow:
                    node_def = workflow[node_id]
                    if node_def.get("class_type") == "SaveImage":
                        prefix = node_def.get("inputs", {}).get("filename_prefix", "")

                for img_info in node_output.get("images", []):
                    filename = img_info.get('filename', '')
                    output_debug.append({
                        "node": node_id,
                        "prefix": prefix,
                        "filename": filename,
                    })
                    from urllib.parse import urlencode
                    view_query = urlencode({
                        "filename": filename,
                        "subfolder": img_info.get('subfolder', ''),
                        "type": img_info.get('type', 'output'),
                    })
                    img_resp = requests.get(
                        f"{self.base_url}/view?{view_query}",
                        timeout=30,
                    )
                    img_resp.raise_for_status()
                    from ..utils.image_utils import load_image_bytes_to_numpy
                    np_img = load_image_bytes_to_numpy(img_resp.content)
                    images.append(np_img)

                    if output_mode == "by_prefix":
                        map_type = self._map_prefix_to_map_type(prefix)
                        if not map_type:
                            map_type = self._map_prefix_to_map_type(filename)
                        if map_type:
                            pbr_maps[map_type] = np_img

            log.debug("pbr_maps keys: %s", list(pbr_maps.keys()))
            log.debug("Total images fetched: %d", len(images))

            if not images and require_images:
                # 尝试从 history 中提取具体错误信息
                status_info = history.get(prompt_id, {}).get("status", {})
                error_msgs = []
                if status_info.get("status_str") == "error":
                    for msg in status_info.get("messages", []):
                        if isinstance(msg, list) and len(msg) >= 2:
                            error_msgs.append(str(msg[1]))
                if error_msgs:
                    detail = "; ".join(error_msgs[:3])
                    raise RuntimeError(f"ComfyUI 执行失败: {detail}")
                raise RuntimeError("ComfyUI 未返回任何图像，请检查 checkpoint 和 ControlNet 模型是否存在于 ComfyUI 中。")

            if self._progress_cb:
                self._progress_cb(1.0, "完成")

            return GenerationResult(
                images=images,
                seed=-1,
                info={},
                metadata={
                    "backend": "comfyui",
                    "timestamp": time.time(),
                    "prompt_id": prompt_id,
                    "outputs": output_debug,
                },
                pbr_maps=pbr_maps,
            )
        finally:
            if ws is not None:
                ws.close()
