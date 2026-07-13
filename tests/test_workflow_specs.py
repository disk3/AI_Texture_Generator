"""Workflow 子图合并与模型族构建的单元测试。"""

import json
import os
import tempfile
import unittest

import numpy as np

# 必须在导入任何插件模块前安装 bpy mock
from .mock_bpy import install

install()

from AI_Texture_Generator.sd_backend import workflow_specs as specs
from AI_Texture_Generator.sd_backend.comfyui_client import ComfyUIClient
from AI_Texture_Generator.sd_backend.abstract_client import GenerationConfig
from AI_Texture_Generator.preferences import _classify_local_model


class WorkflowMergeTests(unittest.TestCase):
    def _assert_no_dangling_refs(self, workflow):
        all_ids = set(workflow.keys())
        for node_id, node_def in workflow.items():
            for value in node_def.get("inputs", {}).values():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], str)
                    and value[0] not in all_ids
                ):
                    self.fail(f"节点 {node_id} 引用了不存在的节点 {value[0]}")

    def test_zimage_lowres_merge(self):
        family = specs.get_family_spec("zimage")
        gen = specs.load_workflow_template(family["gen_template"])
        post = specs.load_workflow_template(family["post_lowres_template"])
        merged, post_id_map = specs.merge_workflow_subgraphs(
            gen,
            post,
            family["nodes"]["vae_decode"]["id"],
            family["nodes"]["post_input"]["id"],
            family["nodes"]["post_input"].get("field", "image"),
        )
        self._assert_no_dangling_refs(merged)
        post_input_id = specs.resolve_node_id(family, "post_input", post_id_map)
        self.assertEqual(merged[post_input_id]["inputs"]["image"], ["9", 0])

    def test_zimage_hires_merge(self):
        family = specs.get_family_spec("zimage")
        gen = specs.load_workflow_template(family["gen_template"])
        post = specs.load_workflow_template(family["post_hires_template"])
        merged, post_id_map = specs.merge_workflow_subgraphs(
            gen,
            post,
            family["nodes"]["vae_decode"]["id"],
            family["nodes"]["post_input"]["id"],
            family["nodes"]["post_input"].get("field", "image"),
        )
        self._assert_no_dangling_refs(merged)
        basecolor_id = specs.resolve_node_id(family, "seedvr2_basecolor", post_id_map)
        normal_id = specs.resolve_node_id(family, "seedvr2_normal", post_id_map)
        self.assertEqual(merged[basecolor_id]["class_type"], "SeedVR2VideoUpscaler")
        self.assertEqual(merged[normal_id]["class_type"], "SeedVR2VideoUpscaler")


class WorkflowBuilderTests(unittest.TestCase):
    def setUp(self):
        self.client = ComfyUIClient()

    def test_zimage_lowres_overrides(self):
        cfg = GenerationConfig(prompt="brick wall", width=512, height=512, seed=123)
        workflow, seed, chord_first = self.client._build_local_workflow(cfg)
        self.assertEqual(seed, 123)
        self.assertFalse(chord_first)
        self.assertIn("四向无缝", workflow["4"]["inputs"]["text"])
        self.assertEqual(workflow["post_11"]["inputs"]["target_width"], 512)

    def test_zimage_hires_overrides(self):
        cfg = GenerationConfig(prompt="brick wall", width=2048, height=2048, seed=123)
        workflow, seed, chord_first = self.client._build_local_workflow(cfg)
        self.assertTrue(chord_first)
        self.assertEqual(workflow["post_11"]["inputs"]["target_width"], 1024)
        self.assertEqual(workflow["post_111"]["inputs"]["resolution"], 2048)

    def test_default_sentinel_not_applied(self):
        """模型选择为 DEFAULT / __EMPTY__ / NONE 时不应覆盖工作流默认值。"""
        cfg = GenerationConfig(
            prompt="brick wall",
            width=1024,
            height=1024,
            seed=42,
            model_family="zimage",
            model_selections={
                "main_model": "DEFAULT",
                "vae": "NONE",
                "text_encoder": "",
            },
        )
        workflow, _, _ = self.client._build_local_workflow(cfg)
        self.assertEqual(workflow["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(workflow["3"]["inputs"]["clip_name"], "qwen_3_4b.safetensors")
        self.assertEqual(workflow["8"]["inputs"]["vae_name"], "ae.sft")

    def test_zimage_reference_image(self):
        cfg = GenerationConfig(
            prompt="brick wall",
            width=1024,
            height=1024,
            seed=42,
            model_family="zimage",
            model_selections={"vae": "ae.sft"},
            init_image=np.zeros((64, 64, 3), dtype=np.uint8),
            denoising_strength=0.6,
        )
        self.client._upload_image = lambda image, name="ref": "ref.png"
        workflow, seed, _ = self.client._build_local_workflow(cfg)
        self.assertEqual(workflow["100"]["inputs"]["image"], "ref.png")
        self.assertEqual(workflow["101"]["inputs"]["vae"], ["8", 0])
        self.assertEqual(workflow["7"]["inputs"]["latent_image"], ["101", 0])
        self.assertEqual(workflow["7"]["inputs"]["denoise"], 0.6)


class ModelClassificationTests(unittest.TestCase):
    def test_known_models(self):
        cases = [
            ("z_image_turbo_bf16.safetensors", "unet", "zimage"),
            ("ae.sft", "vae", "zimage"),
            ("flux1-dev.sft", "unet", "flux1"),
            ("some_random.safetensors", "unet", ""),
        ]
        for filename, kind, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(_classify_local_model(filename, kind), expected)


class DesktopSharedModelTests(unittest.TestCase):
    def _write_yaml(self, root: str, content: str) -> str:
        path = os.path.join(root, "extra_model_paths.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_parse_extra_model_paths_simple(self):
        from AI_Texture_Generator.sd_backend.comfyui_installer import _parse_extra_model_paths_simple
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._write_yaml(
                tmp,
                "shared:\n"
                "  base_path: D:/AI\n"
                "  checkpoints: models/checkpoints/\n"
                "  diffusion_models: |\n"
                "    models/diffusion_models/\n"
                "    models/unet/\n"
                "  vae: models/vae/\n",
            )
            dirs = _parse_extra_model_paths_simple(cfg)
            self.assertIn(os.path.normpath("D:/AI/models/checkpoints"), dirs)
            self.assertIn(os.path.normpath("D:/AI/models/diffusion_models"), dirs)
            self.assertIn(os.path.normpath("D:/AI/models/unet"), dirs)
            self.assertIn(os.path.normpath("D:/AI/models/vae"), dirs)

    def test_find_model_in_desktop_shared(self):
        from AI_Texture_Generator.sd_backend.comfyui_installer import find_model_file
        with tempfile.TemporaryDirectory() as tmp:
            # 模拟桌面版根目录，没有 ComfyUI/models，只有 extra_model_paths.yaml 指向共享目录
            shared_root = os.path.join(tmp, "SharedModels")
            model_dir = os.path.join(shared_root, "models", "diffusion_models")
            os.makedirs(model_dir)
            target = os.path.join(model_dir, "z_image_turbo_bf16.safetensors")
            open(target, "w").close()

            cfg = self._write_yaml(
                tmp,
                "shared:\n"
                f"  base_path: {shared_root.replace(chr(92), '/')}\\n"
                "  diffusion_models: models/diffusion_models/\n",
            )

            model = {
                "id": "zimage_main",
                "filename": "z_image_turbo_bf16.safetensors",
                "dir": "models/diffusion_models",
            }
            found = find_model_file(tmp, model)
            self.assertIsNotNone(found)
            self.assertTrue(os.path.isfile(found))


if __name__ == "__main__":
    unittest.main()
