"""CPU regression tests; no ComfyUI server or model downloads are required."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conditioner = load_module(
    "hy3d_test_conditioner", "hy3dshape/hy3dshape/models/conditioner.py"
)


def make_conditioner():
    return conditioner.SingleImageEncoder({
        "type": "DinoImageEncoder",
        "kwargs": {
            "image_size": 28,
            "config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "mlp_ratio": 2,
                "image_size": 28,
                "patch_size": 14,
            },
        },
    }).eval()


def legacy_checkpoint(model):
    """Serialize the tiny model using the official Hunyuan checkpoint schema."""
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for layer in range(2):
        base = f"main_image_encoder.model.encoder.layer.{layer}.attention."
        for projection, legacy in (
            ("q_proj", "attention.query"),
            ("k_proj", "attention.key"),
            ("v_proj", "attention.value"),
            ("o_proj", "output.dense"),
        ):
            for parameter in ("weight", "bias"):
                current_key = f"{base}{projection}.{parameter}"
                if current_key in state:
                    state[f"{base}{legacy}.{parameter}"] = state.pop(current_key)
    return state


class DinoCheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_loads_through_parent_with_identical_output(self):
        reference = make_conditioner()
        checkpoint = legacy_checkpoint(reference)
        target = make_conditioner()
        result = target.load_state_dict(checkpoint, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
        pixels = torch.rand(1, 3, 28, 28)
        with torch.no_grad():
            torch.testing.assert_close(target(pixels)["main"], reference(pixels)["main"])

    def test_current_checkpoint_still_loads(self):
        reference = make_conditioner()
        target = make_conditioner()
        target.load_state_dict(reference.state_dict(), strict=True)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)

    def test_missing_attention_weight_is_not_ignored(self):
        target = make_conditioner()
        state = legacy_checkpoint(target)
        del state["main_image_encoder.model.encoder.layer.0.attention.attention.query.weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            target.load_state_dict(state, strict=True)

    def test_unexpected_weight_is_not_ignored(self):
        target = make_conditioner()
        state = legacy_checkpoint(target)
        state["main_image_encoder.model.unknown.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            target.load_state_dict(state, strict=True)

    def test_wrong_attention_shape_is_not_ignored(self):
        target = make_conditioner()
        state = legacy_checkpoint(target)
        state["main_image_encoder.model.encoder.layer.0.attention.attention.query.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            target.load_state_dict(state, strict=True)

    def test_additional_dino_encoder_uses_its_own_prefix(self):
        source = make_conditioner()
        reference = torch.nn.Module()
        reference.add_module("additional_image_encoder", make_conditioner().main_image_encoder)
        checkpoint = {
            key.replace("main_image_encoder.", "additional_image_encoder.", 1): value
            for key, value in legacy_checkpoint(source).items()
        }
        reference.load_state_dict(checkpoint, strict=True)
        for key, value in source.main_image_encoder.state_dict().items():
            torch.testing.assert_close(
                reference.additional_image_encoder.state_dict()[key], value, rtol=0, atol=0
            )


class PaintLoadingTests(unittest.TestCase):
    def test_official_custom_model_and_current_vae_api(self):
        # Isolate the GPU pipeline while exercising the real loader constructor.
        # The spec excludes the removed pipeline-level VAE convenience methods.
        pipeline = Mock(spec=[
            "scheduler", "set_progress_bar_config", "eval", "view_size",
            "enable_model_cpu_offload", "to", "vae", "unet",
        ])
        pipeline.scheduler = SimpleNamespace(config={})
        pipeline.to.return_value = pipeline
        pipeline.vae = Mock(spec=["enable_slicing", "enable_tiling"])
        pipeline.unet = SimpleNamespace()

        def from_pretrained(*args, **kwargs):
            if not kwargs.get("trust_remote_code"):
                raise ValueError("Paint checkpoint requires trust_remote_code=True")
            return pipeline

        pipeline_class = Mock()
        pipeline_class.from_pretrained.side_effect = from_pretrained
        packages = {}
        for name in ("hy3d_test_paint", "hy3d_test_paint.utils", "hy3d_test_paint.hunyuanpaintpbr"):
            packages[name] = ModuleType(name)
            packages[name].__path__ = []
        pipeline_module = ModuleType("hy3d_test_paint.hunyuanpaintpbr.pipeline")
        pipeline_module.HunyuanPaintPipeline = pipeline_class
        packages[pipeline_module.__name__] = pipeline_module

        with patch.dict(sys.modules, packages):
            module = load_module(
                "hy3d_test_paint.utils.multiview_utils", "hy3dpaint/utils/multiview_utils.py"
            )
        config = SimpleNamespace(
            device="cpu", multiview_cfg_path="unused.yaml", custom_pipeline="unused",
            multiview_pretrained_path="tencent/Hunyuan3D-2.1",
        )
        cfg = module.OmegaConf.create({
            "model": {"params": {
                "stable_diffusion_config": {"custom_pipeline": "./hunyuanpaintpbr"},
                "view_size": 768,
            }},
        })
        with (
            patch.object(module.OmegaConf, "load", return_value=cfg),
            patch.object(module.huggingface_hub, "snapshot_download", return_value="/models"),
            patch.object(module, "EulerAncestralDiscreteScheduler"),
        ):
            loaded = module.multiviewDiffusionNet(config)

        self.assertIs(loaded.pipeline, pipeline)
        pipeline_class.from_pretrained.assert_called_once_with(
            "/models/hunyuan3d-paintpbr-v2-1", torch_dtype=torch.float16,
            use_fast=False, trust_remote_code=True,
        )
        pipeline.vae.enable_slicing.assert_called_once_with()
        pipeline.vae.enable_tiling.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
