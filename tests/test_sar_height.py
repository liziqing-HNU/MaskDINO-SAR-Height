import unittest

import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config
from maskdino.maskdino import MaskDINO
from maskdino.modeling.criterion import SetCriterion
from maskdino.sar_height.geometry_decoder import GeometryDecoder
from maskdino.sar_height.global_correlation import GlobalCorrelation
from maskdino.sar_height.local_correlation import (
    LocalCorrelation,
    shift_reference_boxes,
)
from maskdino.sar_height.maskdino_sar_height import MaskDINOSARHeightBranch
from maskdino.sar_height.phase_descriptor import PhaseDescriptorGenerator
from maskdino.sar_height.sar_resnet_fpn import SARResNet18FPN
from maskdino.sar_height.sar_stem import SARLightweightStem


class SARHeightModuleTests(unittest.TestCase):
    def test_phase_descriptor_shape_and_no_circular_boundary(self):
        sin_phi = torch.zeros(1, 1, 9, 9)
        cos_phi = torch.ones_like(sin_phi)
        coherence = torch.ones_like(sin_phi)
        sin_phi[..., 4, 8] = 1.0
        cos_phi[..., 4, 8] = 0.0
        descriptor = PhaseDescriptorGenerator()(sin_phi, cos_phi, coherence)
        self.assertEqual(tuple(descriptor.shape), (1, 39, 9, 9))
        # First neighbourhood is (x+1,y). The right-edge source must neither
        # wrap to the left nor invent an out-of-bounds pair on the right.
        self.assertEqual(float(descriptor[0, 3, 4, 0]), 0.0)
        self.assertEqual(float(descriptor[0, 3:6, 4, 8].abs().sum()), 0.0)

    def test_global_shift_sign(self):
        torch.manual_seed(7)
        optical = torch.nn.functional.normalize(
            torch.randn(1, 8, 16, 16), dim=1
        )
        sar = torch.zeros_like(optical)
        sar[..., 1:] = optical[..., :-1]
        sar = torch.nn.functional.normalize(sar, dim=1)
        optical_valid = torch.ones(1, 16, 16, dtype=torch.bool)
        sar_valid = optical_valid.clone()
        sar_valid[..., 0] = False
        offset, details = GlobalCorrelation(
            search_radius=2, temperature=0.01, matching_stride=8
        )(
            optical,
            sar,
            image_size=(128, 128),
            optical_valid_mask=optical_valid,
            sar_valid_mask=sar_valid,
            return_details=True,
        )
        self.assertEqual(tuple(offset.shape), (1, 2))
        self.assertGreater(float(details["offset_feature_pixels"][0, 0]), 0.9)
        self.assertAlmostEqual(float(offset[0, 0]), 1.0 / 16.0, places=4)

    def test_offset_shapes_and_disabled_alignment_identity(self):
        query = torch.randn(2, 5, 256)
        boxes = torch.rand(2, 5, 4)
        feature = torch.nn.functional.normalize(torch.randn(2, 64, 8, 8), dim=1)
        global_offset = torch.zeros(2, 2)
        local = LocalCorrelation(search_radius=1)(
            query,
            boxes,
            feature,
            feature,
            global_offset,
            image_size=(64, 64),
        )
        self.assertEqual(tuple(local.shape), (2, 5, 2))
        unchanged = shift_reference_boxes(
            boxes, global_offset, torch.zeros_like(local)
        )
        self.assertTrue(torch.equal(unchanged, boxes))

    def test_sar_encoder_and_geometry_shapes(self):
        stem = SARLightweightStem()
        encoder = SARResNet18FPN(pretrained_weights="", freeze_batch_norm=True)
        stem_feature, _ = stem(
            torch.randn(1, 39, 128, 128), torch.randn(1, 2, 128, 128)
        )
        pyramid, _ = encoder(stem_feature)
        expected = [(32, 32), (16, 16), (8, 8), (4, 4)]
        self.assertEqual([item.shape[-2:] for item in pyramid], expected)
        self.assertTrue(all(item.shape[1] == 256 for item in pyramid))

        query = torch.randn(1, 5, 256, requires_grad=True)
        boxes = torch.rand(1, 5, 4)
        boxes[..., 2:] = boxes[..., 2:] * 0.3 + 0.05
        small_pyramid = [
            torch.randn(1, 256, size, size, requires_grad=True)
            for size in (16, 8, 4, 2)
        ]
        decoded = GeometryDecoder(num_layers=3, dropout=0.0)(
            query,
            boxes,
            small_pyramid,
            valid_mask=torch.ones(1, 64, 64, dtype=torch.bool),
        )
        self.assertEqual(tuple(decoded.shape), (1, 5, 256))
        decoded.square().mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(all(item.grad is not None for item in small_pyramid))

    def test_height_loss_ignores_unmatched_queries(self):
        criterion = SetCriterion(
            num_classes=1,
            matcher=None,
            weight_dict={"loss_height": 1.0},
            eos_coef=0.1,
            losses=["heights"],
            num_points=4,
            oversample_ratio=1.0,
            importance_sample_ratio=0.5,
            height_scale=1.0,
            height_loss_beta=1.0,
        )
        predictions = torch.tensor([[10.0, 20.0, 9999.0]])
        targets = [
            {
                "heights": torch.tensor([20.0, 10.0]),
                "height_valid": torch.tensor([True, True]),
            }
        ]
        indices = [(torch.tensor([0, 1]), torch.tensor([1, 0]))]
        loss = criterion.loss_heights(
            {"pred_heights": predictions}, targets, indices, num_boxes=2
        )["loss_height"]
        self.assertEqual(float(loss), 0.0)

    def test_postprocessor_uses_one_query_index(self):
        model = MaskDINO.__new__(MaskDINO)
        nn.Module.__init__(model)
        model.register_buffer("pixel_mean", torch.zeros(3, 1, 1), persistent=False)
        model.sem_seg_head = type("Head", (), {"num_classes": 1})()
        model.num_queries = 4
        model.test_topk_per_image = 3
        model.panoptic_on = False
        model.focus_on_box = False
        model.height_scale = 1.0
        mask_cls = torch.tensor([[4.0], [1.0], [3.0], [2.0]])
        masks = torch.randn(4, 8, 8)
        boxes = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0],
             [2.0, 2.0, 3.0, 3.0], [3.0, 3.0, 4.0, 4.0]]
        )
        heights = torch.tensor([11.0, 22.0, 33.0, 44.0])
        result = model.instance_inference(mask_cls, masks, boxes, heights)
        selected = result.pred_query_indices
        self.assertTrue(torch.equal(result.pred_heights, heights[selected]))
        self.assertTrue(torch.equal(result.pred_boxes.tensor, boxes[selected]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA gradient smoke test")
    def test_full_branch_propagates_nonzero_height_gradients(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.MODEL.SAR.ENABLED = True
        cfg.MODEL.SAR.RESNET18_WEIGHTS = ""
        cfg.MODEL.ALIGN.GLOBAL_SEARCH_RADIUS = 1
        cfg.MODEL.ALIGN.LOCAL_SEARCH_RADIUS = 1
        cfg.MODEL.GEOMETRY_DECODER.DROPOUT = 0.0
        cfg.MODEL.HEIGHT.DROPOUT = 0.0
        branch = MaskDINOSARHeightBranch(cfg).cuda().train()
        height = width = 64
        query = torch.randn(1, 3, 256, device="cuda")
        boxes = torch.rand(1, 3, 4, device="cuda")
        boxes[..., 2:] = boxes[..., 2:] * 0.25 + 0.05
        optical = torch.randn(1, 256, 8, 8, device="cuda")
        sar = torch.randn(1, 4, height, width, device="cuda")
        sar[:, 0:2] = torch.nn.functional.normalize(sar[:, 0:2], dim=1)
        sar[:, 2:3] = torch.sigmoid(sar[:, 2:3])
        sar[:, 3:4] = sar[:, 3:4] * 4.0 - 12.0
        valid = torch.ones(1, height, width, dtype=torch.bool, device="cuda")
        output = branch(query, boxes, optical, sar, valid, valid)
        output["pred_heights"].mean().backward()
        gradients = [
            branch.height_head.linear2.weight.grad,
            branch.geometry_decoder.layers[0].cross_attention.value_proj.weight.grad,
            branch.encoder.output_convs[0].weight.grad,
            branch.encoder.layer4[-1].conv2.weight.grad,
            branch.stem.phase_stem[0][0].weight.grad,
            branch.stem.amp_stem[0][0].weight.grad,
        ]
        self.assertTrue(
            all(
                gradient is not None
                and torch.isfinite(gradient).all()
                and float(gradient.abs().sum()) > 0
                for gradient in gradients
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA optical parity test")
    def test_disabling_sar_preserves_optical_class_box_and_mask(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(
            "configs/sp6/instance-segmentation/"
            "maskdino_R50_bs16_50ep_height.yaml"
        )
        cfg.defrost()
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.SAR.RESNET18_WEIGHTS = ""
        cfg.freeze()
        torch.manual_seed(31)
        model = build_model(cfg).cuda().eval()
        image = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
        sar = torch.randn(4, 128, 128)
        sar[0:2] = torch.nn.functional.normalize(sar[0:2], dim=0)
        sar[2:3] = torch.sigmoid(sar[2:3])
        sar[3:4] = sar[3:4] * 4.0 - 12.0
        multimodal_input = {
            "image": image,
            "sar_image": sar,
            "sar_valid_mask": torch.ones(128, 128, dtype=torch.bool),
            "height": 128,
            "width": 128,
        }
        with torch.no_grad():
            enabled = model([multimodal_input])[0]["instances"]
            model.sar_enabled = False
            model.height_on = False
            disabled = model([{"image": image, "height": 128, "width": 128}])[0][
                "instances"
            ]
        for field in (
            "scores",
            "pred_classes",
            "pred_query_indices",
            "pred_masks",
        ):
            self.assertTrue(
                torch.equal(enabled.get(field), disabled.get(field)), field
            )
        self.assertTrue(
            torch.equal(enabled.pred_boxes.tensor, disabled.pred_boxes.tensor)
        )


if __name__ == "__main__":
    unittest.main()
