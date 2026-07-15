"""LMFTrack model developed from the TBSI and OSTrack codebases."""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.lmftrack.vit_lmftrack import vit_base_patch16_224_lmftrack
from lib.models.lmftrack.checkpoint_utils import remap_legacy_state_dict
from lib.utils.box_ops import box_xyxy_to_cxcywh


class LMFTrack(nn.Module):
    """Language-guided RGB-T tracker."""

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.search_fusion = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.search_fusion(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_lmftrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_dir = os.path.abspath(os.path.join(current_dir, '../../../pretrained_models'))
    pretrained_name = str(cfg.MODEL.PRETRAIN_FILE or '')
    pretrained_file = os.path.join(pretrained_dir, pretrained_name) if pretrained_name else ''
    legacy_pretrained_file = os.path.join(pretrained_dir, 'TBSITrack_SOT_Pretrained.pth.tar')
    if pretrained_name == 'LMFTrack_SOT_Pretrained.pth.tar' and not os.path.isfile(pretrained_file) \
            and os.path.isfile(legacy_pretrained_file):
        pretrained_file = legacy_pretrained_file

    # The released initialization checkpoint is a complete tracker checkpoint
    # inherited from the earlier TBSI-named development version. Generic ViT or
    # OSTrack backbone checkpoints are instead loaded inside the backbone factory.
    tracker_checkpoint = any(token in os.path.basename(pretrained_name) for token in ('TBSITrack', 'LMFTrack'))
    backbone_pretrained = pretrained_file if training and pretrained_name and not tracker_checkpoint else ''

    if backbone_pretrained:
        if not os.path.isfile(backbone_pretrained):
            raise FileNotFoundError(f'Backbone checkpoint not found: {backbone_pretrained}')
        print(f'Load backbone checkpoint from: {backbone_pretrained}')

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_lmftrack':
        backbone = vit_base_patch16_224_lmftrack(
            backbone_pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            fusion_loc=cfg.MODEL.BACKBONE.FUSION_LOC,
            fusion_drop_path=cfg.TRAIN.FUSION_DROP_PATH,
        )
    else:
        raise NotImplementedError(f'Unsupported backbone: {cfg.MODEL.BACKBONE.TYPE}')

    hidden_dim = backbone.embed_dim
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    box_head = build_box_head(cfg, hidden_dim)

    model = LMFTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if training and tracker_checkpoint:
        if not os.path.isfile(pretrained_file):
            raise FileNotFoundError(f'Tracker checkpoint not found: {pretrained_file}')
        checkpoint = torch.load(pretrained_file, map_location='cpu')
        state_dict = remap_legacy_state_dict(checkpoint.get('net', checkpoint))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f'Loaded tracker checkpoint from: {pretrained_file}')
        if missing_keys:
            print('Missing keys:', missing_keys)
        if unexpected_keys:
            print('Unexpected keys:', unexpected_keys)

    return model
