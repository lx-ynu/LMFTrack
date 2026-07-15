from lib.models.lmftrack import build_lmftrack
from lib.models.lmftrack.checkpoint_utils import remap_legacy_state_dict
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class LMFTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(LMFTrack, self).__init__(params)
        network = build_lmftrack(params.cfg, training=False)
        if not os.path.isfile(self.params.checkpoint):
            raise FileNotFoundError(f'LMFTrack checkpoint not found: {self.params.checkpoint}')
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = remap_legacy_state_dict(checkpoint.get('net', checkpoint))
        missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print('Missing checkpoint keys:', missing_keys)
        if unexpected_keys:
            print('Unexpected checkpoint keys:', unexpected_keys)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def _get_cspg_module(self):
        """Return the CSPG module in the backbone.

        The expected module name is self.network.backbone.cspg.
        This helper keeps the tracker code robust and makes debugging easier.
        """
        if hasattr(self.network, "backbone") and hasattr(self.network.backbone, "cspg"):
            return self.network.backbone.cspg

        # Fallback for possible DataParallel-like wrappers.
        if hasattr(self.network, "module") and hasattr(self.network.module, "backbone") \
                and hasattr(self.network.module.backbone, "cspg"):
            return self.network.module.backbone.cspg

        return None

    def reset_cspg_cache(self):
        cspg_module = self._get_cspg_module()
        if cspg_module is not None:
            cspg_module.reset_template_cache()
        else:
            print("[CSPG Warning] CSPG module was not found in the backbone.")

    def set_cspg_template(self, template_patch):
        cspg_module = self._get_cspg_module()
        if cspg_module is not None:
            cspg_module.set_external_template(template_patch)
        else:
            print("[CSPG Warning] CSPG module was not found in the backbone.")

    def get_cspg_info(self):
        cspg_module = self._get_cspg_module()
        if cspg_module is not None:
            return cspg_module.get_last_cspg_info()

        return {
            "selected_class": None,
            "selected_class_index": -1,
            "semantic_confidence": -1.0,
        }

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'], self.params.template_factor,
            output_sz=self.params.template_size
        )

        # Provide the raw RGB template patch to CSPG before preprocessing.
        # set_external_template() already clears the old CSPG cache.
        self.z_patch_arr = z_patch_arr
        self.set_cspg_template(z_patch_arr)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            num_queries = int(getattr(self.cfg.MODEL, 'NUM_OBJECT_QUERIES', 1))
            all_boxes_save = info['init_bbox'] * num_queries
            return {'all_boxes': all_boxes_save, 'all_scores': [1.0] * num_queries}

    def update_template(self, image, bbox):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, bbox, self.params.template_factor,
            output_sz=self.params.template_size
        )

        # Provide the updated raw RGB template patch to CSPG before preprocessing.
        # set_external_template() already clears the old CSPG cache.
        self.z_patch_arr = z_patch_arr
        self.set_cspg_template(z_patch_arr)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_dict1 = template
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                bbox, resize_factor, template.tensors.device
            ).squeeze(1)
            self.box_mask_z = generate_mask_cond(
                self.cfg, 1, template.tensors.device, template_bbox
            )

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=[self.z_dict1.tensors[:, :3, :, :], self.z_dict1.tensors[:, 3:, :, :]],
                search=[x_dict.tensors[:, :3, :, :], x_dict.tensors[:, 3:, :, :]],
                ce_template_mask=self.box_mask_z,
            )

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        # pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        cspg_info = self.get_cspg_info()
        template_update_flag = int(max_score > self.params.template_update_threshold)

        cspg_record = {
            "frame_id": self.frame_id,
            "selected_class": cspg_info["selected_class"],
            "selected_class_index": cspg_info["selected_class_index"],
            "semantic_confidence": cspg_info["semantic_confidence"],
            "template_update_flag": template_update_flag,
            "tracking_score": max_score,
        }
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                visible_image = image[..., :3]
                image_BGR = cv2.cvtColor(visible_image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image[..., :3], info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr[..., :3]).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[..., :3]).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()

            return {
                "target_bbox": self.state,
                "all_boxes": all_boxes_save,
                "all_scores": [max_score] * pred_boxes.shape[0],
                "best_score": max_score,
                "tracking_score": max_score,
                "frame_id": self.frame_id,
                "selected_class": cspg_info["selected_class"],
                "selected_class_index": cspg_info["selected_class_index"],
                "semantic_confidence": cspg_info["semantic_confidence"],
                "template_update_flag": template_update_flag,
                "cspg_records": cspg_record,
            }
        else:
            return {
                "target_bbox": self.state,
                "best_score": max_score,
                "tracking_score": max_score,
                "frame_id": self.frame_id,
                "selected_class": cspg_info["selected_class"],
                "selected_class_index": cspg_info["selected_class_index"],
                "semantic_confidence": cspg_info["semantic_confidence"],
                "template_update_flag": template_update_flag,
                "cspg_records": cspg_record,
            }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)



def get_tracker_class():
    return LMFTrack
