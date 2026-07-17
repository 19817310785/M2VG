# -*- coding: utf-8 -*-

import cv2
import hashlib
import math
import random
import numpy as np
from PIL import Image
from collections import Iterable
import pycocotools.mask as maskUtils
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from mmdet.core import BitmapMasks
from transformers import XLMRobertaTokenizer
from ..builder import PIPELINES   # <-- 保留这个原配的注册表

import random
import numpy as np
# 删掉了那句 mmdet 的导入

import random
from ..builder import PIPELINES


@PIPELINES.register_module()
class StaticHardNegative:
    def __init__(self, prob=0.2):
        self.prob = prob
        # 这里放一个备用的干扰词表，或者您可以直接套用您原本的替换逻辑
        self.hard_texts = [
            "the blue car", "a man in red", "the dog on the grass",
            "a person riding a bike", "the black cat", "a large building"
        ]

    def __call__(self, results):
        # 1. 安全检查：保护原本就是无目标的 empty 样本
        is_empty = results.get('empty', False)
        if is_empty:
            results['gt_exist'] = np.array([0], dtype=np.int64)
            return results

        # ================= 核心 Hack：锁定随机种子 =================
        # 获取当前图像的 ID。不同的图片 ID 不同，同一张图片 ID 永远不变
        img_info = results.get('img_info', {})
        img_id = img_info.get('id', 12345)

        # 确保 img_id 是一个整数，以便作为 random.seed() 的有效输入
        if isinstance(img_id, str):
            img_id = int(hashlib.md5(img_id.encode('utf-8')).hexdigest(), 16) % (10 ** 8)

        # 【极其重要】记住当前系统全局的随机状态！
        # 否则固定种子会破坏后续 RandomFlip 等数据增强的随机性
        original_state = random.getstate()

        # 将这部分代码的随机种子，死死钉在这个图像的 ID 上！
        random.seed(img_id)
        # =========================================================

        # 因为种子固定了，这张图片每次走到这里，产生的 random() 概率值永远一模一样！
        if random.random() < self.prob:
            # 这张图片永远会被分配到固定的一句假话（因为 choice 的结果也被种子锁死了）
            # 如果您之前有自己写好的单词替换逻辑（比如换颜色/名词），直接放在这里执行即可
            results['text'] = random.choice(self.hard_texts)

            # 清空标签体系 (安全写法)
            results['gt_bbox'] = np.zeros((0, 4), dtype=np.float32)
            if 'gt_mask_rle' in results:
                results['gt_mask_rle'] = []
            if 'gt_mask_parts_rle' in results:
                results['gt_mask_parts_rle'] = []

            # 强制打上负样本标签
            results['gt_exist'] = np.array([0], dtype=np.int64)
        else:
            # 正常样本，赋予 gt_exist = 1
            results['gt_exist'] = np.array([1], dtype=np.int64)

        # ================= 核心 Hack：恢复随机状态 =================
        # 办完事后，立刻把系统的随机状态还回去！深藏功与名。
        random.setstate(original_state)
        # =========================================================

        return results

@PIPELINES.register_module()
class RandomNegative:
    def __init__(self, prob=0.2):
        self.prob = prob
        # 准备一个“毫不相干”的随机文本池
        # （您可以根据需要自己多加几句，越和原始图片不搭边越好）
        self.random_texts = [
            "a flying elephant in the sky",
            "someone is cooking pasta in the kitchen",
            "a tiny black cat sleeping on the sofa",
            "the man playing basketball",
            "a large ship sailing on the ocean"
        ]

    def __call__(self, results):
        # 1. 检查这个样本是否原本在 GRefCOCO 数据集里就是 "无目标" (empty)
        is_empty = results.get('empty', False)

        if is_empty:
            # 如果本身就是无目标样本，顺理成章地将存在性设为 0
            results['gt_exist'] = np.array([0], dtype=np.int64)
        else:
            # 2. 如果是有目标的正常样本，按 prob (20%) 概率构造“随机无关负样本”
            if random.random() < self.prob:

                # 👉 【核心替换逻辑】：将原本正确的文本，强行替换为毫不相干的废话
                results['text'] = random.choice(self.random_texts)

                # 清空 BBox (安全写法)
                results['gt_bbox'] = np.zeros((0, 4), dtype=np.float32)

                # 清空 Mask (安全写法)
                if 'gt_mask_rle' in results:
                    results['gt_mask_rle'] = []
                if 'gt_mask_parts_rle' in results:
                    results['gt_mask_parts_rle'] = []

                # 设置存在性标签为 0
                results['gt_exist'] = np.array([0], dtype=np.int64)

            else:
                # 正常样本，必须赋予 gt_exist = 1 标签
                results['gt_exist'] = np.array([1], dtype=np.int64)

        return results

@PIPELINES.register_module()
class DynamicFocusNegativeMining:
    """Build conservative no-target samples for expression satisfiability."""

    def __init__(
        self,
        prob=0.2,
        removal_prob=0.5,
        semantic_sample_size=256,
        enable_target_removal=True,
        enable_semantic_mismatch=True,
        tokenizer_path="pretrain_weights/beit3.spm",
        min_context_ratio=0.8,
        dilation_iter=3,
        max_patch_trials=30,
    ):
        self.prob = prob
        self.removal_prob = removal_prob
        self.semantic_sample_size = semantic_sample_size
        self.enable_target_removal = enable_target_removal
        self.enable_semantic_mismatch = enable_semantic_mismatch
        self.tokenizer_path = tokenizer_path
        self.min_context_ratio = min_context_ratio
        self.dilation_iter = dilation_iter
        self.max_patch_trials = max_patch_trials
        self.tokenizer = None

    def __call__(self, results):
        if results.get("empty", False):
            return self._mark_negative(results)
        if random.random() >= self.prob:
            return self._mark_positive(results)

        prefer_removal = random.random() < self.removal_prob
        if self.enable_target_removal and (prefer_removal or not self.enable_semantic_mismatch) and self._can_remove_target(results):
            return self._build_target_removed(results)

        if self.enable_semantic_mismatch and self._build_semantic_mismatch(results):
            return results

        if self.enable_target_removal and self.enable_semantic_mismatch and self._can_remove_target(results):
            return self._build_target_removed(results)

        return self._mark_positive(results)

    def _mark_positive(self, results):
        results["gt_exist"] = np.array([1], dtype=np.int64)
        return results

    def _mark_negative(self, results):
        results["gt_exist"] = np.array([0], dtype=np.int64)
        return results

    def _get_category_ids(self, results):
        ann = results.get("ann", {})
        categories = []
        for instance in ann.get("annotations", []):
            if instance.get("empty", False):
                continue
            category_id = instance.get("category_id", None)
            if category_id is not None:
                categories.append(category_id)
        category_id = ann.get("category_id", None)
        if category_id is not None:
            categories.append(category_id)
        return categories

    def _can_remove_target(self, results):
        categories = self._get_category_ids(results)
        if len(categories) != 1 or not results.get("gt_mask_rle", None):
            return False
        mask = maskUtils.decode(results["gt_mask_rle"])
        return mask is not None and mask.sum() > 0

    def _build_target_removed(self, results):
        mask = maskUtils.decode(results["gt_mask_rle"]).astype(np.uint8)
        results["img"] = self._replace_masked_region(results["img"], self._dilate_mask(mask))
        return self._clear_targets(results)

    def _build_semantic_mismatch(self, results):
        expression_pool = results.get("expression_pool", [])
        image_categories = set(self._get_category_ids(results))
        if not expression_pool or not image_categories:
            return False
        sampled_pool = random.sample(expression_pool, min(len(expression_pool), self.semantic_sample_size))
        current_expression = results.get("expression", "")
        candidates = []
        for item in sampled_pool:
            text = item.get("text", "").strip()
            category_ids = set(item.get("category_ids", ()))
            if text and text != current_expression and category_ids and category_ids.isdisjoint(image_categories):
                candidates.append(text)
        if not candidates:
            return False
        negative_text = max(candidates, key=lambda text: self._token_similarity(current_expression, text))
        results["expression"] = negative_text
        results["text"] = negative_text
        self._retokenize(results, negative_text)
        self._clear_targets(results)
        return True

    def _clear_targets(self, results):
        h, w = results["ori_shape"][:2]
        zero_mask = np.zeros((h, w), dtype=np.uint8)
        zero_rle = maskUtils.encode(np.asfortranarray(zero_mask))
        results["gt_bbox"] = np.zeros((0, 4), dtype=np.float32)
        results["gt_ori_mask"] = zero_rle
        results["gt_mask"] = BitmapMasks(zero_mask[None], h, w)
        results["gt_mask_rle"] = zero_rle
        results["gt_mask_parts"] = []
        results["gt_mask_parts_rle"] = []
        results["target"] = []
        results["empty"] = True
        results["gt_exist"] = np.array([0], dtype=np.int64)
        return results

    def _dilate_mask(self, mask):
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=self.dilation_iter)

    def _replace_masked_region(self, img, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return img
        out = img.copy()
        h, w = img.shape[:2]
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        patch_w, patch_h = x2 - x1, y2 - y1
        valid_context = mask == 0
        for _ in range(self.max_patch_trials):
            if patch_w >= w or patch_h >= h:
                break
            cx = random.randint(0, w - patch_w)
            cy = random.randint(0, h - patch_h)
            context_crop = valid_context[cy : cy + patch_h, cx : cx + patch_w]
            if context_crop.mean() < self.min_context_ratio:
                continue
            source_patch = img[cy : cy + patch_h, cx : cx + patch_w]
            target_patch = out[y1:y2, x1:x2]
            alpha = mask[y1:y2, x1:x2].astype(bool)
            target_patch[alpha] = source_patch[alpha]
            out[y1:y2, x1:x2] = target_patch
            return out
        return cv2.inpaint(out, (mask > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    def _retokenize(self, results, expression):
        max_token = int(results.get("max_token", 50))
        if "text_attention_mask" in results:
            tokenizer = self._get_tokenizer()
            tokenized_words = tokenizer.tokenize(expression)
            tokens = tokenizer.convert_tokens_to_ids(tokenized_words)
            if len(tokens) == 0:
                tokens = [tokenizer.unk_token_id]
            tokens = tokens[: max_token - 2]
            token_ids = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
            num_tokens = len(token_ids)
            padding_mask = [0] * num_tokens + [1] * (max_token - num_tokens)
            ref_expr_inds = token_ids + [tokenizer.pad_token_id] * (max_token - num_tokens)
            results["ref_expr_inds"] = np.array(ref_expr_inds, dtype=int)
            results["text_attention_mask"] = np.array(padding_mask, dtype=int)
            results["tokenized_words"] = tokenized_words
            return
        token2idx = results.get("token2idx", {})
        ref_expr_inds = np.zeros(max_token, dtype=np.int64)
        for idx, word in enumerate(expression.lower().split()[:max_token]):
            ref_expr_inds[idx] = token2idx.get(word, token2idx.get("UNK", 0))
        results["ref_expr_inds"] = ref_expr_inds

    def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = XLMRobertaTokenizer(self.tokenizer_path)
        return self.tokenizer

    def _token_similarity(self, a, b):
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


@PIPELINES.register_module()
class RandomHardNegative:
    def __init__(self, prob=0.2, text_pool=None):
        self.prob = prob

    def __call__(self, results):
        # 1. 检查这个样本是否原本在 GRefCOCO 数据集里就是 "无目标" (empty)
        is_empty = results.get('empty', False)

        if is_empty:
            # 如果本身就是无目标样本，顺理成章地将存在性设为 0
            results['gt_exist'] = np.array([0], dtype=np.int64)
        else:
            # 2. 如果是有目标的正常样本，按 prob (20%) 概率清空目标，制造“难负样本”
            if random.random() < self.prob:
                # 清空 BBox (注意：这里的键名必须是 gt_bbox)
                results['gt_bbox'] = np.zeros((0, 4), dtype=np.float32)

                # 清空 Mask
                if 'gt_mask_rle' in results:
                    results['gt_mask_rle'] = []
                if 'gt_mask_parts_rle' in results:
                    results['gt_mask_parts_rle'] = []

                # 设置存在性标签为 0
                results['gt_exist'] = np.array([0], dtype=np.int64)

            else:
                # 👉 【核心修复】：如果不触发难负样本，必须赋予它正常的 gt_exist = 1 标签！
                # 这样 CollectData 就永远都能找到 'gt_exist' 键了。
                results['gt_exist'] = np.array([1], dtype=np.int64)

        return results

@PIPELINES.register_module()
class VGTRAugment(object):
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, results):
        img = results["img"]
        phrase = results["expression"]
        bbox = results["gt_bbox"]
        img, phrase, bbox = self.trans(img, phrase, bbox, self.img_size)
        results["img"] = img
        results["phrase"] = phrase
        results["bbox"] = bbox
        return results

    def trans(self, img, phrase, bbox, imsize):

        img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)
        a = (random.random() * 2 - 1) * 0.5 + 1
        # S = S * a
        if a >= 1:
            np.clip(S, a_min=0, a_max=255, out=S)
        a = (random.random() * 2 - 1) * 0.5 + 1
        V = V * a
        if a >= 1:
            np.clip(V, a_min=0, a_max=255, out=V)
        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = colorjitter(img)
        img = gauss(np.array(img))
        img, bbox = reshape(img, bbox, imsize)
        img, _, bbox, M = random_affine(
            img,
            None,
            bbox,
            degrees=(-15, 15),
            translate=(0.15, 0.15),
            scale=(0.75, 1.25),
        )
        if random.random() > 0.5:
            img, phrase, bbox = horizontal_flip(img, phrase, bbox)

        return img, phrase, bbox


def reshape(img, bbox, height):
    shape = img.shape[:2]
    color = (123.7, 116.3, 103.5)
    ratio = float(height) / max(shape)
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded square
    bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
    bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

    return img, bbox


def horizontal_flip(img, phrase, bbox):
    w = img.shape[1]
    img = cv2.flip(img, 1)
    bbox[0], bbox[2] = w - bbox[2] - 1, w - bbox[0] - 1
    phrase = (
        phrase.replace("right", "*&^special^&*")
        .replace("left", "right")
        .replace("*&^special^&*", "left")
    )

    return img, phrase, bbox


def random_affine(
    img,
    mask,
    targets,
    degrees=(-10, 10),
    translate=(0.1, 0.1),
    scale=(0.8, 1.2),
    shear=(-2, 2),
    borderValue=(123.7, 116.3, 103.5),
    all_bbox=None,
):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[
        0
    ] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[
        1
    ] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(
        img, M, dsize=(height, height), flags=cv2.INTER_LINEAR, borderValue=borderValue
    )  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(
            mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST, borderValue=255
        )  # BGR order borderValue
    else:
        maskw = None

    # Return warped points also
    if type(targets) == type([1]):
        targetlist = []
        for bbox in targets:
            targetlist.append(wrap_points(bbox, M, height, a))
        return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii, :] = wrap_points(all_bbox[ii, :], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:  ## previous main
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw


def affine(
    img,
    bbox,
    degrees=(-15, 15),
    translate=(0.15, 0.15),
    scale=(0.75, 1.25),
    shear=(-2, 2),
    borderValue=(123.7, 116.3, 103.5),
):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[
        0
    ] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[
        1
    ] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(
        img, M, dsize=(height, height), flags=cv2.INTER_LINEAR, borderValue=borderValue
    )  # BGR order borderValue

    # Return warped points also
    if type(bbox) == type([1]):
        targetlist = []
        for box in bbox:
            targetlist.append(wrap_points(box, M, height, a))
        return imw, targetlist
    elif bbox is not None:
        targets = wrap_points(bbox, M, height, a)
        return imw, targets
    else:
        return imw


def generate_transM(
    img, degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.75, 1.25), shear=(-2, 2)
):
    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        (random.random() * 2 - 1) * translate[0] * img.shape[0]
    )  # x translation (pixels)
    T[1, 2] = (
        (random.random() * 2 - 1) * translate[1] * img.shape[1]
    )  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!

    return M


def colorjitter(img):
    color_aug = transforms.ColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08
    )
    img = color_aug(img)
    return img


def gauss(img):
    scale = 3
    sigma = 0.3 * ((scale - 1) * 0.5 - 1) + 0.8
    # follow cv2's default routine
    if random.random() > 0.5:
        cv2.GaussianBlur(img, ksize=(scale, scale), sigmaX=sigma, dst=img)

    return img


def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()
    points = targets.copy()
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    ## print(targets, xy)
    ## [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets


def trans(img, phrase, bbox, imsize):

    img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)
    a = (random.random() * 2 - 1) * 0.5 + 1
    # S = S * a
    if a >= 1:
        np.clip(S, a_min=0, a_max=255, out=S)
    a = (random.random() * 2 - 1) * 0.5 + 1
    V = V * a
    if a >= 1:
        np.clip(V, a_min=0, a_max=255, out=V)
    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = colorjitter(img)
    img = gauss(np.array(img))
    img, bbox = reshape(img, bbox, imsize)
    img, _, bbox, M = random_affine(
        img, None, bbox, degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.75, 1.25)
    )
    if random.random() > 0.5:
        img, phrase, bbox = horizontal_flip(img, phrase, bbox)

    return img, phrase, bbox


def trans_simple(img, phrase, bbox, imsize):

    img, bbox = reshape(img, bbox, imsize)
    return img, phrase, bbox


class ResizePad:

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.h, self.w = size

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.h / h, self.w / w)
        resized_h = int(np.round(h * scale))
        resized_w = int(np.round(w * scale))
        pad_h = int(np.floor(self.h - resized_h) / 2)
        pad_w = int(np.floor(self.w - resized_w) / 2)

        resized_img = cv2.resize(img, (resized_w, resized_h))

        # if img.ndim > 2:
        if img.ndim > 2:
            new_img = np.zeros((self.h, self.w, img.shape[-1]), dtype=resized_img.dtype)
        else:
            resized_img = np.expand_dims(resized_img, -1)
            new_img = np.zeros((self.h, self.w, 1), dtype=resized_img.dtype)
        new_img[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w, ...] = resized_img
        return new_img


class CropResize:

    def __call__(self, img, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))
        im_h, im_w = img.data.shape[:2]
        input_h, input_w = size
        scale = max(input_h / im_h, input_w / im_w)
        # scale = torch.Tensor([[input_h / im_h, input_w / im_w]]).max()
        resized_h = int(np.round(im_h * scale))
        # resized_h = torch.round(im_h * scale)
        resized_w = int(np.round(im_w * scale))
        # resized_w = torch.round(im_w * scale)
        crop_h = int(np.floor(resized_h - input_h) / 2)
        # crop_h = torch.floor(resized_h - input_h) // 2
        crop_w = int(np.floor(resized_w - input_w) / 2)
        # crop_w = torch.floor(resized_w - input_w) // 2
        # resized_img = cv2.resize(img, (resized_w, resized_h))
        resized_img = F.upsample(
            img.unsqueeze(0).unsqueeze(0), size=(resized_h, resized_w), mode="bilinear"
        )

        resized_img = resized_img.squeeze().unsqueeze(0)

        return resized_img[0, crop_h : crop_h + input_h, crop_w : crop_w + input_w]


class ToNumpy:

    def __call__(self, x):
        return x.numpy()


class ResizeImage:
    """Resize the largest of the sides of the image to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = (
            F.upsample(
                Variable(img).unsqueeze(0), size=(resized_h, resized_w), mode="bilinear"
            )
            .squeeze()
            .data
        )
        return out


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale = min(self.size / im_h, self.size / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        out = (
            F.upsample(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
            )
            .squeeze()
            .data
        )
        return out
