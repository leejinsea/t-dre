import os
import warnings
import pathlib
import copy
import pickle

import numpy as np

import torch.nn as nn
import torch
import torch.cuda.amp as amp
from collections import OrderedDict
from torch.cuda.amp.grad_scaler import GradScaler
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from models.faster_rcnn import faster_rcnn_resnet50_fpn

from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import VGDataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from language_model import BoxDescriber


class ActionDescriber(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size,
                                                  image_mean,
                                                  image_std)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7, sampling_ratio=2)

        self.describer = BoxDescriber(feat_size=4096,
                                      hidden_size=512,
                                      max_len=16,
                                      emb_size=512,
                                      rnn_num_layers=1,
                                      vocab_size=1791,
                                      fusion_type='init_inject')

        self.box_head = TwoMLPHead(256 * 7 ** 2, 4096)

    def forward(self, proposals, images_list, targets):

        images, targets = self.transform(images_list, targets)
        features = backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        reg_features = self.box_roi_pool(features, proposals, images.image_sizes)
        features = self.box_head(reg_features)

        gt_captions = [t["caps"] for t in targets]
        gt_captions_length = [t["caps_len"] for t in targets]

        caption_predicts = self.describer(features, gt_captions, gt_captions_length)

        if self.training:
            loss_caption = caption_loss(caption_predicts, gt_captions, gt_captions_length)
            return loss_caption
        else:
            return caption_predicts


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


def caption_loss(caption_predicts, caption_gt, caption_length):
    """
    Computes the loss for caption part.
    Arguments:
        caption_predicts (Tensor)
        caption_gt (Tensor or list[Tensor])
        caption_length (Tensor or list[Tensor])
        caption_loss (Tensor)
    """

    if isinstance(caption_gt, list) and isinstance(caption_length, list):
        caption_gt = torch.cat(caption_gt, dim=0)  # (batch_size, max_len+1)
        caption_length = torch.cat(caption_length, dim=0) # (batch_size, )
        assert caption_predicts.shape[0] == caption_gt.shape[0] and caption_predicts.shape[0] == caption_length.shape[0]

    # '<bos>' is not considered
    caption_length = torch.clamp(caption_length-1, min=0)

    predict_pps = pack_padded_sequence(caption_predicts, caption_length.to('cpu'), batch_first=True, enforce_sorted=False)

    target_pps = pack_padded_sequence(caption_gt[:, 1:], caption_length.to('cpu'), batch_first=True, enforce_sorted=False)

    return F.cross_entropy(predict_pps.data, target_pps.data)


def t_dre(predicted, score_thresh=0.5):
    '''
    person이 포함되면 1, 아니면 0
    '''

    all_list = []
    out_bboxes = []

    keep = torch.where(predicted['scores'] >= score_thresh)

    if len(keep[0]) == 0:
        return [predicted['boxes'][0].unsqueeze(0)]

    predicted['boxes'] = predicted['boxes'][keep]
    predicted['labels'] = predicted['labels'][keep]
    predicted['scores'] = predicted['scores'][keep]

    boxes = predicted['boxes']

    labels = predicted['labels']
    persons_boxes = list(torch.split(boxes[torch.where(labels == 1)[0]], 1))
    no_persons_boxes = list(torch.split(boxes[torch.where(labels != 1)[0]], 1))

    if (len(no_persons_boxes) == 1 and len(no_persons_boxes[0]) == 0):
        return persons_boxes

    if (len(persons_boxes) == 1 and len(persons_boxes[0]) == 0):
        return no_persons_boxes


    while len(persons_boxes) != 0:
        target_box = persons_boxes.pop(0)

        # 재귀 A
        list_idx = list()
        for i in range(0, len(no_persons_boxes)):
            if not bbox_iou(target_box, no_persons_boxes[i]) == 0:
                list_idx.append(i)
        list_idx.sort(reverse=True)
        # 재귀 A 끝

        tmp_boxes = list()
        # 재귀 B
        for i in list_idx:
            tmp_boxes.append(no_persons_boxes.pop(i))
        # 재귀 B 끝
        tmp_boxes.append(target_box)
        tmp_boxes = tmp_boxes[::-1]

        idx = 1
        while len(tmp_boxes) > idx:
            target_box = tmp_boxes[idx]
            # 재귀 A
            list_idx = list()
            for i in range(0, len(no_persons_boxes)):
                if not bbox_iou(target_box, no_persons_boxes[i]) == 0:
                    list_idx.append(i)
            list_idx.sort(reverse=True)
            # 재귀 A 끝
            # 재귀 B
            for i in list_idx:
                tmp_boxes.append(no_persons_boxes.pop(i))
            # 재귀 B 끝
            idx += 1
        all_list.append(tmp_boxes)

    for bboxes in all_list:
        out_bboxes.append(torch.cat(
            [torch.min(torch.cat(bboxes)[:, :2], dim=0)[0],
             torch.max(torch.cat(bboxes)[:, 2:], dim=0)[0]]))

    return out_bboxes


def _bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    if len(box1.size()) == 2:
        box1 = box1.squeeze(0)
    if len(box2.size()) == 2:
        box2 = box2.squeeze(0)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0) #torch.clamp(x,min,max)에서 min~max로 범위지정함

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0][0], box1[0][1], box1[0][2], box1[0][3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0][0], box2[0][1], box2[0][2], box2[0][3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0) #torch.clamp(x,min,max)에서 min~max로 범위지정함

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def assign_targets(_preds, trg_cap_list):

    dre_boxes = []
    for i, pred in enumerate(_preds):

        _dre_boxes = t_dre(pred)
        _dre_boxes = torch.stack(_dre_boxes)

        if len(_dre_boxes.size()) == 3:
            if _dre_boxes.size(1) == 1:
                _dre_boxes = _dre_boxes.squeeze(1)
            else:
                _dre_boxes = _dre_boxes.squeeze(0)

        dre_boxes.append(_dre_boxes)

    targets = []
    for dre_box, trgs in zip(dre_boxes, trg_cap_list):

        new_trg = dict()
        _caps = []
        _caps_len = []
        _boxes = []

        for box in dre_box:
            best_iou = 0
            best_idx = 0
            for i, trg_box in enumerate(trgs['boxes']):
                iou = _bbox_iou(box, trg_box)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            _boxes.append(trgs['boxes'][best_idx])
            _caps.append(trgs['caps'][best_idx])
            _caps_len.append(trgs['caps_len'][best_idx])


        new_trg['boxes'] = torch.stack(_boxes)
        new_trg['caps'] = torch.stack(_caps)
        new_trg['caps_len'] = torch.stack(_caps_len)

        targets.append(new_trg)

    assert len(dre_boxes) == len(targets)
    return dre_boxes, targets



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CUDA_LAUNCH_BLOCKING = 1

IMG_DIR_ROOT = pathlib.Path("C:/datasets/visual_genome")

VG_DATA_PATH = './VG-regions-lite_231130.h5'
LOOK_UP_TABLES_PATH = './regions_data_231130.pkl'

STRAT_EPOCHS = 1
END_EPOCHS = 1
BATCH_SIZE = 1

SHUFFLE = True
USE_AMP = False

coco_labels_list = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


test_set = VGDataset(str(IMG_DIR_ROOT), str(VG_DATA_PATH), str(LOOK_UP_TABLES_PATH), dataset_type='test')
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=BATCH_SIZE,
                                          shuffle=SHUFFLE,
                                          collate_fn=VGDataset.collate_fn,
                                          pin_memory=True,
                                          num_workers=0)


detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
        )


detector.to(device)
detector.eval()

backbone = detector.backbone
action_describer = ActionDescriber(backbone=backbone).to(device)
action_describer.load_state_dict(torch.load('./model_2023_1201_1.pth'))
action_describer.eval()

# action_describer.train()
#
# backbone_params = list(action_describer.backbone.parameters())
# for bb_param in backbone_params:
#     bb_param.requires_grad = False

# params = filter(lambda p: p.requires_grad, action_describer.parameters())
# optimizer = Adam(params=params, lr=1e-4)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


epoch_histories = []
for epoch in range(STRAT_EPOCHS, END_EPOCHS):
    history = []
    image_set = []
    trg_cap_set = []
    pred_cap_set = []
    epoch_loss = 0
    for itr, (_, imgs, trgs_r, trg_caps) in enumerate(test_loader):

        start.record()
        # optimizer.zero_grad()
        img_list = [img.to(device) for img in imgs]
        trg_cap_list = [{k: v.to(device) for k, v in trg.items()} for trg in trg_caps]
        preds = detector(img_list)

        new_preds = []
        new_images = []
        new_targets = []
        for img, pred, trg in zip(img_list, preds, trg_cap_list):
            if pred['labels'].numel():
                new_images.append(img)
                new_preds.append(pred)
                new_targets.append(trg)

        # model
        proposals, targets = assign_targets(new_preds, new_targets)

        # del preds, trg_cap_list, img_list

        cap_pred = action_describer(proposals, new_images, targets)
        image_set.append(img_list)
        trg_cap_set.append(trg_cap_list)
        pred_cap_set.append(cap_pred)
        # history.append(cap_pred.item())
        # loss.backward()
        # optimizer.step()

        torch.cuda.empty_cache()

        end.record()
        torch.cuda.synchronize()

        print(f'epc: 1/1 | iter: {itr + 1}/{len(test_loader)} | time: {start.elapsed_time(end) / 1000:.2f}')
        # epoch_histories.append(sum(history) / len(history)) #1: 3.08

    torch.save(action_describer.state_dict(), f"./model_2023_1201_{epoch + 1}.pth")

# finally
# image_set is test image list  .append(img_list)
# trg_cap_set is image.append(trg_cap_list)
# pred_cap_set.append(cap_pred)
#

from nltk.translate.bleu_score import sentence_bleu


with open('./idx_to_token.pickle', 'rb') as fr:
    idx_to_token = pickle.load(fr)

# BLEU-1 SCORE
cnt = 0
score = 0
for target, predictions in zip(trg_cap_set, pred_cap_set):
    reference = []
    for cap in target[0]['caps']:
        reference.append([idx_to_token[idx.item()] for idx in cap])
    for prediction in predictions:
        candidate = [idx_to_token[idx.item()] for idx in prediction]
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score += score1
        cnt += 1
        print(f'No.{cnt} | BLEU SCORE : {score1} | Means : {score / cnt}')


# BLEU-1 SCORE, BLEU-2 SCORE, BLEU-3 SCORE, BLEU-4 SCORE, Meteor SCORE
all_sentence_cnt = 0
for pred_s in pred_cap_set:
    for pred in pred_s:
        all_sentence_cnt += 1

cnt = 0
list_score1 = []
list_score2 = []
list_score3 = []
list_score4 = []
list_meteor_score = []
for target, predictions in zip(trg_cap_set, pred_cap_set):
    reference = []
    for cap in target[0]['caps']:
        reference.append([idx_to_token[idx.item()] for idx in cap])
    for prediction in predictions:
        now_time = time.time()
        candidate = [idx_to_token[idx.item()] for idx in prediction]
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
        score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
        score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
        meteor = meteor_score(reference, candidate)
        list_score1.append(score1)
        list_score2.append(score2)
        list_score3.append(score3)
        list_score4.append(score4)
        list_meteor_score.append(meteor)
        cnt += 1

        print(f'Count {cnt} / {all_sentence_cnt} | BLEU-1 SCORE : {score1:.3f} | BLEU-2 SCORE : {score2:.3f} | '
              f'BLEU-3 SCORE : {score3:.3f} | BLEU-4 SCORE : {score4:.3f} | Meteor SCORE : {meteor:.3f} | '
              f'time : {time.time() - now_time}:.2f')

array_score1 = np.array(list_score1)
array_score2 = np.array(list_score2)
array_score3 = np.array(list_score3)
array_score4 = np.array(list_score4)
array_meteor_score = np.array(list_meteor_score)

np.savetxt('score1.csv', array_score1, delimiter=",")
np.savetxt('score2.csv', array_score2, delimiter=",")
np.savetxt('score2.csv', array_score3, delimiter=",")
np.savetxt('score2.csv', array_score4, delimiter=",")
np.savetxt('meteor_score.csv', array_meteor_score, delimiter=",")

