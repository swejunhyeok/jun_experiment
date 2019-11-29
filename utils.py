import csv
import numpy as np
import torch
import random

def mixup(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else :
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def shift_bbox(size, bbox_origin):
    w = size[3]
    h = size[2]
    x_length_origin = bbox_origin[2] - bbox_origin[0]
    y_length_origin = bbox_origin[3] - bbox_origin[1]

    x_coor = random.randrange(w)
    y_coor = random.randrange(h)

    bbx1 = np.clip(x_coor - x_length_origin // 2, 0, w)
    bby1 = np.clip(y_coor - y_length_origin // 2, 0, h)
    bbx2 = np.clip(x_coor + x_length_origin // 2, 0, w)
    bby2 = np.clip(y_coor + y_length_origin // 2, 0, h)

    x_length_shift = bbx2 - bbx1
    y_length_shift = bby2 - bby1

    if x_length_origin != x_length_shift :
        if bbx1 == 0 :
            bbx2 += x_length_origin - x_length_shift
        else :
            bbx1 -= x_length_origin - x_length_shift
    if y_length_origin != y_length_shift :
        if bby1 == 0 :
            bby2 += y_length_origin - y_length_shift
        else :
            bby1 -= y_length_origin - y_length_shift
    
    return bbx1, bby1, bbx2, bby2

def shift_Area(bbox_mix, bbox_shift_origin, bbox_shift):
    bbox_area = (bbox_mix[2] - bbox_mix[0]) * (bbox_mix[3] - bbox_mix[1])

    RB_x1 = max(bbox_mix[0], bbox_shift_origin[0])
    RB_x2 = min(bbox_mix[2], bbox_shift_origin[2])
    RB_y1 = max(bbox_mix[1], bbox_shift_origin[1])
    RB_y2 = min(bbox_mix[3], bbox_shift_origin[3])
    RB_w = RB_x2 - RB_x1
    RB_h = RB_y2 - RB_y1

    GB_x1 = max(bbox_mix[0], bbox_shift[0])
    GB_x2 = min(bbox_mix[2], bbox_shift[2])
    GB_y1 = max(bbox_mix[1], bbox_shift[1])
    GB_y2 = min(bbox_mix[3], bbox_shift[3])
    GB_w = GB_x2 - GB_x1
    GB_h = GB_y2 - GB_y1

    if RB_w > 0 and RB_h > 0 :
        bbox_area += RB_w * RB_h
    if GB_w > 0 and GB_h > 0 :
        bbox_area -= GB_w * GB_h
    return bbox_area

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def save_checkpoint(state, model_path, test_id):
    filename = model_path + test_id +'.pth.tar'
    torch.save(state, filename)