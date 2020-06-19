import pandas as pd
import numpy as np

def evaluate(truth_annot, ans_annot):
    iou_thresh = 0.5
    ans_annot = pd.DataFrame(ans_annot)
    truth_groups = truth_annot.groupby(['image_id','category_id'])
    num_positives = truth_annot.index.stop # Number of positive annotations expected
    num_detections = ans_annot.index.stop # Number of detections

    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    
    counter = 0
    for idx,row in ans_annot.iterrows(): # Each detection/annotation
        # TP = False
        
        scores = np.append(scores, row['score'])
        try:
            truth_group = truth_groups.get_group((row['image_id']-1,row['category_id']+1)) # +- cos rohittttttt
        except:
            false_positives = np.append(false_positives, 1)
            true_positives  = np.append(true_positives, 0)
            continue
        else:   
            max_iou = 0
            max_iou_truth  = pd.DataFrame()
            for truth_idx, truth_row in truth_group.iterrows():
                iou = calc_iou(row['bbox'],truth_row['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_truth = truth_row
            
            
            if max_iou >= 0.5 and max_iou_truth['detected'] == False: # Detection is a prediction
                TP = True
                counter += 1
                max_iou_truth['detected'] = True
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
    # sorts by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    recall    = true_positives / num_positives
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    # print(recall[1145])
    # return calc_MAP(precision,recall)
    return recall, precision


def calc_MAP(precision, recall):
    cumulatives = []
    local_max = precision[0]
    length = 1
    for val in precision:
        if val < local_max: #drop detected
            cumulatives.append(length*local_max)
            local_max = val
            length = 1
        else:
            local_max = val
            length +=1
    return sum(cumulatives)/(precision.size+1)

    

def calc_iou(ans_bbox,val_bbox):
    ans_bbox = to_xyxy(ans_bbox)
    val_bbox = to_xyxy(val_bbox)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(ans_bbox[0], val_bbox[0])
    yA = max(ans_bbox[1], val_bbox[1])
    xB = min(ans_bbox[2], val_bbox[2])
    yB = min(ans_bbox[3], val_bbox[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    ans_boxArea = (ans_bbox[2] - ans_bbox[0] + 1) * (ans_bbox[3] - ans_bbox[1] + 1)
    val_boxArea = (val_bbox[2] - val_bbox[0] + 1) * (val_bbox[3] - val_bbox[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(ans_boxArea + val_boxArea - interArea)
	# return the intersection over union value
    return iou


def to_xyxy(bbox):
    bbox[2] += bbox[0] 
    bbox[3] += bbox[1]
    return bbox