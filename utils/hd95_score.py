from hausdorff import hausdorff_distance

def hd95_score(pred, label):
    hd95_total = 0
    conut = 0
    B, C, S, H, W  = pred.shape
    for b in range(B):
        for c in range(C):
            for s in range(S):
                pred_single = pred[b][c][s].cpu().numpy()
                label_single = label[b][c][s].cpu().numpy()
                pred_single[pred_single >= 0.5] = 1
                pred_single[pred_single < 0.5] = 0
                label_single[label_single >= 0.5] = 1
                label_single[label_single < 0.5] = 0

                hd95_single = hausdorff_distance(pred_single, label_single)

                if hd95_single != 0.0:
                    hd95_total = hd95_total + hd95_single
                    conut = conut + 1
    hd95 = hd95_total / conut
    return hd95