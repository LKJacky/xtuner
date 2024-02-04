import json
import re

answers = json.load(open("answers.json"))


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


right = 0
for ans in answers:
    bbox = ans["bbox"]
    ans = [int(x) for x in re.findall("\d+", ans["ans"])]
    iou = computeIoU(ans, bbox)
    print(bbox, ans)
    if iou > 0.5:
        right += 1
print(right / len(answers))