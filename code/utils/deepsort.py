import os
import numpy as np
import cv2
from deep_sort.deep_sort import DeepSort

# 假设你的检测结果CSV文件路径（离线检测后的结果）
detection_file = "code/model_data/hdpe_grade.pth"

# 读取检测结果：frame_id, x1, y1, x2, y2, score, class_id
dets = np.loadtxt(detection_file, delimiter=',', dtype=float)

# 根据检测结果获取帧范围
all_frames = dets[:, 0].astype(int)
min_frame = all_frames.min()
max_frame = all_frames.max()

# 初始化Deep SORT
# model_path为Deep SORT的外观特征提取模型权重文件
model_path = "code/model_data/hdpe_grade.pth"
deepsort = DeepSort(model_path,
                    max_dist=0.2,
                    max_iou_distance=0.7,
                    max_age=30,
                    n_init=3,
                    nn_budget=100,
                    use_cuda=True)

# 假设处理后的帧储存在processed_frames目录下，
# 文件命名例如：frame_000001.jpg, frame_000002.jpg,...
processed_frame_dir = "processed_frames"

results = []

for frame_id in range(min_frame, max_frame + 1):
    # 获取当前帧的所有检测框
    frame_dets = dets[dets[:, 0] == frame_id]

    if frame_dets.size == 0:
        # 当前帧没有检测结果
        bboxes_xyxy = np.empty((0, 4))
        confs = np.empty((0))
        clss = np.empty((0))
    else:
        # 有检测框
        bboxes_xyxy = frame_dets[:, 1:5]  # [x1,y1,x2,y2]
        confs = frame_dets[:, 5]
        if frame_dets.shape[1] > 6:
            clss = frame_dets[:, 6]
        else:
            clss = np.zeros((bboxes_xyxy.shape[0],))

    # 加载对应的处理后帧图像，用于Deep SORT提取外观特征
    img_path = os.path.join(processed_frame_dir, f"frame_{frame_id:06d}.jpg")
    if os.path.exists(img_path):
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        # 若没有对应图像，也可传None，此时Deep SORT可能只能基于IOU进行匹配，
        # 但这样外观特征将无法使用，会影响跟踪效果。
        im = None

    # Deep SORT需要xywh格式的检测框
    bboxes_xywh = np.copy(bboxes_xyxy)
    bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]  # w = x2 - x1
    bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]  # h = y2 - y1
    bboxes_xywh[:, 0] = bboxes_xyxy[:, 0] + bboxes_xywh[:, 2] / 2.0  # cx = x1 + w/2
    bboxes_xywh[:, 1] = bboxes_xyxy[:, 1] + bboxes_xywh[:, 3] / 2.0  # cy = y1 + h/2

    # 调用deepsort进行更新
    # outputs格式为: [x1,y1,x2,y2,track_id,class_id]
    outputs = deepsort.update(bboxes_xywh, confs, clss, im)

    # 将输出结果存储
    for t in outputs:
        x1, y1, x2, y2, track_id, c_id = t
        w = x2 - x1
        h = y2 - y1
        # 保存格式：frame_id, track_id, x1, y1, w, h, class_id
        # 你可以根据自己的需求决定输出格式
        results.append([frame_id, track_id, x1, y1, w, h, c_id])

# 所有帧处理完毕后，将结果写入文件
out_file = "path/to/tracking_result.txt"
with open(out_file, 'w') as f:
    for r in results:
        f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,%d\n" % tuple(r))

print("离线跟踪完成，结果已保存至:", out_file)
