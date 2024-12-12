#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#

import time
import os
import csv
import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    mode = 'video'
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    #video_path      = r'D:\code\fasterrcnn-demo\bugvideo.avi'
    video_path = r'/content/output_video.avi'
    video_save_path = r'sdf.mp4'
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    #fps_image_path  = "img/street.jpg"
    fps_image_path = "/content/gdrive/MyDrive/3aac4a5af8fd58b25dcd9f91ed14b46.png"


    if mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        i = 1

        # 创建保存检测后的图像文件夹
        save_dir = "/content/detected_frames"
        os.makedirs(save_dir, exist_ok=True)

        # 创建并打开CSV文件
        csv_file = "detection_results.csv"
        f = open(csv_file, 'w', newline='')
        writer = csv.writer(f)
        # 根据frcnn.detect_image返回的box信息顺序定义列名
        # 假设box为 [x1, y1, x2, y2, score, class_id, grade, grade_score]
        writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "score", "class_id", "grade", "grade_score"])

        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame, box = frcnn.detect_image(frame)
            frame = np.array(frame)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #cv2.imshow("video",frame)
            # 将检测结果写入CSV文件
            # box中每行为 [x1, y1, x2, y2, score, class_id, grade, grade_score]
            for b in box:
                x1, y1, x2, y2, score, cls_id, grade, grade_score = b
                writer.writerow([i, x1, y1, x2, y2, score, cls_id, grade, grade_score])

            # 保存当前检测后的图像帧
            save_path = os.path.join(save_dir, f"frame{str(i).zfill(6)}.jpg")
            cv2.imwrite(save_path, frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
