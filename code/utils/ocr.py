# 对于Windows用户，可以安装Microsoft Visual C++ Bu

import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from paddleocr import PaddleOCR

PATH = './inference'
keywords = ['任务名称',  '检测地点', '检测日期','管道长度', '检测方向', '管道类型', '速度', '管材', '管道材质', '管径', '沙井深度', '检测单位', '检测公司', '检测人员', '距离', r'//', '坡度']
material = {'1': ['混凝土', '钢筋混泥土', '砼'],
            '2': ['金属'],
            '3': ['波纹', 'HDPE', 'PE', 'PVC', 'PUC', '增强缠绕']}

def split_string_by_keywords(input_string, keywords):
    dic_keywords = {}
    for keyword in keywords:
        i = input_string.find(keyword)
        if i == -1:
            continue
        else:
            dic_keywords[i] = keyword
    _keywords = [dic_keywords[key] for key in sorted(dic_keywords)]

    # print(_keywords)

    pattern = '|'.join(map(re.escape, _keywords))
    segments = re.split(pattern, input_string)

    print(segments)

    result = {_keywords[i]: segments[i+1].strip('：；') for i in range(len(_keywords)) if i < len(segments) - 1}
    try:
        distance = result['距离']
        distance = re.findall(r'\b\d+\.\d+\b', distance)
        distance = float(distance[0])
    except:
        distance = 0

    return result, distance

def ocr_detect(model_path, input):
    img_path = input
    result = model_path.ocr(img_path, cls=True)

    r_s = ''
    for s in result[-1]:
        print(s[1][0])
        r_s += s[1][0]
    # print(r_s)

    ret_material = ''
    for key, value in material.items():
        for m in value:
            if m in r_s:
                ret_material = key
    # print('material:' + ret_material)

    return r_s, ret_material
# print(result[-1][0][1])

def do_ocr(model, input):
    r_s, ret_material = ocr_detect(model, input)
    result, distance = split_string_by_keywords(r_s, keywords)
    return result, distance, ret_material

if __name__ == "__main__":
    model_path = PATH
    # 很多文章没有 lang="ch" 后面的代码，导致不断报错，无法运行！
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", det_model_dir=os.path.join(model_path, 'det'),
                    rec_model_dir=os.path.join(model_path, 'rec'),
                    cls_model_dir=os.path.join(model_path, 'cls'))

    result, distance, ret_material = do_ocr(ocr, r'./5719_CK_44.png')



    print(result)
    print(distance)
    print(ret_material)

# 对于Windows用户，可以安装Microsoft Visual C++ Bu