import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool
from utils.utils_bbox import loc2bbox

from nets.resnet50 import resnet50

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        # print(pool.size())
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)

        grade_num = [4, 4, 3, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 3]
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        out_features = sum(grade_num)
        self.grade4score = nn.Linear(2048, out_features + 1)

        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        # 把第一维度去掉 1，600，4 -》 600，4
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # roi_indices[:,None] size=1,600,1
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_grade_scores = self.grade4score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        roi_grade_scores = roi_grade_scores.view(n, -1, roi_grade_scores.size(1))
        return roi_cls_locs, roi_scores, roi_grade_scores

class Grade4RoIHead(nn.Module):
    def __init__(self, roi_size, spatial_scale, classifier):
        super(Grade4RoIHead, self).__init__()
        grade_num = [4, 4, 3, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 3]
        self.classifier = classifier
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        out_features = sum(grade_num)
        self.score = nn.Linear(2048, out_features)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size, roi_cls_locs, roi_scores):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()

        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        # 去除第一维度
        labels = torch.flatten(roi_scores, 0, 1)
        rois_locs = torch.flatten(roi_cls_locs, 0, 1)

        # 将分类器算出来的偏移，加到原roi上
        labels = torch.argmax(labels, dim=1)
        # labels += 1
        indices = torch.stack([labels*4,labels*4+1,labels*4+2,labels*4+3], dim=-1)
        rois_locs = torch.gather(rois_locs, dim=1, index=indices)

        rois = loc2bbox(rois, rois_locs)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #600,c,roi_size,roi_size
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        # fc7 = self.classifiers[](pool)
        fc7 = self.classifier(pool)
        fc7 = torch.cat(fc7,dim=0)

        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # --------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_scores = self.score(fc7)
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


if __name__ == '__main__':
    extractor, classifier = resnet50(False)
    extractor2, classifier2 = resnet50(False)

    roihead = Resnet50RoIHead(17,14,1,classifier)
    gradehead = Grade4RoIHead(14,1,classifier2)

    input = torch.randn(1, 1024, 60, 60)
    rois = torch.randn(1,600,4)
    roi_indices = torch.randn(1,600)
    roi_cls_locs, roi_scores = roihead.forward(input, rois, roi_indices, (960,960))

    grade_score = gradehead.forward(input,rois,roi_indices,(960,960),roi_cls_locs, roi_scores)

    print(roi_cls_locs.shape) #1,600,68
    print(roi_scores.shape) #1,600,17
    print(grade_score.shape) #1,600,4