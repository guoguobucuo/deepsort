import numpy as np
import torch
import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead, Grade4RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16
import torchvision.transforms as transforms

class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            self.grade4classifiers = nn.ModuleList([decom_vgg16(pretrained)[1] for _ in range(17)])
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 7,
                spatial_scale   = 1,
                classifier      = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            _, self.grade4classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )

            # self.grade4head = Grade4RoIHead(
            #         # n_class         = 4,
            #         roi_size=14,
            #         spatial_scale=1,
            #         classifier=self.grade4classifier
            # )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小
            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores, roi_grade_scores= self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices, roi_grade_scores
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores, roi_grade_scores  = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, roi_grade_scores
        # elif mode == 'grade4head':
        #     base_feature, rois, roi_indices, img_size , roi_cls_locs, roi_scores= x
        #
        #     # ---------------------------------------#
        #     #   获得classifier的分类结果和回归结果
        #     # ---------------------------------------#
        #     grade4_roi_scores = self.grade4head.forward(base_feature, rois, roi_indices, img_size, roi_cls_locs, roi_scores)
        #     return grade4_roi_scores
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class Fasterrcnn_with_grade_head(FasterRCNN):
    def forward(self, x, scale=1., mode="forward"):
        if mode=="forward":
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)

            grade4_roi_scores = self.grade4head.forward(base_feature, rois, roi_indices, img_size, roi_cls_locs,
                                                        roi_scores)

            return roi_cls_locs, roi_scores, rois, roi_indices, grade4_roi_scores
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores
        if mode == 'fasterrcnn':
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        if mode == 'grade4head':
            base_feature, rois, roi_indices, img_size, roi_cls_locs, roi_scores = x

            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            grade4_roi_scores = self.grade4head.forward(base_feature, rois, roi_indices, img_size, roi_cls_locs,
                                                        roi_scores)
            return grade4_roi_scores

if __name__ == '__main__':
    model =  FasterRCNN(16, "predict", anchor_scales = [8, 16, 32], backbone = "resnet50")
    input = torch.randn(4,3,920,920)
    roi_cls_locs, roi_scores, rois, roi_indices, grade_scores= model(input)

    print(roi_cls_locs.shape)
    print(roi_scores.shape)
    print(rois.shape)
    print(roi_indices.shape)
    print(grade_scores.shape)

    # grade = torch.argmax(grade_scores,-1)
    # print(grade.shape)
    # torch.Size([1, 300, 68])
    # torch.Size([1, 300, 17])
    # torch.Size([1, 300, 4])
    # torch.Size([1, 300])
    # torch.Size([1, 300, 5])

    # torch.onnx.export(model, input, './onnx_model_t.onnx')
    # import netron
    # netron.start(r'D:\code\faster-rcnn-pytorch-master\nets\onnx_model_t.onnx')