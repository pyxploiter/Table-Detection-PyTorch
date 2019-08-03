import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from parser import params

def frcnn_resnet50_fpn(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=params['pretrained'])
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    # This block contains code for freezing layers
    if params['freeze_first_block']:
        d = 0
        for n, block in model.backbone.named_children():
            for name,param in block.named_parameters():
                    if( d < 11 ):
                        param.requires_grad = False
                        # print(n,"\t", name, param.requires_grad)
                    d+=1
    return model

def frcnn_resnet50(num_classes, **kwargs):
    resnet50 = torchvision.models.resnet50(pretrained=params['pretrained'])
    backbone = nn.Sequential(*list(resnet50.children())[:-2])
    backbone.out_channels = 2048

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes, 
                        min_size=600, max_size=1024, 
                        rpn_pre_nms_top_n_train=12000,
                        rpn_pre_nms_top_n_test=12000,
                        rpn_post_nms_top_n_train=2000,
                        rpn_post_nms_top_n_test=2000,
                        rpn_anchor_generator=rpn_anchor_generator,
                        box_roi_pool=roi_pooler,
                        **kwargs)

     # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    ######## This block contains code for freezing layers #########
    if params['freeze_first_block']:
        d = 0
        for n, block in model.backbone.named_children():
            for name,param in block.named_parameters():
                    if( d < 15 ):
                        param.requires_grad = False
                        # print(n,"\t", name, param.requires_grad)
                    d+=1
    ###############################################################

    return model

def frcnn_resnet101_fpn(num_classes, **kwargs):
    backbone = resnet_fpn_backbone('resnet101', params['pretrained'])
    model = FasterRCNN(backbone, num_classes, **kwargs)

     # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    ######## This block contains code for freezing layers #########
    if params['freeze_first_block']:
        d = 0
        for n, block in model.backbone.named_children():
            for name,param in block.named_parameters():
                    if( d < 11 ):
                        param.requires_grad = False
                        # print(n,"\t", name, param.requires_grad)
                    d+=1
    ###############################################################
    return model

def frcnn_resnet101(num_classes, **kwargs):
    resnet101 = torchvision.models.resnet101(pretrained=params['pretrained'])
    backbone = nn.Sequential(*list(resnet101.children())[:-2])
    backbone.out_channels = 2048

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes, 
                        min_size=600, max_size=1024, 
                        rpn_pre_nms_top_n_train=12000,
                        rpn_pre_nms_top_n_test=12000,
                        rpn_post_nms_top_n_train=2000,
                        rpn_post_nms_top_n_test=2000,
                        rpn_anchor_generator=rpn_anchor_generator,
                        box_roi_pool=roi_pooler,
                        **kwargs)

     # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    ######## This block contains code for freezing layers #########
    if params['freeze_first_block']:
        d = 0
        for n, block in model.backbone.named_children():
            for name,param in block.named_parameters():
                    if( d < 15 ):
                        param.requires_grad = False
                        # print(n,"\t", name, param.requires_grad)
                    d+=1
    ###############################################################

    return model