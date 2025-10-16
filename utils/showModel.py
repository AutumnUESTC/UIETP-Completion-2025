from nets.pspnet import PSPNet

num_classes     = 6
backbone        = "resnet50"
downsample_factor   = 8
HL_Train            = True
pretrained      = False
aux_branch      = True
model = PSPNet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, HL_Train = HL_Train, pretrained=pretrained, aux_branch=aux_branch)

print(model)