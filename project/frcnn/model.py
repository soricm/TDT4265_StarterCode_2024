import torchvision


def get_model(device='cpu', model_name='v2'):
    if model_name == 'v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT'
        )
    elif model_name == 'v1':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT'
        )

    # load the model
    model = model.eval().to(device)
    return model
