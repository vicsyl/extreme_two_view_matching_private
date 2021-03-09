import torch

def upsample_bilinear(depth_data, height, width):
    upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
    depth_data = upsampling(depth_data)
    return depth_data


def upsample_nearest_numpy(data, height, width):
    data = torch.from_numpy(data)
    data = data.view(1, 1, data.shape[0], data.shape[1])
    #data = upsample_bilinear(data, img.shape[0], img.shape[1])
    upsampling = torch.nn.Upsample(size=(height, width), mode='nearest')
    data = upsampling(data)
    return data.squeeze(dim=0).squeeze(dim=0).numpy()
