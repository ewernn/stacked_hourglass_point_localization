import os
import torch
import tqdm
import numpy as np
from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from PIL import Image

import cv2
import matplotlib.pyplot as plt


def do_inference(img_tensor, model):
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        preds = model(img_tensor)
    return preds

def draw_cross(img, center, color, size=1):
    x, y = center
    cv2.line(img, (x - size, y-size), (x + size, y+size), color, 1)
    cv2.line(img, (x+size, y - size), (x-size, y + size), color, 1)

def draw_predictions(img_tensor, pred_keypoints, true_points, config, save_path=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    nstack = pred_keypoints.shape[0]
    oup_dim = pred_keypoints.shape[1]
    
    im_sz = config['inference']['inp_dim']
    output_res = config['train']['output_res']
    
    for stack in range(nstack):
        if stack != nstack-1: continue
        for k in range(oup_dim):
            x, y, conf = pred_keypoints[stack, k]
            x, y = int(x * im_sz / output_res), int(y * im_sz / output_res)
            print(f"Predicted: {x}, {y}")
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"P{k}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    true_points = true_points.squeeze(0)
    for k, point in enumerate(true_points):
        x, y = point
        x, y = int(x * im_sz), int(y * im_sz)
        print(f"Actual: {x}, {y}")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"T{k}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    return img

def extract_keypoints_from_heatmaps(config, heatmaps):
    nstack = config['inference']['nstack']
    oup_dim = config['inference']['oup_dim']
    output_res = config['train']['output_res']

    heatmaps = heatmaps.view(nstack, oup_dim, -1)
    maxval, idx = torch.max(heatmaps, dim=2)

    x = (idx % output_res).view(nstack, oup_dim, 1)
    y = (idx // output_res).view(nstack, oup_dim, 1)
    maxval = maxval.view(nstack, oup_dim, 1)

    keypoints = torch.cat((x, y, maxval), dim=2)
    return keypoints


def main():
    from train import init, reload, parse_command_line
    
    opt = parse_command_line()
    task, config = init(opt)
    train_func = task.make_network(config)

    pretrained_model_path = '/content/drive/MyDrive/MM/EqNeck/EqNeck3pts/exps2/eq_neck_1000/checkpoint_0.0002_4.pt'
    if config['opt']['pretrained_model'] is not None:
        pretrained_model_path = config['opt']['pretrained_model']
    if os.path.isfile(pretrained_model_path):
        print("=> loading pretrained model '{}'".format(pretrained_model_path))
        checkpoint = torch.load(pretrained_model_path)
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
        config['inference']['net'].load_state_dict(state_dict, strict=False)

    test_dir = '/content/drive/MyDrive/MM/EqNeck/EqNeck3pts/Test'
    
    im_sz = config['inference']['inp_dim']
    heatmap_res = config['train']['output_res']
    test_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, testing=True,\
                        output_res=heatmap_res, augment=False, only10=config['opt']['only10'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = config['inference']['net']
    model.eval()

    for i, (img_tensor, true_points) in enumerate(test_loader):
        preds = do_inference(img_tensor, model)
        pred_keypoints = extract_keypoints_from_heatmaps(config, preds)

        im_sz = config['inference']['inp_dim']
        output_res = config['train']['output_res']
        
        pred_keypoints_scaled = pred_keypoints.clone().cpu()[:, :, :2] * im_sz / output_res

        print(f"Image {i}:")
        print("Predicted points:")
        print(pred_keypoints_scaled[-1])
        print("Actual points:")
        print(true_points[0] * im_sz)

        mse = torch.mean((pred_keypoints_scaled[-1] - true_points[0] * im_sz) ** 2).item()

        save_dir = '/content/drive/MyDrive/MM/EqNeck/EqNeck3pts/exps2/eq_neck_1000/'
        save_path = os.path.join(save_dir, f'img_{i}.png')
        
        draw_predictions(img_tensor[0], pred_keypoints, true_points, config, save_path=save_path)

        print(f"MSE for image {save_path}: {mse}")
        print()


if __name__ == '__main__':
    main()