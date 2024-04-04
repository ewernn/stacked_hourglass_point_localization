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
    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # # Forward pass
    with torch.no_grad():
        preds = model(img_tensor)
    print(f"preds type: {type(preds)}")
    print(f"preds shape: {preds.shape}")
    return preds


##############################################################################################












def draw_heatmap(img, heatmap, alpha=0.6, cmap='viridis'):
    """
    Draws a single heatmap on an image.
    """
    colormap = plt.get_cmap(cmap)
    heatmap_normalized = heatmap / heatmap.max()  # Normalize the heatmap
    heatmap_colored = colormap(heatmap_normalized)[:, :, :3]  # Get only the RGB channels
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)  # Scale to 0-255
    
    # Resize heatmap_colored to match img dimensions
    heatmap_resized = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))
    
    heatmap_img = cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2BGR)
    combined_img = cv2.addWeighted(img, alpha, heatmap_img, 1 - alpha, 0)
    return combined_img


def draw_predictions_with_heatmaps(img_tensor, pred_keypoints, true_points, config, preds, save_path=None):
    # Ensure img_tensor is CPU tensor and in the correct format
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    last_stack_heatmaps = preds[0, -1].cpu().numpy()  # Use last stack; [oup_dim, height, width]

    # Create a figure to visualize heatmaps alongside the original image
    fig, axs = plt.subplots(1, last_stack_heatmaps.shape[0] + 1, figsize=(20, 5))
    
    # Original image with keypoints on the first subplot
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image with Keypoints")
    axs[0].axis('off')

    # Draw predicted and true keypoints on the original image
    scale_factor = config['inference']['inp_dim']
    for k in range(pred_keypoints.shape[1]):
        x, y, conf = pred_keypoints[0, k]  # Assuming batch size of 1
        x, y = int(x * scale_factor), int(y * scale_factor)
        draw_cross(img, (x, y), (0, 0, 255))  # Red color for predicted points
    
    for point in true_points.squeeze(0):
        x, y = point
        x, y = int(x * scale_factor), int(y * scale_factor)
        draw_cross(img, (x, y), (0, 255, 0))  # Green color for true points

    # Draw each heatmap on subsequent subplots
    for i, heatmap in enumerate(last_stack_heatmaps):
        combined_img = draw_heatmap(img.copy(), heatmap)  # Draw heatmap on a copy of the original image
        axs[i+1].imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        axs[i+1].set_title(f"Heatmap {i+1}")
        axs[i+1].axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory











##############################################################################################

import matplotlib.pyplot as plt
import cv2

def draw_cross(img, center, color, size=5):
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)

def draw_predictions(img_tensor, pred_keypoints, true_points, config, save_path=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    nstack = pred_keypoints.shape[0]  # Number of stacks
    oup_dim = pred_keypoints.shape[1]  # Number of keypoints

    # Print shapes and content of pred_keypoints and true_points
    print(f"pred_keypoints shape: {pred_keypoints.shape}")
    print(f"true_points shape: {true_points.shape}")
    print(f"pred_keypoints from last stack: {pred_keypoints[-1]}")
    print(f"true_points: {true_points}")

    scale_factor_pred = config['inference']['inp_dim'] / config['train']['output_res']
    scale_factor_true = config['inference']['inp_dim']

    # Draw predicted keypoints from each stack
    for stack in range(nstack):
        if stack != nstack-1: continue  # Only draw from the last stack
        for k in range(oup_dim):
            x, y, conf = pred_keypoints[stack, k]
            x, y = int(x * scale_factor_pred), int(y * scale_factor_pred)
            draw_cross(img, (x, y), (0, 0, 255))  # Red color for predicted points

    # Draw true points
    true_points = true_points.squeeze(0)  # Remove batch dimension
    for point in true_points:
        x, y = point
        x, y = int(x * scale_factor_true), int(y * scale_factor_true)
        draw_cross(img, (x, y), (0, 255, 0))  # Green color for true points

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off the axis

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

    # Save the heatmaps from stack 3
    if save_path:
        heatmap_save_path = save_path.replace('.png', '_heatmaps_stack3.png')
        # Assuming the heatmap is normalized in [0, 1]
        last_stack_heatmaps = pred_keypoints[-1]  # Last stack heatmaps
        fig, axs = plt.subplots(1, oup_dim, figsize=(20, 5))
        for i, ax in enumerate(axs.flat):
            hm = last_stack_heatmaps[i].cpu().numpy()
            ax.imshow(hm, cmap='hot', interpolation='nearest')
            ax.axis('off')
        plt.savefig(heatmap_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return img


# #print(heatmaps.shape)  # torch.Size([bs, nstack, oup_dim, output_res, output_res])
#                           torch.Size([1, 2, 6, 64, 64])
# def extract_keypoints_from_heatmaps(config, heatmaps):
#     nstack = config['inference']['nstack']
#     oup_dim = config['inference']['oup_dim']  # Number of keypoints
#     output_res = config['train']['output_res']

#     # Flatten the last two dimensions and find the index and max value of the highest point
#     heatmaps = heatmaps.view(nstack, oup_dim, -1)
#     maxval, idx = torch.max(heatmaps, dim=2)

#     # Convert the 1D index to 2D coordinates
#     x = (idx % output_res).view(nstack, oup_dim, 1)
#     y = (idx // output_res).view(nstack, oup_dim, 1)
#     maxval = maxval.view(nstack, oup_dim, 1)

#     # Concatenate the x, y coordinates and max value
#     keypoints = torch.cat((x, y, maxval), dim=2)
#     return keypoints
def extract_keypoints_from_heatmaps(config, heatmaps):
    # No change up to this point
    # Extract the last stack heatmaps
    #print(f"heatmaps shape!: {heatmaps.shape}")
    last_stack_heatmaps = heatmaps[:, -1, :, :, :]
    #print(f"last_stack_heatmaps shape!: {last_stack_heatmaps.shape}")
    
    # Correctly flattening the spatial dimensions
    # Flatten just the last two dimensions (75*75)
    last_stack_heatmaps_flat = last_stack_heatmaps.reshape(last_stack_heatmaps.size(0), last_stack_heatmaps.size(1), -1)
    #print("last_stack_heatmaps shape:", last_stack_heatmaps.shape)
    #print("last_stack_heatmaps_flat shape:", last_stack_heatmaps_flat.shape)

    # Now find the max value and its index for each keypoint
    maxval, idx = torch.max(last_stack_heatmaps_flat, dim=2)
    #print(f"maxval shape: {maxval.shape}")
    
    # Convert indices to 2D coordinates
    x = (idx % heatmaps.size(3)).float() / heatmaps.size(3)  # Normalize X
    y = (idx // heatmaps.size(3)).float() / heatmaps.size(3)  # Normalize Y
    maxval = maxval.unsqueeze(-1)  # Add an extra dimension for concatenation
    
    keypoints = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), maxval), dim=2)  # Concatenate along the last dimension
    
    return keypoints



def main():
    from train import init, reload, parse_command_line
    
    opt = parse_command_line()  # Correctly reference 'train' for the function
    task, config = init(opt)
    train_func = task.make_network(config)

    #pretrained_model_path = '/home/eawern/Eq/stacked_hourglass_point_localization/eq_2e-05_march26.pt'
    pretrained_model_path = 'exps/eq_filtered/checkpoint_1e-05_1.pt'
    if config['opt']['pretrained_model'] is not None:
        pretrained_model_path = config['opt']['pretrained_model']
    if os.path.isfile(pretrained_model_path):  # Correctly check if the pretrained model exists
        print("=> loading pretrained model '{}'".format(pretrained_model_path))
        checkpoint = torch.load(pretrained_model_path)
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
        config['inference']['net'].load_state_dict(state_dict, strict=False)  # Set strict=False if the model architectures are not exactly the same

    test_dir = '/home/eawern/Eq/stacked_hourglass_point_localization/Test/'
    
    im_sz = config['inference']['inp_dim']
    heatmap_res = config['train']['output_res']
    test_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, testing=True,\
                        output_res=heatmap_res, augment=False, only10=config['opt']['only10'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = config['inference']['net']
    model.eval()  # Set the model to evaluation mode

    #for i, (img_tensor, true_points) in enumerate(test_loader):
    print(f"starting test_loader")
    for img_tensor, true_points, invalid_points_bool in test_loader:
        if invalid_points_bool: continue
        #print(f"\ntesting an image with true points: \n{true_points}\n")
        preds = do_inference(img_tensor, model)
        pred_keypoints = extract_keypoints_from_heatmaps(config, preds)

        # scale_down_factor = 1.0/config['train']['output_res']
        # pred_keypoints_scaled = pred_keypoints.clone().cpu()[:, :, :2]
        # pred_keypoints_scaled *= scale_down_factor


        # save_dir = '/home/eawern/Eq/Shoulder_Data/Outputs'
        # #save_path = os.path.join(save_dir, f'img_{i}.png')
        # #save_path = os.path.join(save_dir, f'img.png')
        # save_path = '/home/eawern/Eq/stacked_hourglass_point_localization/exps/eq_exps'
        
        # draw_predictions(img_tensor[0], pred_keypoints, true_points, config, save_path=save_path)
        print(f"abc: {pred_keypoints.shape}, {true_points.shape}")
        print(f"abc: {pred_keypoints[:,:,:2].shape}, {true_points.shape}")
        mse = torch.mean((pred_keypoints.cpu()[:,:,:2] - true_points) ** 2).item()
        print(f"pred_keypoints: {pred_keypoints}, true_points: {true_points}")
        print(f"MSE for image: {mse}\n")

        save_path = '/home/eawern/Eq/stacked_hourglass_point_localization/exps/eq_exps/out.png'
        draw_predictions_with_heatmaps(img_tensor[0], pred_keypoints, true_points, config, preds, save_path=save_path)

if __name__ == '__main__':
    main()
