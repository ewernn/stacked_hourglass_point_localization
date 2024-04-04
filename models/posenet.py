import os
import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss
import matplotlib.pyplot as plt
import numpy as np
import random

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        #self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        #print("Input image shape:", imgs.shape)  # Print the shape of the input images # ERIC
        x = self.pre(imgs)
        combined_hm_preds = []
        imgs_to_display = imgs[0].cpu().detach().numpy()
        imgs_to_display = np.repeat(imgs_to_display[0][np.newaxis, :, :], 3, axis=0)
        # plt.imshow(np.transpose(imgs_to_display, (1, 2, 0)), cmap='gray')
        # plt.savefig(os.path.join('/home/eawern/Eq/stacked_hourglass_point_localization/exps', f'orig_image.png'))
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            #print(f"Shape of heatmap predictions at stack {i}:", preds.shape)  # Print the shape of heatmap predictions at each stack # ERIC
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        combined_hm_preds_stacked = torch.stack(combined_hm_preds, 1)
        #print("Shape of combined heatmap predictions:", combined_hm_preds_stacked.shape)  # Print the shape of combined heatmap predictions # ERIC
        return combined_hm_preds_stacked

    def calc_loss(self, combined_hm_preds, heatmaps):

        def heatmapLoss(pred, gt):
            import matplotlib.pyplot as plt

            # print(f"pred.shape: {pred.shape}")
            # print(f"gt.shape: {gt.shape}")
            #weighted_squared_diff = (pred - gt)**2
            l1_loss = torch.abs(pred - gt).mean() # OH SHIT L1 loss
            # print(f"weighted_squared_diff.shape: {weighted_squared_diff.shape}")
            # print(f"\n\npred.shape: {pred}\n\n")
            # print(f"gt.shape: {gt}")
            # print(f"\n\nweighted_squared_diff.shape: {weighted_squared_diff}")

            # total_loss = weighted_squared_diff.mean(dim=3).mean(dim=2).mean(dim=1)
            total_loss = l1_loss.unsqueeze(0)
            # if random.random() > .99:
            #     randint=random.randint(1,20)
            #     # Save the heatmaps as images
            #     for idx, hm in enumerate([pred[0,0],pred[0,1],pred[0,2],gt[0,0],gt[0,1],gt[0,2]]):
            #         plt.imshow(hm.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            #         plt.savefig(os.path.join('/home/eawern/Eq/stacked_hourglass_point_localization/exps', f'hm_in_loss_{randint}_{idx}.png'))
            #         print(f"total_loss (id:{randint}): {total_loss}")


            # # Function to plot the distribution of weighted squared differences
            # def plot_weighted_squared_diff_distribution(weighted_squared_diff):
            #     plt.figure(figsize=(10, 5))
            #     with torch.no_grad():
            #         plt.hist(weighted_squared_diff.view(-1).cpu().numpy(), bins=100, color='blue', alpha=0.7)
            #     plt.title('Distribution of Weighted Squared Difference')
            #     plt.xlabel('Weighted Squared Difference')
            #     plt.ylabel('Frequency')
            #     plt.grid(True)
            #     plt.savefig('/home/eawern/Eq/stacked_hourglass_point_localization/exps/distribution.png')
            #     plt.close()

            # # Function to visualize the error heatmap
            # def visualize_error_heatmap(weighted_squared_diff):
            #     error_heatmap = weighted_squared_diff.sum(dim=1).squeeze(0)
            #     plt.figure(figsize=(6, 6))
            #     with torch.no_grad():
            #         plt.imshow(error_heatmap.cpu().numpy(), cmap='hot', interpolation='nearest')
            #     plt.title('Error Heatmap')
            #     plt.colorbar()
            #     plt.savefig('/home/eawern/Eq/stacked_hourglass_point_localization/exps/error_heatmap.png')
            #     plt.close()

            # plot_weighted_squared_diff_distribution(weighted_squared_diff)
            # visualize_error_heatmap(weighted_squared_diff)



            return {'total_loss': total_loss}

        combined_total_loss = []
        for i in range(self.nstack):
            # for a single batch
            loss_outputs = heatmapLoss(combined_hm_preds[0][:,i], heatmaps)
            combined_total_loss.append(loss_outputs["total_loss"])

        # print(f"len(combined_total_loss): {len(combined_total_loss)}")
        # print(f"combined_total_loss: {combined_total_loss}")

        # Stack the total, basic, and focused losses separately
        combined_total_loss = torch.stack(combined_total_loss, dim=1)
        # print(f"shape(combined_total_loss): {combined_total_loss.shape}")
        # print(f"combined_total_loss): {combined_total_loss}")


        # Return a dictionary containing the combined losses
        return {
            "combined_total_loss": combined_total_loss
        }

