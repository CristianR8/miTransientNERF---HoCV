import imgviz
import numpy as np
from utils import render_transient
import matplotlib.pyplot as plt
import torchvision
import torch
import os 

@torch.no_grad()
def write_summary_histogram(
    radiance_field,
    occupancy_grid,
    writer,
    dataset,
    step,
    render_step_size,
    args,
    dataset_label="test",
):
    img_scale = args.img_scale
    radiance_field.eval()
    occupancy_grid.eval()
    rgb_images = []
    depth_images = [] 
    gt_imgs = []
    accs = []

    pixels_to_plot = args.pixels_to_plot
    plotting_transients = []
    plotting_transients_depth = []

    plotting_transients_gt = []

    view_list = list(range(len(dataset)))
    # if args.version == "simulated":
    #     color_channels = 3
    # else:
    #     color_channels = 1
    # n_output_dim = args.n_bins*(color_channels)
    with torch.no_grad():

        # sample transients from network
        transient_l1_sum = 0.0
        transient_value_count = 0
        for ind, i in enumerate(view_list):
            data = dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            pixels = pixels.reshape(rays.origins.shape[0], rays.origins.shape[1], -1, 3)


            # # rendering
            out = render_transient(
            radiance_field,
            occupancy_grid,
            rays,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            render_step_size=render_step_size,
            cone_angle=args.cone_angle,
            alpha_thre=args.alpha_thre,
            use_normals = False, 
            args = args
            )

            rgb, acc, n_rendering_samples, comp_weights, depth = [out[key] for key in ['colors', 'opacities', 'n_rendering_samples', "comp_weights", "depths"]]
            del out

            rgb = rgb.reshape(rays.origins.shape[0], rays.origins.shape[1], -1, 3)

            torch.save(rgb, os.path.join(args.outpath, f"{dataset_label}_{ind}_conv.pt"))
            torch.save(depth, os.path.join(args.outpath, f"{dataset_label}_{ind}_depth.pt"))

            # Match the photometric term used for training, but evaluate the
            # complete rendered view. Chunk the temporal axis to avoid two
            # additional full-size transient tensors in GPU memory.
            for bin_start in range(0, rgb.shape[-2], 64):
                bin_stop = min(bin_start + 64, rgb.shape[-2])
                pred_chunk = rgb[..., bin_start:bin_stop, :]
                gt_chunk = pixels[..., bin_start:bin_stop, :]
                transient_l1_sum += torch.abs(
                    torch.log1p(pred_chunk) - torch.log1p(gt_chunk)
                ).sum().item()
                transient_value_count += pred_chunk.numel()

            # if color_channels ==1:
            #     gt_imgs.append(torch.clip(pixels.sum(-2).cpu().repeat(1, 1, 3).permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            #     rgb_images.append(torch.clip(rgb.sum(-2).cpu().repeat(1, 1, 3).permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            # else:
            gt_imgs.append(torch.clip(pixels.sum(-2).cpu().permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            rgb_images.append(torch.clip(rgb.sum(-2).cpu().permute(2, 0, 1)/img_scale, 0, 1)**(1/2.2))
            accs.append(acc.repeat(1, 1, 3).permute(2, 0, 1).cpu())
            dp = imgviz.depth2rgb(depth.cpu().squeeze().numpy(), colormap="inferno")
            depth_images.append(torch.from_numpy(dp).permute(2, 0, 1))

            if ind == 0:
                for pixel in pixels_to_plot:
                    plotting_transients.append(rgb[pixel[0], pixel[1], :, 0])
                    plotting_transients_gt.append(pixels[pixel[0], pixel[1], :, 0])
                    plotting_transients_depth.append(depth[pixel[0], pixel[1]])


        images = torchvision.utils.make_grid(torch.stack(gt_imgs + rgb_images + depth_images + accs), nrow=len(view_list), normalize=False)
        mse = torch.mean((torch.stack(gt_imgs, dim=0) - torch.stack(rgb_images, dim=0))**2, (1,2,3))
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        print(f"{dataset_label} image psnr: {psnr.mean():.2f}\n")
        writer.add_image(f"{dataset_label}/rgbdn", images, step)
        writer.add_scalar(
            f"Loss_eval/{dataset_label}_l1",
            transient_l1_sum / transient_value_count,
            step,
        )

        figure = plt.figure(figsize=((len(pixels_to_plot)+1), 4), dpi=250)

        # plot the predicted intensity
        plt.subplot(2, (len(pixels_to_plot)+1)//2, 1)
        plt.imshow(gt_imgs[0].permute(1, 2, 0))
        for i, pixel in enumerate(pixels_to_plot):
            plt.plot(pixel[1], pixel[0], '.', markersize=10, color='red')
            plt.text(pixel[1], pixel[0], str(i), color="yellow", fontsize=10)
        plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
        plt.title('gt intensity')

        for i, pixel in enumerate(pixels_to_plot):
            # plot transients
            plt.subplot(2, (len(pixels_to_plot)+1)//2, i+2)
            plt.plot(np.arange(args.n_bins), plotting_transients[i].detach().cpu(), label='pred', linewidth=0.5)
            plt.plot(np.arange(args.n_bins), plotting_transients_gt[i].detach().cpu(), label='gt', linewidth=0.5)
            plt.axvline(
                x=((2 * plotting_transients_depth[i] - args.start_opl) / args.exposure_time)
                .detach().cpu().numpy(),
                color='y',
            )
            plt.title(f"pixel {i}")
            plt.ylabel('intensity')
            plt.legend(borderpad=0, labelspacing=0)
            plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable='box')
        
        plt.tight_layout()
        writer.add_figure(f"{dataset_label}/transient_plots", figure, step)
        plt.close(figure)


    radiance_field.train()
    occupancy_grid.train()
    
