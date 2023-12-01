#!/usr/bin/env python3
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G
#SBATCH --time=1-00:00
#SBATCH --mail-user=$USER@omrf.org
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

# Check nvidia setup
#get_ipython().system('nvcc --version')
#get_ipython().system('nvidia-smi')

import os, shutil
import numpy as np
#import matplotlib.pyplot as plt
import argparse
from cellpose import core, utils, io, models, metrics
from glob import glob
import tifffile
from skimage.measure import label, regionprops

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

color_options = ['Grayscale', 'Red', 'Green', 'Blue']


# MAIN FUNCTION
def main(args):
    model_path = args.model
    dir = args.images
    Channel_to_use_for_segmentation = args.ch1
    Second_segmentation_channel = args.ch2

    if Channel_to_use_for_segmentation == "Grayscale":
        chan = 0
    elif Channel_to_use_for_segmentation == "Blue":
        chan = 3
    elif Channel_to_use_for_segmentation == "Green":
        chan = 2
    elif Channel_to_use_for_segmentation == "Red":
        chan = 1

    if Second_segmentation_channel == "Blue":
        chan2 = 3
    elif Second_segmentation_channel == "Green":
        chan2 = 2
    elif Second_segmentation_channel == "Red":
        chan2 = 1
    elif Second_segmentation_channel == "None":
        chan2 = 0

    diameter = 0
    flow_threshold = 0.4
    cellprob_threshold = 0

    channels = [chan, chan2]

    files = io.get_image_files(dir, '_masks')
    print(files)

    images = (io.imread(f) for f in files)

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    diameter = model.diam_labels if diameter == 0 else diameter

    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Iterate over each file and its corresponding image in parallel
    for file, image in zip(files, images):
        basename = os.path.splitext(os.path.basename(file))[0]
        print(f'Starting on\nImage name: {basename}\nImage Shape: {image.shape}\n')

        print(f'Running Model on Image: {basename}\n(This can take a while)')

        if len(image.shape) == 3:
            # Loop through each frame of the 3D image
            for frame_idx in range(image.shape[0]):
                # Get the current frame
                current_frame = image[frame_idx]

                # Run the model's evaluation method on the current frame
                masks, flows, styles = model.eval(current_frame,
                                                  channels=[chan, chan2],
                                                  diameter=diameter,
                                                  flow_threshold=flow_threshold,
                                                  cellprob_threshold=cellprob_threshold,
                                                  do_3D=False)

                print(f'Successfully applied model to frame {frame_idx + 1} of image: {basename}!')

                # Print the shape of the 'masks' variable
                print(f"Shape of masks: {masks.shape}")

                # Check if the masks array is 2D or a list of 2D masks
                if len(masks.shape) == 2:
                    masks = [masks]

                # Iterate over each mask and apply it to the original frame
                for idx, mask in enumerate(masks):
                    # Threshold the mask to get a binary mask (0 or 1)
                    binary_mask = mask > 0.5

                    print(f"Shape of binary_mask: {binary_mask.shape}")
                    print(f"Binary mask contents:\n{binary_mask}")
                    print(f"Sum of True values in binary mask: {binary_mask.sum()}")


                    # Use the label function to identify individual cells in the mask
                    labeled_mask = label(binary_mask)
                    num_cells = labeled_mask.max()  # Calculate the number of labels (connected components)

                    print(f"Shape of labeled_mask: {labeled_mask.shape}")
                    print(f"Number of cells: {num_cells}")

                    # Iterate over each cell identified by the label function
                    for cell_label in range(1, num_cells + 1):
                        # Extract the individual cell from the labeled mask
                        cell_mask = (labeled_mask == cell_label).astype(np.uint8)

                        # Find the bounding box of the cell
                        rows, cols = np.where(cell_mask)
                        minr, minc = rows.min(), cols.min()
                        maxr, maxc = rows.max() + 1, cols.max() + 1

                        # Extract the region from the original frame using the bounding box coordinates
                        cell_image = current_frame[minr:maxr, minc:maxc]

                        # Save the resulting cell image as a separate TIFF file with metadata axes as 'YX'
                        tiff_name = os.path.join(outdir, f"{basename}_frame{frame_idx}_cell_{cell_label}.tif")
                        tifffile.imwrite(tiff_name, cell_image, imagej=True, metadata={'axes': 'YX'})


        else:
            # The same code as before when dealing with a single 2D image
            masks, flows, styles = model.eval(image,
                                            channels=[chan, chan2],
                                            diameter=diameter,
                                            flow_threshold=flow_threshold,
                                            cellprob_threshold=cellprob_threshold,
                                            do_3D=False)

            print(f'Successfully applied model to image: {basename}!')
            # Print the shape of the 'masks' variable
            print(f"Shape of masks: {masks.shape}")

            # Check if the masks array is 2D or a list of 2D masks
            if len(masks.shape) == 2:
                masks = [masks]

            # Iterate over each mask and apply it to the original image
            for idx, mask in enumerate(masks):
                # Apply the current mask to the original image using element-wise multiplication
                masked_image = image * mask

                # Save the resulting masked image as a separate TIFF file with metadata axes as 'YX'
                tiff_name = os.path.join(outdir, f"{basename}_mask_{idx}.tif")
                tifffile.imwrite(tiff_name, masked_image, imagej=True, metadata={'axes': 'YX'})

                # Optionally, you can keep all the masked images in a list if needed
                # masked_images.append(masked_image)

# Set up command-line argument parser:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script which applies a cellpose model to a directory of .tif images and returns segmented label images.")

    parser.add_argument("--model", metavar="model_path", required=True, help="Path to the desired cellpose model file")
    parser.add_argument("--images", metavar="image_path", required=True, help="Path to the folder containing .tif images")
    parser.add_argument("--ch1", metavar="Ch1_color", required=True, help=f"Primary channel to use for segmentation. Options are {color_options}", choices=color_options)
    parser.add_argument("--ch2", metavar="Ch2_color", help=f"Optional channel to use as secondary for segmentation. Options are {color_options}", choices=color_options, default="None")
    parser.add_argument("--outdir", metavar="outdir_path", help="Name or path of directory to store output. If dir doesn't exist, it will be created automatically. Default is current directory.", default=os.getcwd())
    args = parser.parse_args()

    main(args)
