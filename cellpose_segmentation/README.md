# Image Preprocessing
1. Using Fiji, load all images with bioformates importer (note: make sure you keep you axis in the correct orientation)
2. Once you have all images loaded combine them into a single hyperstack. This can be a multi channel z-stack that includes many different focal planes.
3. Remove any extra channels you are not intereted in
4. Create a maximum intensity projection from the z-stack for a single representative image for each image in the hyperstack.
5. Open up the LUT and apply this to all the images. This should leave you with a single channel, single plane, with the final axis showing all the images you are processing for this experiment.
6. Save this hyperstack in a specific folder that you will access later with a script

# Cellpose Image Segmentation
1. Set up cellpose
2. Run Human-in-the-loop training on base nuclear segmentation model
3. Observe segmentations on an unseen image to determine how successful the segmentation was
4. Continue training until you are happy with the segmentation results
5. Output the final model file in a location you can access

# Run Cellpose on Hyperstack and Crop Images
1. Once you have your pre-trained model, its time to apply it to our hyperstack of preprocessed images.
2. The code below will take the pre-trained cellpose model and apply it to our images to draw masks around each cell and crop out each of the cells into an individual .tif with a unique name based on the masks into a specified output directory. 

## Example of code to submit to HPC cluster
bash run_cellpose_model_HPC.sh --model /s/cbds/boydk/SMU_Capstone/models/Nuclear_Model01 --images /s/cbds/boydk/SMU_Capstone/images --ch1 Grayscale --outdir cropped_images/
