1. model.py contains the model structure of our CNN.
2. data.py contains a class extention of Dataset, and is what we use for training process
3. make_patches.py preprocesses a folder of images and crop them into patches of square size and save them as .npy files
	-run with ----image-folder  --input-npy --label-npy
	-patch size, stride, label size, scale factor,data augmentation,interpolation method and color mode can be further specified.
4. train.py trains the CNN model, with .npy data files as input.
	-run with --model-save-path --input-npy --label-npy
	-epochs, model save interval, vgg loss can be further specified.
5. run predict.py with a pretrained model .pth to produce prediction on images.
6. interpolation_psnr.py gives us the PSNR and SSIM metrics on the 3 interpolation methods for analysis.
	-run with --image-path --interpolation
	-scale can be further specified
7. test.py gives us the average PSNR over the test set.
	-run with --model-path --test-folder --output-path --interpolation
	-scale can be further specified
8. image_wise_psnr.py gives us PSNR and SSIM metrics of two images, used in analysis for prediction performance.
	-run with --input-path --output-path