<h2>Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel (2025/02/05)</h2>

This is the first experiment of Tiled Image Segmentation for <b>DRIVE Retinal Vessel</b>
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/1ysb43j3TzzE-PxhySPCAUshOsQzXIYi2/view?usp=sharing">
Augmented-Tiled-DRIVE-ImageMask-Dataset.zip</a>, which was derived by us from the following dataset:<br><br>
<a href="https://data.mendeley.com/public-files/datasets/frv89hjgrr/files/f61c5f08-f18d-4206-8416-a4c8a69b3fce/file_downloaded">
DRIVE.7z</a> in <a href="https://data.mendeley.com/datasets/frv89hjgrr/1"><b>Mendeley Data Retinal Vessel</b></a>.
<br>
<br>
On detail of <b>DRIVE</b>, 
please refer to the official site:<br>
<a href="https://drive.grand-challenge.org/">
DRIVE: Digital Retinal Images for Vessel Extraction
</a>, and github repository <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md">
DRIVE</a>.
<br><br>
Please see also our experiments:<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a> based on 
<a href="https://cecas.clemson.edu/~ahoover/stare/">STructured Analysis of the Retina</a>.
<br>

<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
Tensorflow-Image-Segmentation-Retinal-Vessel</a> based on <a href="https://researchdata.kingston.ac.uk/96/">CHASE_DB1 dataset</a>.
<br>
<br>
<b>Experiment Strategies</b><br>
As demonstrated in our experiments 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a>
 and 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-IDRiD-HardExudates">
Tensorflow-Tiled-Image-Segmentation-IDRiD-HardExudates </a>, 
the Tiled Image Segmentation based on a simple UNet model trained by a tiledly-splitted images and masks dataset, 
is an effective method for the large image segmentation over 4K pixels.
Furthermore, as mentioned in 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a>,  
it is difficult to precisely segment Retinal Blood Vessels in small images using a simple UNet model 
because these vessels are typically very thin. 
Therefore, we generate a high-resolution retinal 
image dataset by upscaling the original images and use it to train the UNet model to improve segmentation performance.
<br>
<br>
In this experiment, we employed the same strategies in this project as we did in the 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
STARE-Retinal-Vessel</a>.
<br>
<b>1. Enlarged Dataset</b><br>
We generated a 6x enlarged dataset of 40 JPG images and masks, each with 3390x3504 pixels, from the original DRIVE 565x584 pixels 
TIF image and GIF mask files using bicubic interpolation.
<br>
<br>
<b>2. Pre Augemtned Tiled DRIVE ImageMask Dataset</b><br>
We generated a pre-augmented image mask dataset from the enlarged dataset, which was tiledly-splitted to 512x512 pixels 
and reduced to 512x512 pixels image and mask dataset.
<br>
<br>
<b>3. Train Segmention Model </b><br>
We trained and validated a TensorFlow UNet model by using the <b>Pre Augmented Tiled DRIVE ImageMask Dataset</b>
<br>
<br>
<b>4. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict the DRIVE Retinal Vessel for the mini_test images 
with a resolution of 3390x3504 pixels of the Enlarged Dataset.<br><br>

<hr>
<b>Actual Tiled Image Segmentation for Images of 3390x3504 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/1.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/2.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/4.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/4.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/4.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this DRIVESegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the dataset 
<a href="https://data.mendeley.com/public-files/datasets/frv89hjgrr/files/f61c5f08-f18d-4206-8416-a4c8a69b3fce/file_downloaded">
DRIVE.7z</a>
in <a href="https://data.mendeley.com/datasets/frv89hjgrr/1"> <b>Retinal Vessel </b></a>
<br>
On more detail, please refer to the github repository <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md">
DRIVE</a>
<br><br>
<b>Authors and Institutions</b><br>
Joes Staal (Image Sciences Institute, University Medical Center Utrecht)<br>
Michael D. Abràmoff (Department of Ophthalmology and Visual Sciences, University of Iowa)<br>
Meindert Niemeijer (Image Sciences Institute, University Medical Center Utrecht)<br>
Max A. Viergever (Image Sciences Institute, University Medical Center Utrecht)<br>
Bram van Ginneken (Image Sciences Institute, University Medical Center Utrecht)<br>
<br>
<b>Citation</b><br>
@ARTICLE{1282003,<br>
  author={Staal, J. and Abramoff, M.D. and Niemeijer, M. and Viergever, M.A. and van Ginneken, B.},<br>
  journal={IEEE Transactions on Medical Imaging}, <br>
  title={Ridge-based vessel segmentation in color images of the retina},<br> 
  year={2004},<br>
  volume={23},<br>
  number={4},<br>
  pages={501-509},<br>
  doi={10.1109/TMI.2004.825627}}<br>
<br>
<!--
<b>Licence</b><br>
CC BY 4.0
<br>
 -->
<h3>
<a id="2">
2 Augmented-Tiled-DRIVE ImageMask Dataset
</a>
</h3>
 If you would like to train this DRIVE Segmentation model by yourself,
 please download the pre-augmented dataset from the google drive  
<a href="https://drive.google.com/file/d/1ysb43j3TzzE-PxhySPCAUshOsQzXIYi2/view?usp=sharing">
Augmented-Tiled-DRIVE-ImageMask-Dataset.zip</a>,
 expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Augmented-Tiled-DRIVE
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a 512x512 pixels pre augmented tiles dataset generated from 3500x3025 pixels 20 <b>Enlarged-images</b> and
their corresponding <b>Enlarged-masks</b>.<br>
.<br>

The folder structure of the original <b>DRIVE/training</b> dataset is the following.<br>

<pre>
./DRIVE
└─training
    ├─1st_manual
    │   ├─21_manual1.gif
    │   ├─22_manual1.gif
     ...    
    │   └─40_manual1.gif
    └─images
        ├─21_training.tif
        ├─22_training.tif
         ...
        └─40_training.tif
</pre>
We excluded all black (empty) masks and their corresponding images to generate our dataset from the original DRIVE.<br>  
On the derivation of the dataset, please refer to the following Python scripts.<br>
<li><a href="./generator/Preprocessor.py">Preprocessor.py</a></li>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_tiled_master.py">split_tiled_master.py</a></li>
<br>
<br>
<b>Augmented-Tiled-DRIVE Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/Augmented-Tiled-DRIVE_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained DRIVE TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_LINEAR"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Tiled inference</b><br>
We used 3390x3504 pixels enlarged images and masks generated by <a href="./generator/Preprocessor.py">
Preprocessor.pys
</a>  as a mini_test dataset for our TiledInference.
<pre>
[tiledinfer] 
overlapping   = 64
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer      = False
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiledinfer procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at ending (73,74,75)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 75 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/train_console_output_at_epoch_75.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for DRIVE.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/evaluate_console_output_at_epoch_75.png" width="720" height="auto">
<br><br>Image-Segmentation-DRIVE

<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Augmented-Tiled-DRIVE/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.1524
dice_coef,0.8389
</pre>
<br>

<h3>
5 Tiled inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for DRIVE.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (3390x3504 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks (3390x3504 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 3390x3504 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/3.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/5.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/7.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/7.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/7.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/9.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/9.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/11.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/11.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/11.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/images/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test/masks/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-DRIVE/mini_test_output_tiled/12.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Locating Blood Vessels in Retinal Images by Piecewise Threshold Probing of a Matched Filter Response</b><br>
Adam Hoover, Valentina Kouznetsova, and Michael Goldbaum<br>
<a href="https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf">
https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf
</a>
<br>
<br>
<b>2. DRIVE: Digital Retinal Images for Vessel Extraction</b><br>
<a href="https://drive.grand-challenge.org/">https://drive.grand-challenge.org/</a>
<br>
<br>
<b>3. DRIVE</b><br>
<a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md">
https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md
</a>
<br>
<br>
<b>4. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed<br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>5. Retinal blood vessel segmentation using a deep learning method based on modified U-NET model</b><br>
Sanjeewani, Arun Kumar Yadav, Mohd Akbar, Mohit Kumar, Divakar Yadav<br>
<a href="https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3">
https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3</a>
<br>
<br>
<b>6. Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a>
<br>
<br>
<b>7, Tensorflow-Image-Segmentation-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel</a>
<br>
<br>
<b>8. Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer
</a>
<br>
<br>
<b>9. Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma
</a>
<br>
<br>
<b>10. Tiled-ImageMask-Dataset-Breast-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer
</a>
<br>
<br>

