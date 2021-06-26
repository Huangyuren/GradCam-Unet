import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model_HRNet import *
from PIL import Image

# for tensorflow 1.x
# from tensorflow.keras.preprocessing.image import load_img,img_to_array
# import tensorflow.keras.backend as K

# for tensorflow 2.x
from keras.preprocessing.image import load_img,img_to_array
import keras.backend as K

def get_images(img_path):
    """
    generate all sample images absoute paths, and put into file_labels list
    """
    file_labels = []
    classes = []
    image_dir = os.path.normpath(img_path)
    label_i = 0
    for r, d, f in os.walk(image_dir):
        if r == image_dir:
            classes = d
        if os.path.dirname(r) == image_dir:
            f = sorted(f)
            path_f = [os.path.join(r, name) for name in f]
            file_labels += zip(path_f, len(f) * [label_i])
            label_i += 1
    return file_labels


K.set_learning_phase(1) #set learning phase
 
weight_file_dir = './Model/model0415_HRNet-3.h5'
img_path = './SampleData/'
img_all_paths = get_images(img_path)
 
model = tf.keras.models.load_model(weight_file_dir, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, \
    'acc':'acc','f1': f1, 'precision': precision, 'recall': recall})
# model.summary()
last_conv_layer = model.get_layer("conv2d_32")
# Hard code on class index
class_idx = 1
heatmap_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

for i in range(len(img_all_paths)):
    print("Currently processing: {}".format(i))
    image = cv2.imread(img_all_paths[i][0], cv2.IMREAD_GRAYSCALE)
    # print(f"Current shape: {image.shape}")
    image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    image_gray = np.expand_dims(image, axis=0)

    # code from stackoverflow: https://stackoverflow.com/questions/58322147/how-to-generate-cnn-heatmaps-using-built-in-keras-in-tf2-0-tf-keras
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(image_gray)
        loss = predictions[:, class_idx]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    # OpenCV version
    img_src_gray = cv2.imread(img_all_paths[i][0], cv2.IMREAD_COLOR)
    img_src_gray = cv2.resize(img_src_gray, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    heatmap = np.resize(heatmap, (256, 256)).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_overall = cv2.addWeighted(heatmap_img, 0.4, img_src_gray, 0.6, 0)
    print(f"[Shape] Image source: {img_src_gray.shape}, Heatmap: {heatmap_img.shape}")
    cv2.imwrite('./heatmapPlot/heatmapBlended_'+str(i)+'.png', img_overall)


    # Pillow + matplotlib version, PIL.Image size returns as (width, height); while cv2.shape returns as (height, width)
    # img_src = Image.open(img_all_paths[i][0]).convert('RGBA').resize((256, 256)) # convert to gray scale then resize to correct shape
    # heatmap = np.resize(heatmap, (img_src.size[1], img_src.size[0])).astype(np.uint8)
    # cm_jet = mpl.cm.get_cmap('jet')
    # heatmap = cm_jet(heatmap)
    # heatmap = (255 * heatmap).astype(np.uint8)
    # # Image.fromarray(heatmap).save('./heatmapPlot/heatmapPure_'+str(i)+'.png')

    # img_src = np.array(img_src)
    # dst = img_src * 0.6 + heatmap * 0.4
    # print(f"[Shape] Image source: {img_src.shape}, Heatmap: {heatmap.shape}, dst: {dst.shape}")
    # # print(f"[dtype] Image source: {img_src.dtype}, Heatmap: {heatmap.dtype}, dst: {dst.dtype}")
    # Image.fromarray(dst.astype(np.uint8)).save('./heatmapPlot/heatmapBlended_'+str(i)+'.png')