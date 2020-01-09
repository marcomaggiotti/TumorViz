import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import argparse
from PIL import Image, ImageDraw
from PIL import ImagePath
from matplotlib.collections import PolyCollection
import matplotlib.patches as pat
import sklearn.feature_extraction.image
import matplotlib.cm
import torch
from skimage.external.tifffile import imread
from torchvision.models import DenseNet
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
from skimage import measure
import skimage.color as color
import skimage
import openslide
import PIL.Image

from torchviz import make_dot
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",nargs="*")

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)


args = parser.parse_args(["-mbrca1_no_58_densenet_best_model_2_8_8_8_8_8.pth"])

device = torch.device(args.gpuid if args.gpuid!=-2 and torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("/home/sudopizzai/Documents/Lab/deepLearning/brac/models/brca1_no_58_densenet_best_model__32_2_2_4_4_8_t_93.pth", map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                 num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                 drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

model.load_state_dict(checkpoint["model_dict"])
model.eval()
osh  = openslide.OpenSlide("/home/sudopizzai/Documents/data/TCGA/TCGA-A8-A090-01Z-00-DX1.01574070-D65E-486F-B69D-0F8E3816D057.svs")
cmap= matplotlib.cm.tab10

#print(tiles._z_t_downsample)
#{patch_size}_{batch_size}_{level}_{mask_level}_{shape[0]}_{shape[1]}_{stride_size}_
#{tile_size}_{downsamples_level}_{cycle_step}

patch_size = 32 #if patch is less than 32 fail
batch_size = 64 #should be a power of 2
level = 1
mask_level = 1 # con la mask = 2 ValueError: could not broadcast input array from shape (16,16,3) into shape (15,16,3)

stride_size= patch_size // 16
tile_size = stride_size * 2 * 2
tile_pad = patch_size - stride_size

downsamples_level = round(osh.level_downsamples[level]) / 4

cycle_step = round(tile_size * downsamples_level) // 2

nclasses = 3

ds=int(downsamples_level)
print(ds)

size_slide = osh.level_dimensions[level]
slide_origin = (0 , 0)

img_resized = osh.read_region(slide_origin, mask_level, size_slide )

plt.imshow(img_resized)
plt.show()
print("image size ", img_resized.size)
shape = img_resized.size #osh.level_dimensions[level]

img = img_resized #np.asarray(new_tile9)[:, :, 0:3] #osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level]) #
imgg=rgb2gray(np.array(img))
#imgg = img.convert('LA')

mask=np.bitwise_and(imgg >0 ,imgg <200/255)
shaperound=[((d//tile_size)+1)*tile_size for d in shape]

def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n,::]

print("Patch Floor Division", patch_size//patch_size,patch_size//patch_size)
npmm = np.zeros((shaperound[1]//stride_size,shaperound[0]//stride_size,3),dtype=np.uint8)

#cycle_step = 8#round(tile_size * downsamples_level)
y_space = range(0,osh.level_dimensions[0][1],round(tile_size * downsamples_level))
x_space = range(0,osh.level_dimensions[0][0],round(tile_size * downsamples_level))

print(range(0,img_resized.height,round(tile_size * downsamples_level)))
print(range(0,img_resized.width,round(tile_size * downsamples_level)))

plt.imshow(mask)
plt.show()

print(f'cycle_step_{cycle_step}_patchSize_{patch_size}_batchSize_{batch_size}_level_{level}_maskLevel_{mask_level}_{shape[0]}_{shape[1]}_StrideSize{stride_size}')
for y in tqdm(range(0, img_resized.height,cycle_step), desc="outer"):
    for x in tqdm(range(0, img_resized.width,cycle_step), desc=f"innter {y}", leave=False):

        maskx = int(x//downsamples_level)
        masky = int(y//downsamples_level)

        if(maskx>= mask.shape[1] or masky>= mask.shape[0] or not mask[masky,maskx]): #need to handle rounding error
            #print("jump the model")
            continue

        #print(mask[masky,maskx])
        #print("x",x, "y",y,"maskx",maskx, " masky", masky, " mask.shape[1]", mask.shape[1], " mask.shape[0]", mask.shape[0] , "tile_size", tile_size, " osh.level_downsamples[level]",  osh.level_downsamples[level] )

        #coordinates = [[x,y], [x+tile_size+tile_pad , y],[x+tile_size+tile_pad,y+tile_size+tile_pad],[x,y+tile_size+tile_pad]]
        patch_size//patch_size
        output = np.zeros((0,nclasses, patch_size//patch_size,patch_size//patch_size ))
        io = np.asarray(osh.read_region((x, y), 3, (tile_size+tile_pad,tile_size+tile_pad)))[:,:,0:3]

        arr_out = sklearn.feature_extraction.image.extract_patches(io,(patch_size,patch_size,3),stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1,patch_size,patch_size,3)

        for batch_arr in divide_batch(arr_out,1):

            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
            #print("arr_out_gpu", arr_out_gpu)
            output_batch = model(arr_out_gpu)
            #make_dot(output_batch)
            output_batch = output_batch.detach().cpu().numpy()
            output_batch_color = cmap(output_batch.argmax(axis=1), alpha=None)[:,0:3]
            output = np.append(output,output_batch_color[:,:,None,None],axis=0)

        output = output.transpose((0, 2, 3, 1))
        output = output.reshape(arr_out_shape[0],arr_out_shape[1],patch_size//patch_size,patch_size//patch_size,output.shape[3])

        output=np.concatenate(np.concatenate(output,1),1)
        #print(y//stride_size//ds,y//stride_size//ds+tile_size//stride_size,x//stride_size//ds,x//stride_size//ds+tile_size//stride_size)
        npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size,x//stride_size//ds:x//stride_size//ds+tile_size//stride_size,:]=output*255

from skimage.external.tifffile import TiffWriter
with TiffWriter(f'/home/sudopizzai/Documents/data/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_{patch_size}_{batch_size}_{level}_{mask_level}_{shape[0]}_{shape[1]}_{stride_size}_{tile_size}_{downsamples_level}_{cycle_step}_svs.tif', bigtiff=True, imagej=True) as tif:
    tif.save(npmm, compress=0, tile=(256,256))

image = Image.open(f'/home/sudopizzai/Documents/data/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_{patch_size}_{batch_size}_{level}_{mask_level}_{shape[0]}_{shape[1]}_{stride_size}_{tile_size}_{downsamples_level}_{cycle_step}_svs.tif')
image.mode = 'I'
image.point(lambda i:i*(1./256)).convert('L').save(f'/home/sudopizzai/Documents/data/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_{patch_size}_{batch_size}_{level}_{mask_level}_{shape[0]}_{shape[1]}_{stride_size}_{tile_size}_{downsamples_level}_{cycle_step}_svs.jpeg')
