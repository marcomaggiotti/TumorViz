from skimage.external.tifffile import imread
from PIL import Image
import matplotlib.pyplot as plt

path = '/home/sudopizzai/Documents/data/images/03_brca1_no_58_densenet_best_model__32_2_2_4_4_64_i_t.tif'
path2 = '/home/sudopizzai/Documents/data/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_32_64_1_1_13120_9408_2_8_1.0_8_svs.tif'
tifImage = imread(path2)

jpegImageArray = Image.fromarray(tifImage)
jpegImage = jpegImageArray.save("/home/sudopizzai/Documents/data/images/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF_32_64_1_1_13120_9408_2_8_1.0_8_svs.jpeg")

plt.imshow(tifImage)
plt.show()
