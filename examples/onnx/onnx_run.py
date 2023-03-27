# Tutorial: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
#%%
from PIL import Image
import torchvision.transforms as transforms

img = Image.open("./_static/img/cat.jpg")
resize = transforms.Resize([224, 224])
img = resize(img)
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,3,1)
plt.imshow(img_y)
plt.title("Y(greyscale)")
plt.subplot(1,3,2)
plt.imshow(img_cb)
plt.title("Cb(blue-difference)")
plt.subplot(1,3,3)
plt.imshow(img_cr)
plt.title("Cr(red-difference)")
plt.tight_layout()
plt.show()

#%%
# Convert the Y component to a tensor because it's more sensitive to the human eye
to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

#%%
import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx",
                                        providers=['CUDAExecutionProvider'])
                                        
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

#%%
import numpy as np
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("./_static/img/cat_superres_with_ort.jpg")
# %%
