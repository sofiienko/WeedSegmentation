import argparse
from scipy import misc
import matplotlib


from Unet import unet_sigmoid
from helpers import load_model,prepare_image_for_model, prediction_to_image, prepare_image_for_model,resize_img,prediction_to_mask

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="input image field")

args = vars(ap.parse_args())
row_img = misc.imread(args["image"])

model = load_model()

print(type(row_img))

smallImage = resize_img(row_img)
dataForModel =  prepare_image_for_model(smallImage)

preds = model.predict(dataForModel)

image_mask = prediction_to_mask(preds,0.20)
image_with_mask = prediction_to_image(preds,smallImage,0.20 )
matplotlib.image.imsave('o.png', smallImage)
matplotlib.image.imsave('m.png', image_mask)
matplotlib.image.imsave('im.png', image_with_mask)

