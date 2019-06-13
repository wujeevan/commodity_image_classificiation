from functions import Functions as Funcs
from src.nnmodels import Models
from keras.models import load_model
import matplotlib.pyplot as plt


# read images and preprocess
test_path = '../test_pictures'
model_path = '../models/daily_50_9337.h5'
json_path = '../data/daily/index.json'
plot_path = '../prediction.jpg'
target_size = (224, 224)
limits = 42
cap_i = 0
x = Funcs.read_images_from_directory(test_path, target_size=target_size, limits=limits, shuffle=False)
x = Funcs.preprocess_input(x)
plt.figure()
plt.imshow(x[cap_i])
plt.axis('off')
x0 = x[cap_i].reshape((1,)+x[cap_i].shape)

model = Models.conv_vgg16()
y0 = model.predict(x0)
rows, cols = 20, 25
plt.figure(figsize=(cols, rows))
for i in range(rows*cols):
    plt.subplot(rows, cols, i+1)
    plt.imshow(y0[0, :, :, i], cmap=plt.cm.gray_r)
    # plt.title(str(i+1))
    plt.axis('off')