from functions import Functions as Funcs
from keras.models import load_model
import warnings

warnings.filterwarnings('ignore')
# read images and preprocess
test_path = '../test_pictures'
model_path = '../models/daily_50_9337.h5'
json_path = '../data/daily/index.json'
plot_path = '../prediction.jpg'
target_size = (224, 224)
limits = 42
x = Funcs.read_images_from_directory(test_path, target_size=target_size, limits=limits, shuffle=True)
x = Funcs.preprocess_input(x)

# load model
model = load_model(model_path)
pred = model.predict(x)
pred = Funcs.decode_prediction(pred, json_path)
x = Funcs.undo_input(x)
Funcs.plot_prediction(x, pred, fname=plot_path)
