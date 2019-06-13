from src.nnmodels import Models
from src.functions import Functions as Funcs
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
import os
import datetime


# initialize base variable
epochs = 120
model_design = Models.mnist_fashion
data_name = 'mnist_fashion'
dirs = '../data/{}'.format(data_name)
json_path = '{}/train/index.json'.format(dirs)
model_save_path = '../models/{}_{}.h5'.format(data_name, epochs)
model_load_path = '../models/{}_{}_{}.h5'.format(data_name, epochs, 10)
model_history_path = '../train_info/{}_{}_modelHis.dat'.format(data_name, epochs)
history_path = '../train_info/{}_{}_his.jpg'.format(data_name, epochs)
pred_path = '../train_info/{}_{}_pred.jpg'.format(data_name, epochs)
target_size = (28, 28)
classes = 10
num_pred = 30
batch_size = 32
gray = True
load_existent_model = False

# load train and test data
print('loading data ...')
(x_train, y_train), (x_test, y_test) = Funcs.load_data(dirs, target_size, gray=gray, classes=classes)
print('loading data done')

# define callbacks for models to render model performance better
checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc', save_best_only=True, verbose=0)
lrschedual = LearningRateScheduler(lambda epoch: Funcs.lr_schedual(epoch, epochs=epochs), verbose=0)

# whether load existent model
if load_existent_model and os.path.exists(model_load_path):
    model = load_model(model_load_path)
else:
    model = model_design(x_train.shape[1:], classes)

# train the model
start_time = datetime.datetime.now()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_test, y_test), callbacks=[checkpoint, lrschedual], verbose=1)
end_time = datetime.datetime.now()
best_model = load_model(model_save_path)
loss_acc = best_model.evaluate(x_test, y_test)
print('loss: {:.3f}, acc: {:.3f}'.format(loss_acc[0], loss_acc[1]))
print('training time: {}s'.format((end_time-start_time).seconds))

# plot the history of training
Funcs.save_history(model.history, model_history_path)
Funcs.plot_history(model.history, history_path, num_xticks=6)

# use highest accuracy model for prediction
pred = best_model.predict(x_test)
pred_idx = Funcs.get_pred_index(pred, num_pred=num_pred, y=y_test)
x = Funcs.undo_input(x_test[pred_idx])
pred = Funcs.decode_prediction(pred[pred_idx], json_path)
y = Funcs.decode_prediction(y_test[pred_idx], json_path)
Funcs.plot_prediction(x, pred, fname=pred_path, y=y)

# amend model filename to easily recognize
new_model_save_path = model_save_path[:-3] + '_' + str(int(loss_acc[1]*10000)) + model_save_path[-3:]
os.rename(model_save_path, new_model_save_path)
print('saved model to {}'.format(new_model_save_path))
