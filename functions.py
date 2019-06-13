import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import pickle
import seaborn
import os
seaborn.set_style(style='darkgrid')


class Functions:
    def __init__(self):
        pass

    @staticmethod
    def read_images_from_directory(directory, target_size=(64, 64), gray=False, limits=np.Inf, shuffle=False):
        images, cnt = [], 0
        for i, image_name in enumerate(os.listdir(directory)):
            image_path = os.path.join(directory, image_name)
            image = np.array(Image.open(image_path).resize(target_size))
            if (image.shape[-1] == 3 or gray) and (cnt < limits):
                images.append(image)
                cnt = cnt + 1
        if shuffle:
            np.random.shuffle(images)
        return np.array(images)

    @staticmethod
    def read_images_from_directories(directory, target_size=(64, 64), gray=False, max_classes=100, per_class=np.Inf,
                                     shuffle=False, get_counts=False, init_json=False, json_path=None):
        images, labels, counts, y_dict = [], [], [], {}
        temp_path = os.path.join(directory, 'index.json')
        if json_path is None:
            json_path = temp_path
        if not os.path.exists(json_path):
            init_json = True
        if init_json:
            y_dict = Functions.write_json(directory, json_path)
        else:
            y_dict = Functions.read_json(json_path)
        if not os.path.exists(temp_path):
            _ = Functions.write_json(json_path=temp_path, y_dict=y_dict)

        for i, label in y_dict.items():
            dir_ = os.path.join(directory, label)
            temp = Functions.read_images_from_directory(dir_, target_size, gray, per_class, shuffle)
            images.append(temp)
            labels.append([i for _ in temp])
            counts.append(temp.shape[0])
            if int(i) >= max_classes:
                break

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        counts = np.array(counts)
        if shuffle:
            index = np.arange(len(images))
            np.random.shuffle(index)
            images = images[index]
            labels = labels[index]
        if get_counts:
            return images, labels, counts
        else:
            return images, labels

    @staticmethod
    def split_train_test(images, labels, counts, train_ratio=0.8):
        x, y, n = images, labels, counts
        x_train, y_train, x_test, y_test = [[] for _ in range(4)]
        now = {}
        for i, z in enumerate(x):
            if y[i] not in now:
                now[y[i]] = 0
            now[y[i]] += 1
            if now[y[i]] < n[y[i]] * train_ratio:
                x_train.append(z)
                y_train.append(y[i])
            else:
                x_test.append(z)
                y_test.append(y[i])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def write_json(directory=None, json_path=None, y_dict=None):
        if y_dict is None:
            i, y_dict = 0, {}
            for label in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, label)):
                    y_dict[i] = label
                    i = i + 1
        with open(json_path, 'w') as f:
            json.dump(y_dict, f, ensure_ascii=False, indent=4)
        return y_dict

    @staticmethod
    def read_json(json_path):
        y_dict, temp = {}, {}
        with open(json_path, 'r') as f:
            temp = json.load(f)
        for i, label in temp.items():
            y_dict[int(i)] = label
        return y_dict

    @staticmethod
    def show_images_from_directory(directory, target_size=(64, 64), gray=False, subdir=True, limits=100, fname=None):
        if subdir:
            classes = 0
            for label in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, label)):
                    classes += 1
            max_classes = min([limits, classes])
            per_class = np.floor(limits / max_classes)
            x, y = Functions.read_images_from_directories(directory, target_size=target_size, gray=gray,
                                                          max_classes=max_classes, per_class=per_class,
                                                          shuffle=False, get_counts=False, init_json=False)
        else:
            x = Functions.read_images_from_directory(directory, target_size=target_size, gray=gray,
                                                     limits=limits, shuffle=True)
        Functions.show_images_from_numpy(x, gray, fname)

    @staticmethod
    def show_images_from_numpy(x, gray=False, fname=None):
        total = len(x)
        rows = np.ceil(np.sqrt(total))
        cols = np.ceil(total / rows)
        plt.figure(figsize=(4, 4))
        for i, image in enumerate(x):
            plt.subplot(rows, cols, i + 1)
            if gray:
                plt.imshow(image, cmap=plt.cm.gray_r)
            else:
                plt.imshow(image)
            plt.axis('off')

        if fname is not None:
            plt.savefig(fname)

    @staticmethod
    def preprocess_input(x):
        x = x / 127.5
        x = x - 1
        return x

    @staticmethod
    def undo_input(x):
        x = x + 1
        x = x * 127.5
        x = x.astype(np.uint8)
        return x

    @staticmethod
    def to_categorical(y, num_class=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(y.shape[:-1])
        y = y.ravel()
        n = y.shape[0]
        if not num_class:
            num_class = y.max() + 1
        categorical = np.zeros((n, num_class), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_class,)
        categorical = categorical.reshape(output_shape)
        return categorical

    @staticmethod
    def decode_prediction(prediction, json_path, y_dict=None):
        if y_dict is None:
            y_dict = Functions.read_json(json_path)
        return [y_dict[pred.argmax()] for pred in prediction]

    @staticmethod
    def lr_schedual(epoch, epochs):
        if epoch < epochs / 4:
            return 0.001
        elif epoch < epochs * 3 / 4:
            return 0.0001
        else:
            return 0.00005

    @staticmethod
    def save_history(his, fname):
        with open(fname, 'wb') as f:
            pickle.dump(his, f)

    @staticmethod
    def load_history(fname):
        with open(fname, 'rb') as f:
            his = pickle.load(f)
        return his

    @staticmethod
    def plot_history(his, fname=None, num_xticks=6):
        xticks = np.linspace(his.epoch[0], his.epoch[-1]+1, num_xticks+1)

        plt.figure(figsize=(6, 3))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplots_adjust(left=0.12, bottom=0.14, right=0.9, top=0.9, wspace=0.25, hspace=0.20)
        plt.subplot(1, 2, 1)
        plt.plot(his.epoch, his.history['acc'])
        plt.plot(his.epoch, his.history['val_acc'])
        plt.legend(['train', 'test'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.xticks(xticks)

        plt.subplot(1, 2, 2)
        plt.plot(his.epoch, his.history['loss'])
        plt.plot(his.epoch, his.history['val_loss'])
        plt.legend(['train', 'test'])
        plt.xticks(xticks)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.show()

        if fname is not None:
            plt.savefig(fname)

    @staticmethod
    def get_pred_index(pred, num_pred=25, y=None, ratio=0.8):
        if y is None:
            pred_idx = np.random.choice(np.arange(len(pred)), num_pred)
        else:
            true_idx, false_idx, num_true = [], [], np.ceil(num_pred * ratio)
            for i in range(len(pred)):
                if pred[i].argmax() == y[i].argmax():
                    true_idx.append(i)
                else:
                    false_idx.append(i)
            t1 = np.random.choice(true_idx, int(num_true))
            t2 = np.random.choice(false_idx, int(num_pred - num_true))
            pred_idx = np.concatenate([t1, t2])
        return pred_idx

    @staticmethod
    def plot_prediction(x, pred, y=None, fname=None):
        total = len(x)
        rows = np.floor(np.sqrt(total))
        cols = np.ceil(total / rows)
        plt.figure(figsize=(cols, rows))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.75, hspace=0.75)
        if x.shape[-1] == 1 and len(x.shape) > 2:
            x = x.reshape(tuple(x.shape[:-1]))

        for i in range(total):
            plt.subplot(rows, cols, i + 1)
            if len(x.shape) < 4:
                plt.imshow(x[i], cmap=plt.cm.gray_r)
            else:
                plt.imshow(x[i])
            if y is not None:
                if y[i] != pred[i]:
                    plt.title('{}({})'.format(pred[i], y[i]), color='red', fontsize='medium')
                else:
                    plt.title('{}({})'.format(pred[i], y[i]), fontsize='medium')
            else:
                plt.title('{}'.format(pred[i]), fontsize='medium')
            plt.axis('off')
        
        plt.show()

        if fname is not None:
            plt.savefig(fname)

    @staticmethod
    def trans_data(x, y, classes=10):
        x = Functions.preprocess_input(x)
        y = Functions.to_categorical(y, classes)
        if len(x.shape) < 4:
            x = np.expand_dims(x, axis=len(x.shape))
        return x, y

    @staticmethod
    def load_data(directory, target_size, classes=None, gray=False, shuffle=False):
        dir_train = os.path.join(directory, 'train')
        dir_test = os.path.join(directory, 'test')
        x_train, y_train = Functions.read_images_from_directories(dir_train, target_size, gray=gray, shuffle=shuffle)
        x_test, y_test = Functions.read_images_from_directories(dir_test, target_size, gray=gray, shuffle=shuffle)
        (x_train, y_train) = Functions.trans_data(x_train, y_train, classes)
        (x_test, y_test) = Functions.trans_data(x_test, y_test, classes)
        return (x_train, y_train), (x_test, y_test)
