__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import pandas as pd
import numpy as np


def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(filename.replace(".", "_"))
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.savefig(filename.replace(".", "_") + "_multi")
    plt.show()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    configs = json.load(open('孤岛采油厂采油管理六区.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)

    # x, y = data.get_train_data(
    #     configs['data']['sequence_length'],
    #     configs['data']['normalise']
    # )

    # in-memory training
    # model.train(
    #     x,
    #     y,
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     save_dir=configs['model']['save_dir']
    # )

    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch_2(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    x_test, y_test = data.get_test_data_2(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # predict point by point
    # x_test, y_test = data.get_test_data_point_by_point(
    #     seq_len=configs['data']['sequence_length'],
    #     normalise=configs['data']['normalise']
    # )

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)

    # 将预测值转换为scaler输入的形状
    predictions = predictions.reshape(-1, 1)
    # 将预测值进行反归一化
    predictions = data.denormalize(predictions)
    # 将预测值转换为整型(四舍五入)
    # predictions_int = np.round(predictions).astype(int)

    y_test = data.denormalize(y_test.reshape(-1, 1))
    datad = np.concatenate((y_test, predictions), axis=1)

    dataplt = pd.read_csv(os.path.join('data', configs['data']['filename']), encoding='GB2312')
    train_test_split = configs['data']['train_test_split']
    plt_size = int(len(dataplt) * train_test_split)
    plttest = dataplt[plt_size:-50]
    pltx = plttest['年 / 周']
    pltnums = plttest['仪表数量']
    plttimes = plttest['运维次数']

    predictions_values = predictions * np.array(pltnums).reshape(-1,1)
    predictions_values = np.round(predictions_values).astype(int)
    datad = np.concatenate((np.array(pltx).reshape(-1,1)[-10:],np.array(plttimes).reshape(-1,1)[-10:], predictions_values[-10:]), axis=1)
    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制“运维次数”线
    plt.plot(np.array(pltx)[-10:], np.array(plttimes)[-10:], label='real')

    # 绘制预测值与“仪表数量”相乘的结果线
    plt.plot(np.array(pltx)[-10:], np.array(predictions_values)[-10:], label='prediction')

    plt.xlabel('year-week')
    plt.ylabel('Values')
    plt.title(configs['data']['title'])
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.savefig(configs['data']['filename'].replace(".", "_"))
    plt.show()

    # 创建一个 DataFrame
    dataf = pd.DataFrame(datad, columns=['年 / 周', '真实运维次数', '预测运维次数'])
    # dataf = np.concatenate(dataf, predictions_values)
    # 将数据保存到 CSV 文件
    dataf.to_csv(configs['data']['filename'], index=False, encoding='GB2312')

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], configs['data']['filename'])
    # plot_results(predictions, y_test, configs['data']['filename'])


if __name__ == '__main__':
    main()
    # use_model()
