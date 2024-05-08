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


def use_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    configs = json.load(open(os.path.join('configs', '孤岛采油厂_采油管理六区.json'), 'r'))
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    x_test, y_test = data.get_test_data_2(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    num_steps = 90
    model = Model()
    model.load_model('saved_models/26042024-060633-e2.h5')
    prediction = model.predict_point_by_point_2(x_test, num_steps)
    plot_results(prediction, prediction, "gudao6")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    configs = json.load(open(os.path.join('configs', '孤岛采油厂_采油管理二区.json'), 'r'))
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
    initial_input = data.data_test[-49:]

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    predictions = model.predict_point_by_point(x_test)

    # 预测结果的长度
    desired_time_steps = 90

    # 循环预测直到达到所需长度
    generated_data  = []
    current_input = initial_input.reshape(1,49,1)
    for _ in range(desired_time_steps - predictions.shape[0]):
        # 使用模型预测下一个时间步长的数据
        next_step_prediction = model.predict_point_by_point(current_input)
        generated_data.append(next_step_prediction)

        next_step_prediction = next_step_prediction.reshape(1,1,1)
        # 更新输入序列，将当前预测结果添加到输入序列中
        current_input = np.concatenate((current_input, next_step_prediction), axis=1)[:, -49:, :]

    generated_data = np.array(generated_data)
    prediction_datas = np.concatenate((predictions.reshape(-1,1), generated_data), axis=0)
    prediction_datas = data.denormalize(prediction_datas)
    plt.plot(prediction_datas)
    plt.show()

    # # 将预测值转换为scaler输入的形状
    # predictions = predictions.reshape(-1, 1)
    # # 将预测值进行反归一化
    # predictions = data.denormalize(predictions)
    # # 将预测值转换为整型(四舍五入)
    # # predictions_int = np.round(predictions).astype(int)
    #
    # y_test = data.denormalize(y_test.reshape(-1, 1))
    # datad = np.concatenate((y_test, predictions), axis=1)
    #
    # dataplt = pd.read_csv(os.path.join('data', configs['data']['filename']))
    # train_test_split = configs['data']['train_test_split']
    # plt_size = int(len(dataplt) * train_test_split)
    # plttest = dataplt[plt_size:-50]
    # pltx = plttest['数据统计日']
    # pltnums = plttest['仪表数量']
    # plttimes = plttest['该运维内容对应次数']
    #
    # predictions_values = predictions * np.array(pltnums).reshape(-1,1)
    # predictions_values = np.round(predictions_values).astype(int)
    # datad = np.concatenate((np.array(pltx).reshape(-1,1),np.array(plttimes).reshape(-1,1), predictions_values), axis=1)
    # # 绘图
    # plt.figure(figsize=(10, 6))
    #
    # # 绘制“运维次数”线
    # plt.plot(np.array(pltx), np.array(plttimes), label='real')
    #
    # # 绘制预测值与“仪表数量”相乘的结果线
    # plt.plot(np.array(pltx), np.array(predictions_values), label='prediction')
    #
    # plt.xlabel('year-week')
    # plt.ylabel('Values')
    # plt.title(configs['data']['title'])
    # plt.xticks(rotation=90)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join('results', configs['data']['filename'].replace(".", "_")))
    # plt.show()
    #
    # # 创建一个 DataFrame
    # dataf = pd.DataFrame(datad, columns=['数据统计日', '真实运维次数', '预测运维次数'])
    # # dataf = np.concatenate(dataf, predictions_values)
    # # 将数据保存到 CSV 文件
    # dataf.to_csv(os.path.join('results', configs['data']['filename']), index=False)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], configs['data']['filename'])
    plot_results(predictions, y_test, configs['data']['filename'])


if __name__ == '__main__':
    main()
    # use_model()
