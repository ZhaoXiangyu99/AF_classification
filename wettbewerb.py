# -*- coding: utf-8 -*-

from typing import List, Tuple
import csv
import scipy.io as sio
import numpy as np
import os


def load_references(folder: str = '../training') -> Tuple[List[np.ndarray], List[str], int, List[str]]:
    """
    Parameters
    ----------
    folder : str, optional
        训练数据的位置。默认值'../training'.
    Returns
    -------
    ecg_leads : List[np.ndarray]
        心电图信号.
    ecg_labels : List[str]
        ECG信号的label，包括: 'N','A','O','~'
    fs : int
        采样频率.
    ecg_names : List[str]
        加载文件的名称
    """
    # Check Parameter
    assert isinstance(folder, str), "Parameter folder must be string".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder  doesn\'t exist!'
    # 初始化ecg_leads,ecg_labels,ecg_names
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []
    # 设置采样频率
    fs: int = 300
    # 加载参reference文件
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # 遍历每一行
        for row in csv_reader:
            # 加载带有 ECG 导联和标签的 MatLab 文件
            data = sio.loadmat(os.path.join(folder, row[0] + '.mat'))
            ecg_leads.append(data['val'][0])
            ecg_labels.append(row[1])
            ecg_names.append(row[0])
    # 显示加载了多少数据
    print("加载了{}条数据.".format(len(ecg_leads)))
    return ecg_leads, ecg_labels, fs, ecg_names


### 危险！请勿更改此功能.

def save_predictions(predictions: List[Tuple[str, str, float]], folder: str = None) -> None:
    """
    将给定的预测保存到名为 PREDICTIONS.csv 的 CSV 文件中
    Parameters
    ----------
    predictions : List[Tuple[str, str,float]]
        List and Tuple，其中每个元组包含文件名和预测标签('N','A','O','~'），以及不确定性
        比如 [('train_ecg_03183.mat', 'N'), ('train_ecg_03184.mat', "~"), ('train_ecg_03185.mat', 'A'),
                  ('train_ecg_03186.mat', 'N'), ('train_ecg_03187.mat', 'O')]
	folder : str
		prediction的位置
    Returns
    -------
    None.
    """
    # Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions muss eine Liste sein aber {} gegeben.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions muss eine nicht leere Liste sein.'
    assert isinstance(predictions[0], tuple), \
        "Elemente der Liste predictions muss ein Tuple sein aber {} gegeben.".format(type(predictions[0]))
    assert isinstance(predictions[0][2], float), \
        "3. Element der Tupel in der Liste muss vom Typ float sein, aber {} gegeben".format(type(predictions[0][2]))

    if folder == None:
        file = "PREDICTIONS.csv"
    else:
        file = os.path.join(folder, "PREDICTIONS.csv")
    # 检查文件是否已经存在，如果存在则删除文件
    if os.path.exists(file):
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:
        # 初始化 CSV 写入器以写入文件
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # 迭代每个预测
        for prediction in predictions:
            predictions_writer.writerow([prediction[0], prediction[1], prediction[2]])
        # 显示信息保存了多少标签（预测）
        print("有{}条标签被保存.".format(len(predictions)))
