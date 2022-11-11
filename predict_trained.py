# -*- coding: utf-8 -*-
from predict import predict_labels, predict_my_model
from wettbewerb import load_references, save_predictions

if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = load_references(
        './test/')  # 导入心电图文件、相关诊断、采样频率 (Hz) 和名称                                                #采样频率 300 Hz

    predictions = predict_labels(ecg_leads, fs, ecg_names, use_pretrained=False)

    save_predictions(predictions)  # 将预测保存在 CSV 文件中
