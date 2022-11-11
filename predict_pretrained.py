# -*- coding: utf-8 -*-
from predict import predict_labels, predict_my_model
from wettbewerb import load_references, save_predictions

if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = load_references(
        './test/')  # 导入心电图文件、相关诊断、采样频率 (Hz) 和名称                                                #采样频率 300 Hz

    # predictions = predict_labels(ecg_leads, fs, ecg_names, use_pretrained=True)
    predictions = predict_my_model(ecg_leads, fs, ecg_names, use_pretrained=True, is_binary_classifier=True,device="cuda:0",
                                    model="my_experiments/10_11_2022__18_35_26ECGCNN_M_physio_net_dataset_challange_two_classes/models/best_model.pt")

    save_predictions(predictions)  # 将预测保存在 CSV 文件中
