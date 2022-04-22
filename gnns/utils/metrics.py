import torch
from sklearn.metrics import f1_score


def micro_macro_f1_score(logits, labels):
    """计算Micro-F1和Macro-F1得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float Micro-F1和Macro-F1得分
    """
    prediction = torch.argmax(logits, dim=1).long().cpu().numpy()
    labels = labels.numpy()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1
