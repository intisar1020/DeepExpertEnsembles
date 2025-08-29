import os
import copy
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

__all__ = [
    'return_topk_args_from_heatmap',
    'heatmap',
    'barchart',
    'save_checkpoint',
    'calculate_matrix',
    'make_list_for_plots',
    'to_csv',
    'imshow',
    'get_cam'
]


def sort_by_value(dict_, reverse=False):
    dict_tuple_sorted = {
        k: v
        for k, v in sorted(dict_.items(), reverse=True, key=lambda item: item[1])
    }
    return dict_tuple_sorted


def return_topk_args_from_heatmap(matrix, n, cutoff_thresold=5, binary_=True):
    visited = np.zeros((n, n))
    tuple_list = []
    value_of_tuple = []
    dict_tuple = {}
    unique_class_set = set()
    for i in range(n):
        for j in range(n):
            if not visited[i][j] and not visited[j][i] and matrix[i][j] > 0:
                dict_tuple[f"{i}_{j}"] = matrix[i][j]
                visited[i][j] = 1
                visited[j][i] = 1

    dict_sorted = sort_by_value(dict_tuple, reverse=True)
    for k, v in dict_sorted.items():
        if v < cutoff_thresold:
            continue
        sub_1, sub_2 = k.split("_")
        sub_1, sub_2 = int(sub_1), int(sub_2)
        unique_class_set.add(sub_1)
        unique_class_set.add(sub_2)
        tuple_list.append([sub_1, sub_2])
        value_of_tuple.append(v)

    if binary_:
        return tuple_list, [], value_of_tuple

    new_tuple = tuple_list.copy()
    for keys in tuple_list:
        key1, key2 = keys
        temp1 = {key1, key2}
        temp2 = {key1, key2}
        for keys2 in tuple_list:
            if key1 in keys2:
                first_elem, second_elem = keys2
                temp1.add(first_elem)
                temp1.add(second_elem)

        for keys2 in tuple_list:
            if key2 in keys2:
                first_elem, second_elem = keys2
                temp1.add(first_elem)
                temp1.add(second_elem)
        if len(temp1) > 2 and (list(temp1) not in tuple_list and list(temp1) not in new_tuple):
            new_tuple.append(list(temp1))
        if len(temp2) > 2 and (list(temp2) not in tuple_list and list(temp2) not in new_tuple):
            new_tuple.append(list(temp2))

    superset_list = []
    for elem1 in new_tuple:
        is_superset = True
        for elem2 in new_tuple:
            if set(elem1) != set(elem2) and set(elem1).issubset(set(elem2)):
                is_superset = False
                break
        if is_superset:
            superset_list.append(elem1)
    return tuple_list, superset_list, dict_sorted


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

    return im, cbar


def barchart(dict_):
    keys_ = list(dict_.keys())
    values_ = list(dict_.values())
    y_pos = np.arange(len(keys_))
    plt.bar(y_pos, values_, align='center', alpha=0.5, color='tab:blue')
    plt.xticks(y_pos, keys_, rotation=45)
    plt.show()


def make_list_for_plots(lois, plot, indexes):
    lst = []
    for loi in lois:
        for index in indexes:
            ind = loi + index
            lst.append(ind)
            plot[ind] = []
    return plot, lst


def calculate_matrix(model, test_loader_single, num_classes, only_top2=True):
    model.eval()
    model.cuda()
    freqMat = np.zeros((num_classes, num_classes))
    for dta, target in test_loader_single:
        dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output = model(dta)
            output = F.softmax(output, dim=1)
            pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
            pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
            if only_top2:
                if pred2.cpu().numpy()[0] == target.cpu().numpy()[0]:
                    s = pred2.cpu().numpy()[0]
                    d = pred1.cpu().numpy()[0]
                    freqMat[s][d] += 1
                    freqMat[d][s] += 1
            else:
                if pred1.cpu().numpy()[0] != target.cpu().numpy()[0]:
                    s = pred1.cpu().numpy()[0]
                    d = target.cpu().numpy()[0]
                    freqMat[s][d] += 1
                    freqMat[d][s] += 1
    return freqMat


def imshow(img, f_name, f_name_no_text, fexpertpred=None, fexpertconf=None, frouterpred=None, frouterconf=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f_name_no_text)
    res = f"Experts prediction: {fexpertpred} : {fexpertconf}\nRouters prediction: {frouterpred}:{frouterconf}"
    plt.text(0, 0, res, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.savefig(f_name)
    plt.close()


def save_checkpoint(model_weights, base_path='checkpoint_experts', filename='checkpoint.pth.tar'):
    filepath = os.path.join(base_path, filename)
    torch.save(model_weights, filepath)
    print("******* Saved New PTH *********")


def to_csv(plots, name):
    data_ = pd.DataFrame.from_dict(plots)
    data_.to_csv(name, index=False)


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    for i in range(2):
        line = f'{probs[i]:.3f} -> {classes[idx[i].item()]}'
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])
    print(f'output CAM.jpg for the top1 prediction: {classes[idx[0].item()]}')
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv.applyColorMap(CAM, cv.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.5
    cv2.imwrite('cam.jpg', result)