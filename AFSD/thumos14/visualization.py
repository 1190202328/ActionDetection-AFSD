import json
import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.transform import resize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits import axisartist

to_visualize_name = {
    'video_test_0000006': [19, 18.8, 57.3, 'VolleyballSpiking', 564, 1719],  # 类别id，开始秒数，结束秒数，类别名称，开始帧，结束帧
    'video_test_0000011': [15, 0.4, 12.4, 'Shotput', 12, 372],
    'video_test_0000058': [3, 1.9, 26.6, 'CleanAndJerk', 57, 798],
    'video_test_0000113': [9, 60.8, 79, 'GolfSwing', 1824, 2370],
}


def load_results(prediction_filename):
    with open(prediction_filename, 'r') as fobj:
        data = json.load(fobj)
    return data['results']


def load_label2id(filename):
    label2id = {}
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            id, label = line.split()
            label2id[label] = int(id)
    return label2id


def get_data():
    num_pred = 200
    output_dir = '../../visualization_figures'
    video_dir = '../../datasets/thumos14/visualization'
    if os.path.exists(output_dir):
        for name in os.listdir(output_dir):
            os.remove(f'{output_dir}/{name}')
        os.removedirs(output_dir)
    os.makedirs(output_dir)

    results = load_results('../../output/thumos14_fusion.json')

    for vid in to_visualize_name:
        draw(results[vid], to_visualize_name[vid], f'{video_dir}/{vid}.mp4',
             f'{output_dir}/{vid}.jpg')
        print(f'{vid}.jpg generated!')


def draw(result, info, video_path, output_file):
    # 绘图
    total_num = 1000
    end = 100

    bound = (info[2] - info[1]) / 3
    factor = total_num / ((info[2] - info[1]) + 2 * bound)
    left = info[1] - bound
    right = info[2] + bound
    label_name = info[3]
    #
    x = np.linspace(0, end, total_num)
    y_onset = np.zeros(shape=x.shape)
    y_offset = np.zeros(shape=x.shape)
    for i in range(len(result)):
        onset, offset = [float(i) for i in result[i]['segment']]
        score = float(result[i]['score'])
        label = result[i]['label']
        if left <= onset <= right and label == label_name:
            y_onset[int((onset - left) * factor)] += score
        if left <= offset <= right and label == label_name:
            y_offset[int((offset - left) * factor)] += score
    # 1.画最上面的图片
    fig = plt.figure(figsize=(20, 8))

    ax = axisartist.Subplot(fig, 211)
    fig.add_axes(ax)
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
        onset_frame = None
        mid_frame = None
        offset_frame = None
        height, width = 180, 320
        for num, im in enumerate(video):
            if num == info[4]:
                onset_frame = np.asarray(resize(im, (height, width)))
            if num == int((info[4] + info[5]) / 2):
                mid_frame = np.asarray(resize(im, (height, width)))
            if num == info[5]:
                offset_frame = np.asarray(resize(im, (height, width)))
        side_margin = np.ones(shape=(height, int(width * 0.5), 3))
        mid_margin = np.ones(shape=(height, int(width * 0.2), 3))
        frame = np.concatenate([side_margin, onset_frame, mid_margin, mid_frame, mid_margin, offset_frame, side_margin],
                               axis=1)

        ax.axis["bottom"].set_visible(False)
        ax.axis["left"].set_visible(False)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        ax.imshow(frame)
    except FileNotFoundError:
        pass

    # 2.画下面的regression
    ax = axisartist.Subplot(fig, 212)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    x_set = MultipleLocator(end / total_num * 50)  # x轴的刻度
    # 设置刻度间隔
    ax.xaxis.set_major_locator(x_set)
    gt_x_onset = np.ones(shape=(10,)) * (info[1] - left) / ((info[2] - info[1]) + 2 * bound) * end
    gt_x_offset = np.ones(shape=(10,)) * (info[2] - left) / ((info[2] - info[1]) + 2 * bound) * end
    max_y = max(max(y_onset), max(y_offset)) * 1.1
    gt_y = np.linspace(0, max_y, 10)
    ax.plot(gt_x_onset, gt_y, c='black', linestyle='--')
    ax.plot(gt_x_offset, gt_y, c='black', linestyle='--')
    ax.bar(x, y_onset, color='steelblue', label='Predict Onset')
    ax.bar(x, y_offset, color='coral', label='Predict Offset')
    # # 设置图表参数
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    # 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')

    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.5)


get_data()
