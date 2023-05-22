## code source : https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/master

from ensemble_boxes import *
import pandas as pd

def get_value(text_list, weights, iou_thr=0.5, skip_box_thr=0.0001):
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for text in text_list:
        arr = str(text).split(' ')[:-1]
        labels = []
        scores = []
        boxes = []
        for i in range(len(arr)//6):
            labels.append(int(arr[6*i]))
            scores.append(float(arr[6*i+1]))
            boxes.append([float(i)/1024 for i in arr[6*i+2:6*i+6]])
        
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0001)

def main():
    n = int(input("num of csv files : "))
    text_list =[] 
    
    for i in range(n):
        df = pd.read_csv(input(f'csv file path {i}: '))
        text_list.append(df.PredictionString)

    weights = list(map(int ,input("weight input(e.g. 1 3 ) : ").split()))
    thresh = float(input("iou threshold(0~1) : "))
    print('please wait ...\n')

    string_list = []
    for i in range(len(text_list[0])):
        boxes, scores, labels = get_value([text[i] for text in text_list], weights= weights, iou_thr=thresh, skip_box_thr=0.0001)
        string =''
        for j in range(len(labels)):
            string += str(int(labels[j])) + ' ' + str(scores[j]) + ' ' + ' '.join([str(num*1024) for num in boxes[j]]) +' '
        string_list.append(string)

    df.PredictionString = string_list

    df.to_csv('wbf_ensemble.csv', index=None)
    print("Done! csv file created\n")

if __name__ == '__main__':
    main()