from ensemble_boxes import *
import pandas as pd

def main():
    lower = 0.2
    upper = 1.0
    num_detection = 100000
    path = input('csv file path :')
    num_detection = int(input('number of detections : '))
    lower = float(input('lower : '))
    upper = float(input('upper : '))
    df1 = pd.read_csv(path)
    
    df_confidence_list = []
    for string in df1.PredictionString:
        df_confidence_list += list(map(float, str(string).split(' ')[1::6]))
    MIN = min(df_confidence_list)
    MAX = max(df_confidence_list)
    thresh = sorted(df_confidence_list, reverse=True)[num_detection]

    ############################################ confidence 걸러내기
    ## confidence thresh 이상만 걸러내기

    string_list = []

    for string in df1.PredictionString:
        arr = list(map(str,string.split(' ')))
        row = []
        for index in range(0, len(arr)-1, 6):
            if float(arr[index+1])>thresh:
                row += arr[index:index+6]
            else:
                continue
                
        string_list.append(' '.join(row) +' ')
    df1.PredictionString = string_list

    ########################################### scaling 

    string_list = []
    for string in df1.PredictionString:
        arr = string.split(' ')
        arr[1::6] = [str(lower + ((upper-lower)*(float(i) -MIN) / (MAX-MIN)))  for i in arr[1::6]]
        string_list.append(' '.join(arr))

    df1.PredictionString =string_list
    df1.to_csv('/opt/ml/calibrated/new_calibrated.csv', index=None)
    
    print("Done! csv file created\n")

if __name__ == '__main__':
    main()