import pandas as pd

def main():
    path = input("name of csv file : ")
    df= pd.read_csv(path)
    
    thresh = float(input("confidence threshold input(0~1) : "))
    
    ## confidence threshold 이상만 걸러내기
    string_list = []
    for string in df.PredictionString:
        arr = list(map(str,str(string).split(' ')))
        row = []
        for index in range(0, len(arr)-1, 6):
            if float(arr[index+1])>thresh:
                row += arr[index:index+6]
            else:
                continue
        string_list.append(' '.join(row) +' ')
    
    df.PredictionString = string_list
    df.to_csv(path[:-4]+'_thresh_'+str(thresh)+'.csv', index=False)

    print("Done! csv file created\n")
    print("file name : ",path[:-4]+'_thresh_'+str(thresh)+'.csv\n')

if __name__ == '__main__':
    main()