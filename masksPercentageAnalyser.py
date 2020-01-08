import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

data = {'ID': ['BRCA03', 'BRCA04', 'BRCA05', 'BRCA06', 'BRCA09', 'BRCA16', 'BRCA37', 'BRCA40', 'BRCA46', 'BRCA47',
               'BRCA58', 'BRCA59', 'BRCA60', 'BRCA62', 'BRCA64', 'BRCA65', 'BRCA67', 'BRCA70', 'BRCA71', 'BRCA72',
               'BRCA80', 'BRCA82', 'BRCA87', 'BRCA89'],
        'Tumor': [250428, 19925, 20235, 187796, 279172, 165729, 215407, 76208, 46921, 148481, 229936, 7154, 3544,
                  205512, 0, 104321, 0, 66347, 8626, 53248, 279228, 33979, 4117, 96326, 130163],
        'Stroma': [0, 29348, 2054, 67738, 15313, 17543, 53515, 1536, 30120, 12166, 24220, 27614, 6426, 4160, 43978,
                   18438, 6690, 29480, 5593, 16441, 21786, 5387, 5122, 37850, 9801],
        'Immune': [110984, 74098, 27356, 37212, 63876, 19478, 17776, 6501, 4560, 11614, 49521, 12187, 50400, 55945,
                   61929, 4542, 35259, 42900, 135778, 16659, 9057, 60479, 335474, 43014],
        'Others': [8123, 3029, 0, 1092, 3401, 695, 0, 188, 14759, 2966, 5375, 0, 2806, 7703, 6128, 0, 2224, 34272, 0,
                   22289, 2295, 2434, 1620, 23007, 0]},

resultsExcel = {}
df_resultXsl = pd.DataFrame(
    columns=['ID', 'Tumor', 'Stroma', 'Immune', 'Others', 'FileName', 'ImmuneArea', 'ImmuneAvgAnnotations', 'TumorArea',
             'TumorAvgAnnotations', 'StromaArea', 'StromaAvgAnnotations', 'OthersArea', 'OthersAvgAnnotations', 'TumorRatio', 'StromaRatio', 'ImmuneRatio', 'OthersRatio','TumorContours', 'StromaContours', 'ImmuneContours', 'OthersContours'])
# df_resultXsl = pd.DataFrame(columns=[  'ID' , 'TumorArea' , 'StromaArea' , 'ImmuneArea' , 'OthersArea'])

annotation_dataframe = pd.DataFrame(data)
indexDataFrame = -1
maskDirPath = '/home/maorvelous/Documents/Lab/deepLearning/MaskLou/'

idSample = 000000
entries = os.listdir(maskDirPath)
total_of_pixels = 88363080
annotatedColor = 255
annotations_types = ['Immune cells', 'Tumor', 'Stroma', 'Others']

def countWhiteOnMask(entry, type):
    print(type in entry)

    mask_Immune = cv2.imread(maskDirPath + entry, cv2.IMREAD_GRAYSCALE)

    print(entry)
    n_white_pix = np.sum(mask_Immune == annotatedColor)
    print("Number of white pixels of ", type, ":", n_white_pix)

    if type == "Immune cells":
        print("dataFrame Immune cell ", annotation_dataframe.iloc[0][3][indexDataFrame])
        averageCell = n_white_pix / annotation_dataframe.iloc[0][3][indexDataFrame]

    if type == "Tumor":
        print("dataFrame Immune cell ", annotation_dataframe.iloc[0][1][indexDataFrame])
        averageCell = n_white_pix / annotation_dataframe.iloc[0][1][indexDataFrame]

    if type == "Stroma":
        averageCell = n_white_pix / annotation_dataframe.iloc[0][2][indexDataFrame]

    if type == "Others":
        averageCell = n_white_pix / annotation_dataframe.iloc[0][4][indexDataFrame]

    print(" Average white pixel for annotations ", type, ": ", averageCell)

    percentage = n_white_pix / (total_of_pixels / 100)
    print(" Percentage number of Pixels : ", percentage)

    ret, thresh = cv2.threshold(mask_Immune, 120, annotatedColor, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    print(" Number of annotated Regions of ", type, ": ", len(contours))

    return n_white_pix

for entry in sorted(entries):

    Immune_n_white_pix = 0
    Tumor_n_white_pix = 0
    Stroma_n_white_pix = 0
    Others_n_white_pix = 0

    TumorRatio = 0
    ImmuneRatio = 0
    StromaRatio = 0
    OthersRatio = 0

    TumorContours = 0
    ImmuneContours = 0
    StromaContours = 0
    OthersContours = 0

    if "Immune" in entry:

        indexDataFrame = indexDataFrame + 1

        print("Index of the data frame : ", indexDataFrame)
        id = entry.replace("Immune cells_mask.png", "")
        Immune_n_white_pix = countWhiteOnMask(entry, 'Immune cells')
        tumorFileName = entry.replace('Immune cells', 'Tumor')
        Tumor_n_white_pix = countWhiteOnMask(tumorFileName, 'Tumor')

        mask_Immune = cv2.imread(maskDirPath + entry, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(mask_Immune, 120, annotatedColor, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        ImmuneContours = len(contours)
        ImmuneRatio = Immune_n_white_pix / len(contours)

        stromaFileName = entry.replace('Immune cells', 'Stroma')

        if os.path.isfile(maskDirPath + stromaFileName):
            Stroma_n_white_pix = countWhiteOnMask(stromaFileName, 'Stroma')

            mask_Stroma = cv2.imread(maskDirPath + stromaFileName, cv2.IMREAD_GRAYSCALE)
            ret, thresh = cv2.threshold(mask_Stroma, 120, annotatedColor, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            StromaContours = len(contours)
            StromaRatio = Stroma_n_white_pix / len(contours)

        othersFileName = entry.replace('Immune cells', 'Other')

        if os.path.isfile(maskDirPath + othersFileName):
            Others_n_white_pix = countWhiteOnMask(othersFileName, 'Others')

            mask_Others = cv2.imread(maskDirPath + stromaFileName, cv2.IMREAD_GRAYSCALE)
            ret, thresh = cv2.threshold(mask_Others, 120, annotatedColor, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            OthersContours = len(contours)
            OthersRatio = Others_n_white_pix / len(contours) if len(contours)!= 0 else 0

        # df_resultXsl.loc[indexDataFrame] = [entry, Immune_n_white_pix, Tumor_n_white_pix, Stroma_n_white_pix, Others_n_white_pix]
        print(indexDataFrame)
        avgTumorAnnotations = Tumor_n_white_pix / annotation_dataframe.iloc[0][1][indexDataFrame]
        avgImmuneAnnotations = Immune_n_white_pix / annotation_dataframe.iloc[0][3][indexDataFrame]
        avgStromaAnnotations = Stroma_n_white_pix / annotation_dataframe.iloc[0][2][indexDataFrame] if annotation_dataframe.iloc[0][2][indexDataFrame] != 0 else 0
        avgOthersAnnotations = Others_n_white_pix / annotation_dataframe.iloc[0][4][indexDataFrame] if annotation_dataframe.iloc[0][4][indexDataFrame] != 0 else 0


        mask_Immune = cv2.imread(maskDirPath + entry, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(mask_Immune, 120, annotatedColor, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        TumorContours = len(contours)
        TumorRatio = Tumor_n_white_pix / len(contours) if len(contours)!= 0 else 0

        df_resultXsl.loc[indexDataFrame] = [annotation_dataframe.iloc[0][0][indexDataFrame],
                                            annotation_dataframe.iloc[0][1][indexDataFrame],
                                            annotation_dataframe.iloc[0][2][indexDataFrame],
                                            annotation_dataframe.iloc[0][3][indexDataFrame],
                                            annotation_dataframe.iloc[0][4][indexDataFrame], entry, Immune_n_white_pix,
                                            avgImmuneAnnotations, Tumor_n_white_pix, avgTumorAnnotations,
                                            Stroma_n_white_pix, avgStromaAnnotations, Others_n_white_pix,
                                            avgOthersAnnotations, TumorRatio, StromaRatio, ImmuneRatio, OthersRatio, TumorContours, StromaContours, ImmuneContours, OthersContours]

print(df_resultXsl)
df_resultXsl.to_excel("output.xlsx")