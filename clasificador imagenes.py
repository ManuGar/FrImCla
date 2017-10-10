import argparse
import glob
import csv
import shutil, os



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=False, help="Path to the directory that contains our dataset",
                    default="melanoma/")
args = vars(ap.parse_args())

with open('melanoma_Training_Part3_GroundTruth.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        file, is_mel, _ = row
        print file
        if row[1]=='1.0':
            shutil.move("melanoma/"+file+".jpg", "melanoma/pos")
            shutil.move("melanoma/"+file+"_superpixels.png", "melanoma/pos")
        else:
            shutil.move("melanoma/"+file+".jpg", "melanoma/neg")
            shutil.move("melanoma/"+file+"_superpixels.png", "melanoma/neg")

'''
for path in glob.glob(args["disks"] + "/*.tif"):
'''