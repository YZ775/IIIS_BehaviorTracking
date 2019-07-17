import csv
import sys
import os
import re
import glob
#result txt file -> combine 1file

#出力フォルダのパスを取得
output_folder = open("output_place.dat")
path = output_folder.readlines()
print(path)

t_path = path[0]
true_path = t_path + "/*.txt"
print(true_path)

output_folder.close()

#出力フォルダのtxtをリストで取得
outtxt = glob.glob(true_path)
print(outtxt)

#txt file から，3行取得
csv_path = t_path + "/combine.csv"
with open(csv_path, "w") as csv_f:
    writer = csv.writer(csv_f)

    #初回書込み
    writer.writerow(["filename", "Sum of motion index Before", "Sum of motion index After", "A-B/A+B"])

    for file_path in outtxt:
        #filepathを変換
        real_path = file_path[file_path.rfind("\\")+1:]
        print(real_path)

        f = open(file_path)
        l = f.readlines()
        before = l[0]
        after = l[1]
        sum = l[2]
        csv_before = before.replace("Before_Shock Sum of Motion Index: ", "")
        csv_after = after.replace("After_Shock Sum of Motion Index: ", "")
        csv_sum = sum.replace("A-B/A+B: ", "")
        print("{} {} {}\n".format(csv_before, csv_after,csv_sum))

        writer.writerow([real_path, csv_before, csv_after, csv_sum])
        f.close()
