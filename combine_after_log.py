# coding:utf-8
# パスも自主取得
import os

def get_path():
    folder_path = ""

    # output先のパスが書かれたファイルを開く
    output_place = os.path.join("Subprocess", "output_place.dat")
    print(output_place)
    if not os.path.isfile(output_place):
        print("Output folder is not found.\nFiles will be saved in log_after.")
        folder_path = os.path.join("log_after")
    else:
        file_out = open(output_place,"r")
        folder_path = file_out.readline()
        folder_path.replace("\n","")
        file_out.close()
    if folder_path == "":
        print("Output folder is not found.\nFiles will be saved in log_after.")
        folder_path = os.path.join("log_after")

    return folder_path

def output_csv(folder_path):
    files = os.listdir(folder_path)
    log_files = []
    for file in files:
        if ".txt" in file and "after" in file:
            log_files.append(file)
    
    write_data = []
    for l_file in log_files:
        b_flag, a_flag, ab_flag = False, False, False
        b, a, ab = "", "", ""
        file_path = os.path.join(folder_path,l_file)
        fo = open(file_path, "r")
        lines = fo.readlines()
        for line in lines:
            if not b_flag and "Before_Shock" in line:
                b = line.split(":")[1]
                b = b.replace("\r","")
                b = b.replace("\n","")
                b_flag = True
            if not a_flag and "After_Shock" in line:
                a = line.split(":")[1]
                a = a.replace("\r","")
                a = a.replace("\n","")
                a_flag = True
            if not ab_flag and "A-B/A+B" in line:
                ab = line.split(":")[1]
                ab_flag = True
            if a_flag and b_flag and ab_flag:
                break
        str = l_file.replace(".txt", "") + "," + b + "," + a + "," + ab # マウス名 + b + a + ab
        write_data.append(str)
        fo.close()
    # print(write_data)

    out_file = "combined_after_log.csv"
    file_path = os.path.join(folder_path, out_file)
    fo = open(file_path, "w")
    fo.write(",Before_Shock Sum of Motion Index,After_Shock Sum of Motion Index,A-B/A+B\n")
    for w in write_data:
        fo.write(w)
    fo.close()
        

def main():
    path = get_path()
    output_csv(path)


if __name__ == "__main__":
    main()