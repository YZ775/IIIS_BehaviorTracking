import os, tkinter, tkinter.filedialog, tkinter.messagebox

def output(folder):
    path = os.getcwd()
    movie_dat = os.path.join(path, "movie_list.dat")
    filew = open(movie_dat,"w")
    files = os.listdir(folder)
    files.sort()
    for file in files:
        if "mov" not in file and "mp4" not in file and "avi" not in file:
            continue
        filew.write(file + "\n")
    filew.close()

# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
tkinter.messagebox.showinfo('Motion Analyzer of Shock','Please select a folder.')
folder = tkinter.filedialog.askdirectory()

# 処理ファイル名の出力
tkinter.messagebox.showinfo('',folder)
# print(folder)

output(folder)