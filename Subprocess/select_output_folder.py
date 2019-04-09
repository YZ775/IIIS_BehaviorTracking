import os, tkinter, tkinter.filedialog, tkinter.messagebox

def output(folder):
    path = os.getcwd()
    folder_path = os.path.join(path, "output_place.dat") # folder path がタイトルになっているdatの場所
    filew = open(folder_path,"w")
    filew.write(os.path.join(folder))
    filew.close()

# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
tkinter.messagebox.showinfo('Motion Analyzer of Shock','Please select an output folder.')
folder = tkinter.filedialog.askdirectory()

# 処理ファイル名の出力
# tkinter.messagebox.showinfo('',folder)
print(folder)

output(folder)
