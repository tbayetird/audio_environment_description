import pandas as pd

def csv_to_dic(csv_path):
    dic=pd.Series.from_csv(csv_path, header=0).to_dict()
    return dic

def custom_csv_to_dic(csv_path):
    df = pd.read_csv(csv_path)
    df_ind = df.set_index('filename')
    dic_tmp = df_ind.to_dict()
    # print(dic_tmp)
    dic = {}
    fish_dic = dic_tmp['fish_activity']
    # click_dic = dic_tmp['click_activity']
    for cle in fish_dic:
        # dic[cle]=[fish_dic[cle],click_dic[cle]]
        dic[cle]=[fish_dic[cle]]
    return dic

def custom_load_list_IDs(csv_path):
    df = pd.read_csv(csv_path)
    Ids = df['filename']
    return Ids
