import os
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from .wavutils import load_tensor

def sep_filename_and_ext(file):
    ## Will fail if there is a '.' in the filename/pathname
    file_sep= file.split('.')
    ext = file_sep[len(file_sep)-1]
    filename = file_sep[:len(file_sep)-1][0]
    return(filename,ext)

def img_generate(height,width,tab,mode="PIL"):
    #generates an image from a rgb tab
    if mode=="PIL":
        img = Image.new('RGB', (width,height))
        for i in range(width):
            for j in range(height):
                value=tab[j,i]
                img.putpixel((i,j),tuple(value))
        img.show()

    if mode=="CV2":
        image = np.zeros((height,width,3), np.uint8)
        for i in range(width):
            for j in range(height):
                image[j,i]=tab[j,i]
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def transform_tab(tab):
    #transforms a binary tab in a rgb tab
    # tab = np.array([[[255*int(i),255*int(i),255*int(i)]] for i in tab])
    tab = np.array([[[255*int(i),0,0]] for i in tab])
    return tab

def visualize_one_recording(directory,display=False):
    ## Pour visualiser les fichiers de deux minutes tels que préparés
    tab_fish = np.zeros(shape=(120))
    tab_click = np.zeros(shape=(120))
    ind=0
    for root,dirs,files in os.walk(directory):
        for file in files:
            filename,ext = sep_filename_and_ext(file)
            if (ext!='csv'):
                continue
            if(filename=='labels'):
                continue
            if(filename=='output_labels'):
                continue
            if(filename=='train'):
                continue
            if(filename=='val'):
                continue
            tmp_file = pd.read_csv(os.path.join(directory,file))
            # print(tmp_file)
            tab_fish[ind]=tmp_file['fish_activity']
            tab_click[ind]=tmp_file['click_activity']
            ind +=1
        break
    # print('\n     ### Pourcentages d\'activité dans l\'extrait de deux minutes : #### \n')
    fish_percent = sum(tab_fish)*100/120
    click_percent = sum(tab_click)*100/120
    if display :
        print('Activité poisson : {} %'.format(fish_percent))
        print('Activité cliquetis : {} %'.format(click_percent))
        tab_fish_visu = transform_tab(tab_fish)
        tab_click_visu = transform_tab(tab_click)
        img_generate(120,1,tab_fish_visu)
        img_generate(120,1,tab_click_visu)
    return fish_percent,click_percent,tab_fish,tab_click

def plot_activity(table,axe,fig,day_length):
    newtab=[]
    tmp_tab=np.zeros(day_length)
    ind=0
    k=0
    while ind < len(table):
        while k<day_length and ind < len(table):
            tmp_tab[k]=table[ind]
            ind+=1
            k+=1
        k=0
        newtab.append(tmp_tab)
        tmp_tab=np.zeros(day_length)
    c=axe.pcolor(newtab)
    fig.colorbar(c,ax=axe)

def sum(tab):
    sum=0
    for i in tab:
        sum +=i
    return sum

def visualize_all_recordings(directory):
    tab_fish_visu = []
    tab_click_visu = []
    count=0
    for root,dirs,files in os.walk(directory):
        for dir in dirs:
            count+=1
            fish_percent,click_percent,tab_fish,tab_flick = visualize_one_recording(os.path.join(root,dir))
            tab_fish_visu.append(fish_percent)
            tab_click_visu.append(click_percent)
        break
    print('number of recordings : {}'.format(count))
    print('average percentage of fish activity on these recordings :{}'.format(sum(tab_fish_visu)/count))
    print('average percentage of fish activity on these recordings :{}'.format(sum(tab_click_visu)/count))
    fig,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(tab_fish_visu)
    ax1.set_title('percentage of fish activity for each 2 min. samples')
    plot_activity(tab_fish_visu,ax2,fig,142)
    plot_activity(tab_click_visu,ax3,fig,142)
    fig.tight_layout()
    plt.show()


def visualize_one_tensor(tensor_path,title=None,label=None):
    tensor=load_tensor(tensor_path,(51,96))
    axes = librosa.display.specshow(tensor.T)
    plt.colorbar(format='%+2.0f dB')
    if title is not None:
        plt.title(title)
    if label is not None:
        plt.text(0,-4,label)
    plt.show()
    # print(tensor)

def visualize_all_from_labels(csv_path,
                              filenames='filename',
                              column='fish_activity'):
    labels = pd.read_csv(csv_path)
    df = labels[[filenames,column]]
    ## extract the datas file per file using filenames
    size = len(df)
    current_filename = df.iloc[0,0]
    current_filename=current_filename.split('_')[0]
    ## current_filename : name of the original 2min. recording we're working on
    ind = 0
    tmp_tab = []
    pct_tab=[]
    while(ind < size):
        tmp_filename = df.iloc[ind,0].split('_')[0]
        if tmp_filename==current_filename:
            tmp_tab.append(df.iloc[ind,1])
        else:
            pct_tab.append(sum(tmp_tab)/len(tmp_tab))
            current_filename = tmp_filename
            tmp_tab=[]
        ind +=1
    fig,(ax1,ax2,) = plt.subplots(2,1)
    ax1.plot(pct_tab)
    ax1.set_title('percentage of fish activity for each 2 min. samples')
    plot_activity(pct_tab,ax2,fig,144)
    fig.tight_layout()
    plt.show()

def retrieve_day(timestamp,format=2019):
    if format==2019:
        date = timestamp.split('T')[0]
        day = int(date[6:])
    if format==2018:
        day = int(timestamp[0:2])
    return day



def retrieve_time_index(timestamp,format=2019):
    if format==2019:
        time = timestamp.split('T')[1]
        hour = int(time[0:2])
        min = int(time[2:4])
    if format==2018:
        hour = int(timestamp[2:4])
        min = int(timestamp[4:6])
    indice = int(6*hour + np.floor(min/10))
    return indice

def visualize_all_from_labels_per_day(csv_path,
                                      filenames='filename',
                                      column='fish_activity',
                                      records_per_day=144,
                                      format=2019):
    labels = pd.read_csv(csv_path)
    df = labels[[filenames,column]]
    ## extract the datas file per file using filenames
    size = len(df)
    number_of_days = int(size/(records_per_day*120))+1
    current_filename = df.iloc[0,0]
    current_filename=current_filename.split('_')[0]
    ## current_filename : name of the original 2min. recording we're working on
    initial_day=retrieve_day(current_filename,format)
    #the day of the month the record started - should not be possible to find any previous day in the data
    ind = 0
    tmp_tab = []
    pct_tab=np.zeros((number_of_days,records_per_day))
    while(ind < size):
        tmp_filename = df.iloc[ind,0].split('_')[0]
        tmp_indice = int(df.iloc[ind,0].split('_')[1])
        if tmp_filename==current_filename:
            tmp_tab.append(df.iloc[ind,1])
        else:
            day = retrieve_day(tmp_filename,format) - initial_day
            time_index = retrieve_time_index(tmp_filename,format)
            pct_tab[day][time_index]=(sum(tmp_tab)/len(tmp_tab))
            current_filename = tmp_filename
            tmp_tab=[]
        ind +=1
    pct_tab = pct_tab.reshape(number_of_days*records_per_day)
    fig,(ax1,ax2,) = plt.subplots(2,1)
    ax1.plot(pct_tab)
    ax1.set_title('percentage of fish activity for each 2 min. samples')
    plot_activity(pct_tab,ax2,fig,records_per_day)
    ax2.set_title(' activity per day visualisation')
    ax2.set_ylabel('day')
    ax2.set_xlabel('hour')
    #setting labels
    ticks = np.arange(0,records_per_day,int(4*records_per_day/(24)))
    labels = list(range(ticks.size))
    labels = [4*i for i in labels]
    ax2.set_autoscale_on(False)
    plt.sca(ax2)
    plt.xticks(ticks,labels)
    # ax2.set_xticklabels(range(24),np.arange(0,24))
    # ax2.set_xticklabels(range(4),['0','8','16','24'])
    fig.tight_layout()
    plt.show()
