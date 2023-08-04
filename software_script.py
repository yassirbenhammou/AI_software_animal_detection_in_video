#!/usr/bin/env python
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import transforms, models
import os
import sys
import tarfile
import zipfile
from collections import defaultdict
import io
from io import StringIO
import cv2 as cv2
import humanfriendly
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import json
from PIL import Image
import csv
import io
import PySimpleGUI as sg
import os.path
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

######################################### Model dependencies #############################################################
# load labels extracted from annotations to find at https://github.com/visipedia/inat_comp/tree/master/2021
with open('./inat2021_categories/categories_inat2021.json') as f:
  categories = json.load(f)

df_transformed = pd.read_json('./inat2021_categories/categories_inat2021_transformed.json')

model = None
use_gpu = False


def load_model():
    global model
    # TODO: Download pre-trained models from https://github.com/EibSReM/newt/tree/main/benchmark
    # TODO: adapt path to respective model
    model_weights_fp = './Model/inat2021_supervised_large_from_scratch.pth.tar'
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10000)
    checkpoint = torch.load(model_weights_fp, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()
    if use_gpu:
        model.cuda()

###########################################################################################################################


######################################### Prediction functions ##############################################################
def prepare_image(image, target_size):
    #print(type(image))
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)


def predict(image):
    labels=[]
    probabilities=[]
    data = {"success": False}
    image = prepare_image(image, target_size=(224, 224))

    preds = F.softmax(model(image), dim=1)
    # adapt number of results k as needed
    results = torch.topk(preds.cpu().data, k=6, dim=1)
    results = (results[0].cpu().numpy(), results[1].cpu().numpy())
    data['predictions'] = list()

    for prob, label in zip(results[0][0], results[1][0]):
        label_name = categories['categories'][label]['name']
        r = {"label": label_name, "probability": float(prob)}
        labels.append(label_name)
        probabilities.append(float(prob))
        data['predictions'].append(r)
        #print(r)

    return labels,probabilities
###########################################################################################################################



def process_frame_model(frame):
    Animal='No'
    start_time = time.time()
    #frame = cv2.resize(frame, (224, 224))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    labels,probabilities=predict(frame)
    id_max_score=np.argmax(probabilities)
    probability=probabilities[id_max_score]
    print(probability)
    max_score=probability
    if max_score>=0.5:
        class_name=labels[id_max_score]
        line_df=df_transformed[df_transformed.name == class_name]
        kingdom_class=line_df.kingdom.values[0]
        common_name_class=line_df.common_name.values[0]
        if (kingdom_class=="Animalia"):
            Animal='Yes'
            print(class_name)
            print(common_name_class)
            print(kingdom_class)
            print(max_score*100)
    else:
        class_name=None
        kingdom_class=None
        common_name_class=None
        max_score=0
    elapsed = time.time() - start_time
    return common_name_class,Animal,frame,class_name,max_score*100,elapsed



def get_dur(filename):
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = frame_count / fps
    minutes = int(seconds / 60)
    rem_sec = int(seconds % 60)
    summary=f" which has {int(fps)}fps, and a total duration of {minutes}:{rem_sec}"
    video.release()
    cv2.destroyAllWindows()
    return summary,fps,frame_count,frame_width,frame_height




def main(): 
    font = ("Arial", 13)
    sg.theme('DarkGreen')
    image = Image.open('./interface/logoLW.png')
    image.thumbnail((100, 100))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    head = [
            sg.Image(data=bio.getvalue(), key="-LWlogo-")
       ],
    file_list_column = [
        [
            sg.Text("Folders Browser"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
       ],
       [
            sg.Listbox(
               values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
        [sg.Button("Process selected video")],
        [sg.Button("Process all videos")]
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose a a folder that contains videos to process", key="-notif-", font=font)],
        [sg.Text(size=(100, 4), key="-PRINCIPAL-", font=font)],
        [sg.Text(size=(100, 4), key="-TOUT-", font=font)],
        [sg.Image(filename='',key='image')],
    ]

    # ----- Full layout -----
    layout = [
       [
            
            sg.Frame(layout=head, title='BY:'), 
            #sg.HSeparator(pad=(100,0)),  
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window("Animal Detection Software", layout)

    filename=None
    video_filename=None
    selected_one=None
    file_list = []
    previsualisation=False
    processing_one=False
    # Run the Event Loop
    #first this should be outside
    iterating=False
    Animal_found=False
    Animal_not_found=False
    Animal_summary=None
    Animal_image=None
    fnames = []
    vars = {}
    id_video_in_list=0
    processing_all=False
    processing_all_done=False
    processing_all_one=False
    while True:
        event, values = window.read(timeout=2)
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        # Folder name was filled in, make a list of files in the folder
        
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            #print(folder)
            try:
            # Get list of files in folder
                file_list = os.listdir(folder)
                folder_name=os.path.basename(folder)
            except:
                file_list = []
            fnames = [
                f
                for f in file_list
                if f.endswith((".MP4", ".AVI"))
                ]
            number_videos=len(file_list)
            window["-FILE LIST-"].update(fnames)
            window["-notif-"].update("")
        
        if event == "-FILE LIST-":  # A file was chosen from the listbox
            if(processing_one):
                print("You can't select a new file because a video is currently being processed")
            else:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                    )
                print("Selected "+filename)
                cap = cv2.VideoCapture(filename)
                selected_one=filename
                video_filename=values["-FILE LIST-"][0]
                previsualisation=True
        
        if event == "Process selected video":
            cap_process_one = cv2.VideoCapture(selected_one)
            summary,fps,frame_count,frame_width,frame_height=get_dur(selected_one)
            processing_one=True
            frame_counter=0

        if event == "Process all videos":
            if (len(file_list) == 0):
                window["-TOUT-"].update("Please select a folder that contains videos to process")
            else:
                print(fnames)
                processing_all=True
                #for word in fnames:
                #    vars[word + "_Boolean"] = False
                #vars[fnames[id_video_in_list] + "_Boolean"]=True
                #print(vars)
                #print(vars["Mouse urine.AVI_Boolean"])
                #print(vars[fnames[0] + "_Boolean"])
                #print(vars[fnames[1] + "_Boolean"])
                #print(vars[fnames[2] + "_Boolean"])
                processing_all=True
                df_csv_results = pd.DataFrame(columns = ['Video', 'Animal presence','Animal specie','common name','probability(%)'])

        if (processing_all):
            if id_video_in_list>=len(fnames):
                processing_all=False
                processing_all_one=False
                processing_all_done=True
                id_video_in_list=0
            else:    
                print('///////////'+str(id_video_in_list))
                path_file = os.path.join(values["-FOLDER-"], fnames[id_video_in_list])
                print(path_file)
                summary,fps,frame_count,frame_width,frame_height=get_dur(path_file)
                print('processing video '+str(id_video_in_list+1)+' out of '+str(len(fnames))+' called '+fnames[id_video_in_list] + '_Boolean')
                cap_process_one = cv2.VideoCapture(path_file)
                #print(vars[fnames[id_video_in_list] + "_Boolean"])
                processing_all_one=True
                processing_all=False
                frame_counter=0
                
                

        if (processing_all_one):
            summary,fps,frame_count,frame_width,frame_height=get_dur(path_file)
            window["-PRINCIPAL-"].update('processing video '+str(id_video_in_list+1)+' out of '+str(len(fnames))+' in the folder '+str(folder_name))
            window["-TOUT-"].update("This video: "+fnames[id_video_in_list]+summary+" ; This video has "+str(int(frame_count))+" frames of "+str(int(frame_width))+"x"+str(int(frame_height))+" pixels")
            ret_one, frame_one = cap_process_one.read()
            frame_counter=frame_counter+1
            if(frame_one is None):
                continue
            if frame_counter>0 or frame_counter==1:
                video_filename=fnames[id_video_in_list]
                frame_viz = cv2.resize(frame_one, (int((frame_one.shape[1])/2), int((frame_one.shape[0])/2)))
                frame_viz = cv2.imencode('.png', frame_viz)[1].tobytes()
                frame_one1 = cv2.imencode('.png', frame_one)[1].tobytes()
                imgbytes = Image.open(io.BytesIO(frame_one1))
                inp = cv2.resize(frame_one, (frame_one.shape[1], frame_one.shape[0]))
                inp2 = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                common_name_class,Animal,inp,class_name,max_scor,elapsed=process_frame_model(imgbytes)
                window["-TOUT-"].update("Processing frame "+str(frame_counter)+"/"+str(int(frame_count))+" in the selected video: "+video_filename+" took "+str(elapsed)+" seconds")
                window['image'].update(data=frame_viz)
                if Animal=='Yes':
                    Animal_found=True
                    Animal_not_found=False
                    Animal_image=frame_viz
                    Animal_summary= "Found animal of specie: "+class_name+" also known as: "+common_name_class+" in frame: "+str(frame_counter)+" of video: "+video_filename+" with a probability of "+str(max_scor)+" %"
                    if not os.path.exists('./outputs'):
                        os.makedirs('./outputs')
                    if not os.path.exists('./outputs/All_videos_in_folder_'+str(folder_name)):
                        os.makedirs('./outputs/All_videos_in_folder_'+str(folder_name))
                    cv2.imwrite('./outputs/All_videos_in_folder_'+str(folder_name)+'/Video-'+str(video_filename)+'_Frame-'+str(frame_counter)+'_Animal-'+class_name+'_AKA-'+common_name_class+'.jpg', inp2)
                    df_csv_results = df_csv_results.append({'Video': video_filename, 'Animal presence': 'Yes', 'Animal specie': class_name, 'common name': common_name_class, 'probability(%)':max_scor}, ignore_index = True)
                    processing_all_one=False
                    processing_all=True
                    id_video_in_list=id_video_in_list+1
                    cap_process_one.release()
                if (frame_counter+fps>frame_count and Animal=='No'):
                    Animal_image=frame_viz
                    Animal_not_found=True
                    Animal_found=False
                    if not os.path.exists('./outputs'):
                        os.makedirs('./outputs')
                    if not os.path.exists('./outputs/All_videos_in_folder_'+str(folder_name)):
                        os.makedirs('./outputs/All_videos_in_folder_'+str(folder_name))
                    cv2.imwrite('./outputs/All_videos_in_folder_'+str(folder_name)+'/Video-'+str(video_filename)+'_No animal found.jpg', inp2)
                    df_csv_results = df_csv_results.append({'Video': video_filename, 'Animal presence': 'No', 'Animal specie': class_name, 'common name': common_name_class, 'probability(%)':max_scor}, ignore_index = True)
                    processing_all_one=False
                    processing_all=True
                    id_video_in_list=id_video_in_list+1



        if (processing_all_done):
            window["-TOUT-"].update("All videos in this folder were processed")
            df_csv_results.to_csv('./outputs/All_videos_in_folder_'+str(folder_name)+'/csv_output_'+str(folder_name)+'.csv', index=False)
            processing_all_done=False
        

        if previsualisation: #second, this should be inside
            summary,fps,frame_count,frame_width,frame_height=get_dur(filename)
            window["-TOUT-"].update("Selected video: "+video_filename+summary+" ; This video has "+str(int(frame_count))+" frames of "+str(int(frame_width))+"x"+str(int(frame_height))+" pixels")
            ret, frame = cap.read()
            if(frame is None):
                continue
            frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
        
        if processing_one:
            if selected_one==None:
                window["-TOUT-"].update("Please select a video to process")
            else:
                previsualisation=False
                window["-TOUT-"].update("Processing video: "+video_filename+summary+" ; This video has "+str(int(frame_count))+" frames of "+str(int(frame_width))+"x"+str(int(frame_height))+" pixels")
                iterating=True
        
        if iterating:
            summary,fps,frame_count,frame_width,frame_height=get_dur(selected_one)
            processing_one=False
            ret_one, frame_one = cap_process_one.read()
            frame_counter=frame_counter+1
            if(frame_one is None):
                continue
            if frame_counter>0 or frame_counter==1: #. #frame_counter==70 or frame_counter==112
                frame_viz = cv2.resize(frame_one, (int((frame_one.shape[1])/2), int((frame_one.shape[0])/2)))
                frame_viz = cv2.imencode('.png', frame_viz)[1].tobytes()
                frame_one1 = cv2.imencode('.png', frame_one)[1].tobytes()
                imgbytes = Image.open(io.BytesIO(frame_one1))
                inp = cv2.resize(frame_one, (frame_one.shape[1], frame_one.shape[0]))
                inp2 = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                common_name_class,Animal,inp,class_name,max_scor,elapsed=process_frame_model(imgbytes)
                df_csv_results = pd.DataFrame(columns = ['Video', 'Animal presence','Animal specie','common name','probability(%)'])
                window["-TOUT-"].update("Processing frame "+str(frame_counter)+"/"+str(int(frame_count))+" in the selected video: "+video_filename+" took "+str(elapsed)+" seconds")
                window['image'].update(data=frame_viz)
                if Animal=='Yes':
                    Animal_found=True
                    Animal_not_found=False
                    Animal_image=frame_viz
                    Animal_summary= "Found animal of specie: "+class_name+" also known as: "+common_name_class+" in frame: "+str(frame_counter)+" of video: "+video_filename+" with a probability of "+str(max_scor)+" %"
                    if not os.path.exists('./outputs'):
                        os.makedirs('./outputs')
                    if not os.path.exists('./outputs/'+str(video_filename)):
                        os.makedirs('./outputs/'+str(video_filename))
                    cv2.imwrite('./outputs/'+str(video_filename)+'/Frame-'+str(frame_counter)+'_Animal-'+class_name+'_AKA-'+common_name_class+'.jpg', inp2)
                    df_csv_results = df_csv_results.append({'Video': video_filename, 'Animal presence': 'Yes', 'Animal specie': class_name, 'common name': common_name_class, 'probability(%)':max_scor}, ignore_index = True)
                    df_csv_results.to_csv('./outputs/'+str(video_filename)+'/csv_output_'+str(video_filename)+'.csv', index=False)
                    iterating=False
                    cap_process_one.release()
                if (frame_counter+fps>frame_count and Animal=='No'):
                    Animal_image=frame_viz
                    Animal_not_found=True
                    Animal_found=False
                    iterating=False
                    cap_process_one.release()
                    if not os.path.exists('./outputs'):
                        os.makedirs('./outputs')
                    if not os.path.exists('./outputs/'+str(video_filename)):
                        os.makedirs('./outputs/'+str(video_filename))
                    cv2.imwrite('./outputs/'+str(video_filename)+'/No animal found.jpg', inp2)
                    df_csv_results = df_csv_results.append({'Video': video_filename, 'Animal presence': 'No', 'Animal specie': class_name, 'common name': common_name_class, 'probability(%)':max_scor}, ignore_index = True)
                    df_csv_results.to_csv('./outputs/'+str(video_filename)+'/csv_output_'+str(video_filename)+'.csv', index=False)
        if Animal_not_found:
            window["-TOUT-"].update("No animal was found in the video: "+video_filename)
            cap_process_one.release()
            cv2.destroyAllWindows()
            Animal_not_found=False
        if Animal_found:
            window["-TOUT-"].update(Animal_summary)
            window['image'].update(data=Animal_image)
            cap_process_one.release()
            cv2.destroyAllWindows()
            Animal_found=False

load_model()
main()

