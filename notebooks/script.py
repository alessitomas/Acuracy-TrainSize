import os
import pandas as pd
import numpy as np
import librosa

def textures(x, texture_len=5, texture_step=5):
    tx = []
    for i in range( int( (len(x)-texture_len)/texture_step) ):
        this_frame = x[int(i*texture_step) :int(i*texture_step + texture_len), :]
        this_texture = np.hstack( (np.mean(this_frame, axis=0),np.std(this_frame, axis=0)) )
        tx.append(this_texture)
   
    tx = np.array(tx)
    texture_vector = np.hstack( (tx.mean(axis=0), tx.std(axis=0)) )
    return texture_vector

def audio_to_vec_handcrafted_features(y):
    mfcc_ = librosa.feature.mfcc(y=y).T
    rms_ = librosa.feature.rms(y=y).T
    cent_ = librosa.feature.spectral_centroid(y=y).T
    flat_ = librosa.feature.spectral_flatness(y=y).T
    roll_ = librosa.feature.spectral_rolloff(y=y).T
   
    features = np.hstack ( (mfcc_, rms_, cent_, flat_, roll_) )
    features_d = np.diff(features)
    features_dd = np.diff(features_d)
   
    texture_vector = np.hstack( (textures(features, 30, 15), textures(features_d, 30, 15), textures(features_dd, 30, 15)) )
    return texture_vector
   


column_names = [f'feature_{i+1}' for i in range(276)]
column_names += ['track_id']
df = pd.DataFrame(columns=column_names)




directory_path = "../data/fma_large/"
arquivos_problematicos = []  

for root, dirs, files in os.walk(directory_path):
    for filename in files:
        if filename.endswith(".mp3"):
            try:
                file_path = os.path.join(root, filename)
                track_id = filename.split('.')[0].lstrip("0")
                y, sr = librosa.load(file_path, sr=22050)
                feats = audio_to_vec_handcrafted_features(y)
                new_row_data = list(feats) + [track_id]
                df.loc[len(df)] = new_row_data
            except Exception as e:  
                arquivos_problematicos.append(filename) 



df.to_csv('FMA_large_handcrafted_features.csv', index=False)
with open('my_file.txt', 'w') as file:
    
    for item in arquivos_problematicos:
        file.write(f"{item}\n")