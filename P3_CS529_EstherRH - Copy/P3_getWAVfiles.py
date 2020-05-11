#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import path
from pydub import AudioSegment


# In[2]:


import os
from glob import glob


# In[3]:


list_of_files = glob("test/*")


# In[4]:


#print(list_of_files)


# In[5]:


len(list_of_files)


# In[ ]:


#src = "./train/00907299.mp3"
#dst = "./output/00907299.wav"
                                                           
#sound = AudioSegment.from_mp3(src)
#sound.export(dst, format="wav")


# In[6]:


for i in list_of_files:
    name = i.lstrip("test\\")
    src = './test/' + name
    name = name.replace('.mp3','')
    dst ='./test_wav/' + name + '.wav'
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    


# In[113]:


import wave
obj = wave.open('./output/00907299.wav','r')
print( "Number of channels",obj.getnchannels())
print ( "Sample width",obj.getsampwidth())
obj.close()


# In[ ]:


from scipy.io import wavfile
fs, data = wavfile.read('./output/00907299.wav')


# In[ ]:




