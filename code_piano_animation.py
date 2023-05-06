#-----------------------------------------------------
#PIANO ANIMATION CODE
#Last Updated: Mar. 20, 2023
#-----------------------------------------------------


#-----------------------------------------------------
#IMPORT MODULES
import numpy as np
import mido
import matplotlib.pyplot as plt
import matplotlib
import ffmpeg
from midi2audio import FluidSynth
import time
from selenium import webdriver
import shutil
import os
#-----------------------------------------------------


#-----------------------------------------------------
#EXTRACT MIDI FILE INFO:
#Extract .mid file relevent info to numpy array
def mid_to_arr(piano_file_name, FPS=20):
    '''
    Convert .mid file info to numpy array for single track piano recording.
    '''
    mid = mido.MidiFile(piano_file_name + '.mid', clip=True) #acces file
    data = mid.tracks[0] #keep track info
    
    music_info = [] #setup list to be filled
    total_time_2 = 0 #setup total time counter    
    for msg in data: #iterate over each event in track
        total_time_2 += msg.time
        if msg.type == 'note_off' or msg.type == 'note_on': #only want rows with relevent notes info, msg.type: note_on, note_off, control_change, set_tempo, time_signature, key_signture, end_of_track
            if msg.type == 'note_off':
                check_note_on = 0
            elif msg.type == 'note_on':
                check_note_on = 1
            music_info.append([check_note_on, msg.note, msg.velocity, total_time_2]) #append track info to list
    
    #columns: note on/off, note #, velocity, time since start in strange units
    music_info = np.array(music_info) #convert list to numpy array

    time_between_last_event_and_end_of_track = data[-1].time #duration of last step
    total_time = music_info[-1,3] + time_between_last_event_and_end_of_track #total time in weird units
    time_per_ticks = mid.length/total_time #s per weird units
    frame_per_ticks = (time_per_ticks*FPS) #number of frames per weird units
    #modify music_info time to be the frame number (need to round to nearest int)
    music_info[:,3] = np.array(music_info[:,3]*frame_per_ticks,int)
    
    return music_info, mid.length
#-----------------------------------------------------

#-----------------------------------------------------
#CREATE 3D ARRAY FOR FRAMES

#Define some sort of RGB color continuum
def color_func(x,i):
    return max([0, min(1,3*abs(1-2*((x-i/3)%2))-1)])

#Create 3D array of info for all animation frames
def frame_info(music_info, real_time, FPS=20, size_i=100, size_f=2000,
                color_param=[1.077, -1, 0, 1, 1, 1], prenote_num_frames=10, note_length_frame=30, fade_to=0):
    #Number of Notes N X Number of Frames F
    #Layers: x position, y position, marker size, color tuple (R,G,B,alpha)

    #ARRAY OF UNIQUE NOTES:
    notes_sorted = np.sort(music_info[:,1])
    notes_no_repeat = [notes_sorted[0]]
    counter = 0
    for jj in range(np.shape(notes_sorted)[0]):
        if notes_no_repeat[counter] != notes_sorted[jj]:
            notes_no_repeat.append(notes_sorted[jj])
            counter += 1

    #Relevent variable
    N = np.shape(notes_no_repeat)[0] #number of notes
    F = int(FPS*real_time) #number of frames
    layers = 4
    frame_info_arr = np.empty((N,F,4), dtype=object)


    #NOTE LOCATION:
    y = np.random.uniform(low=0.15,high=0.65,size=N)
    y_pos = np.empty((N,F),dtype=float)
    x_pos = np.empty((N,F),dtype=float)
    for ii in range(F):
        y_pos[:,ii] = y
        x_pos[:,ii] = np.linspace(0,1,N)

    #COLOR AND SIZE:
    #RGB const., alpha not const.
    #size not const., depend if note played
    #put color into 3D array then collapse one dimension into a tuple (FxNx4 ==> FxN)
    color_3d = np.ones((N,F,4),dtype=float) #layer for R, G, B, Alpha
    norm_note_3d = np.linspace(0,1,N)
    color_R = np.empty((N),dtype=float)
    color_G= np.empty((N),dtype=float)
    color_B = np.empty((N),dtype=float)
    for ii in range(N):
        color_R[ii] = color_func(norm_note_3d[ii]*color_param[3],color_param[0])
        color_G[ii] = color_func(norm_note_3d[ii]*color_param[4],color_param[1])
        color_B[ii] = color_func(norm_note_3d[ii]*color_param[5],color_param[2])
    for ii in range(F):
        color_3d[:,ii,0] = color_R
        color_3d[:,ii,1] = color_G
        color_3d[:,ii,2] = color_B
    
    
    #Change marker size based on ON/OFF value
    #change marker position based on if played
    #change marker value based on if played
    marker_size = np.zeros((N,F),float) #zero defaut is not playing
    counter = 0
    kk = 0
    #rate_size = 0.4 #scaling value from velocity to number of frame
        
    while kk < F:
        if kk == music_info[counter,3]:
            if music_info[counter,0] == 1:
                index_note = np.where(notes_no_repeat == music_info[counter,1])
                #note_length_frame = 30 #int(music_info[counter,2]*rate_size) #number of frames to have note affect size, depends on the velocity of note
                #prenote_num_frames = 10
                
                if note_length_frame+kk > F: #makes sure no error when reaching end of piece
                    note_length_frame = F-kk
                elif kk-note_length_frame <= 0:
                    note_length_frame = kk
                
                if prenote_num_frames < kk:
                    prenote_lenght_frame = kk
                
                if kk > prenote_num_frames:
                    marker_size[index_note[0],kk-prenote_num_frames:kk] = np.linspace(size_f,size_i,prenote_num_frames)
                    color_3d[index_note[0],kk-prenote_num_frames:kk,3] = np.linspace(0,0.2,prenote_num_frames)
                
                marker_size[index_note[0],kk:(kk+note_length_frame)] = np.linspace(size_i,size_f,note_length_frame)

                color_3d[index_note[0],kk:(kk+note_length_frame),3] = np.linspace(1,fade_to,note_length_frame)
            
            counter += 1
            kk -= 1
        kk += 1
        if counter == np.shape(music_info)[0]:
            break

    #convert to 2d color tuples
    color_tuple = np.empty((N,F),dtype=tuple)
    for ii in range(N):
        for jj in range(F):
            color_tuple[ii,jj] = tuple(color_3d[ii,jj])
            
    return x_pos, y_pos, marker_size, color_tuple, F
#-----------------------------------------------------


#-----------------------------------------------------
#PLOT RESULTS FOR ANIMATION
def make_ani(FPS, x_pos, y_pos, marker_size, color_tuple, F, marker='o', plot=True, full_marker=True, edge_width=6):
    #FPS = 20

    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    ax.set_facecolor("Black")
    sc = ax.scatter(x_pos[:,0],y_pos[:,0], marker=marker, linewidth=edge_width) #set up the plot
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xticks([])
    plt.yticks([])

      
    def animate(ii):
        if full_marker == True:
            sc.set_edgecolors('none')
            sc.set_facecolors(color_tuple[:,ii])
        else:
            sc.set_edgecolors(color_tuple[:,ii])
            sc.set_facecolors('none')
        
        sc.set_sizes(marker_size[:,ii])
        sc.set_offsets(np.c_[x_pos[:,ii],y_pos[:,ii]])

    ani = matplotlib.animation.FuncAnimation(fig, animate, 
                    frames=F, interval=1000/FPS, repeat=True) 
    
    if plot == True:
        #manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.show() #show the animation
    return ani
#-----------------------------------------------------

#-----------------------------------------------------
#SAVE THE ANIMATION: #very slow
def save_ani(FPS, ani, piano_file_name):
    # saving to mp4 using ffmpeg writer
    writervideo = matplotlib.animation.FFMpegWriter(fps=FPS)
    ani.save(piano_file_name + '.mp4', writer=writervideo, dpi=300, savefig_kwargs={'facecolor':'k'})
    plt.close()
    return
#-----------------------------------------------------

#-----------------------------------------------------
#Merge mp3 to mp4 file #VERY SLOW
def merge(piano_file_name):
    infile1 = ffmpeg.input(piano_file_name + '.mp4')
    infile2 = ffmpeg.input(piano_file_name.replace('_', '-') + '.mp3')

    ffmpeg.concat(infile1, infile2, v=1, a=1).output(piano_file_name + "_Audio_Video.mp4").run()
    return
#-----------------------------------------------------

#-----------------------------------------------------
#Convert midi file to mp3 file: (Using ZamZar accessing website with selenium, need Chrome)
def mid_to_mp3(piano_file_name, v=False):
    url = 'https://www.zamzar.com/convert/midi-to-mp3/' # URL of website

    options = webdriver.ChromeOptions()

    if v == False:
        options.add_argument('headless') #stop page from being displayed

    driver = webdriver.Chrome(options=options) #use Chrome to search url
    driver.get(url) # Opening the website
    time.sleep(2) #keep page open for 2s

    if v == True:
        print(driver.current_url)

    file_input = driver.find_elements('xpath', '//input[@type="file"]')
    while len(file_input) < 2:
        if v == True:
            print('len(file_input):', len(file_input)) 
        time.sleep(2)
        file_input = driver.find_elements('xpath', '//input[@type="file"]')

    time.sleep(2)

    full_mid_path = os.path.join(os.getcwd(), piano_file_name) + '.mid' #get full file location
    file_input[1].send_keys(full_mid_path)
    time.sleep(5)

    convert_button = driver.find_element('id', 'convert')      
    convert_button.click()
    time.sleep(15)

    #moved to new url when pressing convert
    new_url = driver.current_url
    driver.get(new_url)

    if v == True:
        print(driver.current_url)


    time.sleep(5)
    button = driver.find_elements('xpath', '//td')

    if v == True:
        print(button[0].get_attribute('outerHTML'))

    button[0].click() #press download button
    time.sleep(5)

    #move file from downloads:
    download_folder = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Downloads')
    download_full_path = os.path.join(download_folder, piano_file_name) + '.mp3'
    
    shutil.move(download_full_path, os.path.join(os.getcwd(), piano_file_name) + '.mp3')

    if v == True:
        print('File ', os.path.join(os.getcwd(), piano_file_name) + '.mp3', ' created')
    
    return

#-----------------------------------------------------
#Increase volume of mp3 file:
def mp3_volume_change_old(piano_file_name, v=True):
    #open volume change website:
    url = 'https://www.onlineconverter.com/increase-mp3-volume' # URL of website

    options = webdriver.ChromeOptions()

    if v == False:
        options.add_argument('headless') #stop page from being displayed

    driver = webdriver.Chrome(options=options) #use Chrome to search url
    driver.get(url) # Opening the website
    time.sleep(3) #keep page open for 3s

    if v == True:
        print(driver.current_url)

    file_input = driver.find_elements('xpath', '//input[@type="file"]')
    
    time.sleep(2)

    full_mp3_path = os.path.join(os.getcwd(), piano_file_name) + '.mp3' #get full file location
    file_input[0].send_keys(full_mp3_path)
    time.sleep(3)
    
    #select Volume change dropdown
    #NO, default is Increase 100% ==> Double sound by default which is good.
    
    #press convert button:
    convert_button = driver.find_elements('id', 'convert-button')
    convert_button[0].click()
    time.sleep(10) #wait for file to be converted
    
    new_url = driver.current_url
    driver.get(new_url)

    if v == True:
        print(driver.current_url)
    
    time.sleep(2)
    driver.refresh()
    time.sleep(2)
    
    #download file:
    convert_message = driver.find_elements('id', 'convert-message')
    convert_message[0].click()
    time.sleep(5)
    
    #sleep for a second
    time.sleep(2)

    #accept the alert
    #alert.accept()
    
    
    return

#Increase volume of mp3 file:
def mp3_volume_change(piano_file_name, v=True):
    #open volume change website:
    url = 'https://mp3cut.net/change-volume' # URL of website

    options = webdriver.ChromeOptions()

    if v == False:
        options.add_argument('headless') #stop page from being displayed

    driver = webdriver.Chrome(options=options) #use Chrome to search url
    driver.get(url) # Opening the website
    time.sleep(2) #keep page open for 3s

    if v == True:
        print(driver.current_url)

    
    file_input = driver.find_elements('xpath', '//input[@type="file"]')
    time.sleep(1)

    
    full_mp3_path = os.path.join(os.getcwd(), piano_file_name) + '.mp3' #get full file location
    file_input[0].send_keys(full_mp3_path) #load file to website
    time.sleep(3)
    
    
    #select Volume change button
    volume = driver.find_elements('class name', 'val-out')
    print(volume)
    print(volume[0])
    
    for ii in range(len(volume[0].__dir__())):
        print(volume[0].__dir__()[ii])
    
    print('\n')
    print(volume[0].get_attribute)
    #time.sleep(1)


    '''
    #press convert button:
    convert_button = driver.find_elements('id', 'convert-button')
    convert_button[0].click()
    time.sleep(10) #wait for file to be converted
    
    new_url = driver.current_url
    driver.get(new_url)

    if v == True:
        print(driver.current_url)
    
    time.sleep(2)
    driver.refresh()
    time.sleep(2)
    
    #download file:
    convert_message = driver.find_elements('id', 'convert-message')
    convert_message[0].click()
    time.sleep(5)
    
    #sleep for a second
    time.sleep(2)

    #accept the alert
    #alert.accept()
    '''
    
    return


#-----------------------------------------------------
#define input variables:
piano_file_name = 'Musette_in_D_Major'
np.random.seed(1234)

#Run the above functions:
music_info, real_time = mid_to_arr(piano_file_name)

x_pos, y_pos, marker_size, color_tuple, F = frame_info(music_info, real_time, 
                                                        prenote_num_frames=0, note_length_frame=7,
                                                        size_i=10000, size_f=10000, fade_to=0.6)

ani = make_ani(20, x_pos, y_pos, marker_size, color_tuple, F, marker='s', plot=False, full_marker=True)

save_ani(20, ani, piano_file_name) #SLOW

mid_to_mp3(piano_file_name, v=False) #~30s

#mp3_volume_change(piano_file_name, v=False) #To be developed

merge(piano_file_name) #SLOW




