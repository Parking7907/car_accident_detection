import scaper
import numpy as np
import os


# OUTPUT FOLDER / SCAPER Setting

outfolder = '../Train/'
fg_folder = '/home/data/AIHUB_sound/Train/foreground/'
bg_folder = '/home/data/AIHUB_sound/Train/background/'
n_soundscapes = 20000


ref_db = -40
duration = 10.0 

min_events = 1
max_events = 5

event_time_dist = 'truncnorm'
event_time_mean = 5.0
event_time_std = 2.0
event_time_min = 0.0
event_time_max = 10.0

source_time_dist = 'const'
source_time = 0.0

event_duration_dist = 'uniform'
event_duration_min = 0.5
event_duration_max = 4.0

snr_dist = 'uniform'
snr_min = 6
snr_max = 20

pitch_dist = 'uniform'
pitch_min = -3.0
pitch_max = 3.0

time_stretch_dist = 'uniform'
time_stretch_min = 1
time_stretch_max = 1
    
# Generate 1000 soundscapes using a truncated normal distribution of start times

for n in range(n_soundscapes):
    
    print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
    
    # create a scaper
    sc = scaper.Scaper(duration, fg_folder, bg_folder)
    sc.protected_labels = []
    sc.ref_db = ref_db
    
    # add background
    sc.add_background(label=('choose', []), 
                      source_file=('choose', []), 
                      source_time=('const', 0))

    # add random number of foreground events
    n_events = np.random.randint(min_events, max_events+1)
    for _ in range(n_events):
        sc.add_event(label=('choose', []), 
                     source_file=('choose', []), 
                     source_time=(source_time_dist, source_time), 
                     event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
                     event_duration=(event_duration_dist, event_duration_min, event_duration_max), 
                     snr=(snr_dist, snr_min, snr_max),
                     pitch_shift=(pitch_dist, pitch_min, pitch_max),
                     time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
    
    # generate
    audiofile = os.path.join(outfolder, "soundscape_unimodal{:d}.wav".format(n))
    jamsfile = os.path.join(outfolder, "soundscape_unimodal{:d}.jams".format(n))
    txtfile = os.path.join(outfolder, "soundscape_unimodal{:d}.txt".format(n))
    
    sc.generate(audiofile, jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=False,
                reverb=0.1,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile)


