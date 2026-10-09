import tkinter as tk
from tkinter import filedialog
from PIL import Image

import glob
import os

root = tk.Tk()
root.withdraw()

# Select input path
input_folder = filedialog.askdirectory(title="Input folder containing the images",initialdir="/home/xrad/KI_Piktogramm/github/KI-Piktogramme/data/anonymisiert/Pat_mit_Piktogramm")
output_folder = filedialog.askdirectory(title= "Output folder where images are saved to", initialdir="/home/xrad/KI_Piktogramm/github/KI-Piktogramme/data/anonymisiert/Pat_mit_Piktogramm_cut")

# Choose data with different image section
extra_crop_files_zoom = [
    "anon_Pat23_5.jpg",
    
    "anon_Pat24_2.jpg",
    "anon_Pat24_4.jpg",
    
    "anon_Pat25_1.jpg",
    
    "anon_Pat88_1.jpg",
    "anon_Pat88_2.jpg",
    "anon_Pat88_3.jpg",
    "anon_Pat88_4.jpg",
    "anon_Pat88_5.jpg",
    
    "anon_Pat94_2.jpg",
    
    "anon_Pat95_1.jpg",
    "anon_Pat95_3.jpg",
    "anon_Pat95_4.jpg",
    "anon_Pat95_5.jpg",
    
    "anon_Pat96_1.jpg",
    "anon_Pat96_2.jpg",
    "anon_Pat96_3.jpg",
    "anon_Pat96_4.jpg",
    "anon_Pat96_5.jpg",
    
    "anon_Pat100_1.jpg",
    "anon_Pat100_2.jpg",
    "anon_Pat100_3.jpg",
    "anon_Pat100_4.jpg",
    "anon_Pat100_5.jpg",
    
    "anon_Pat130_1.jpg",
    "anon_Pat130_2.jpg",
    "anon_Pat130_3.jpg",
    "anon_Pat130_4.jpg",
    "anon_Pat130_5.jpg",
    
    "anon_Pat205_3.jpg",
    
    "anon_Pat213_1.jpg",
    "anon_Pat213_2.jpg",
    "anon_Pat213_3.jpg",
    "anon_Pat213_4.jpg",
    "anon_Pat213_5.jpg",
    
    "anon_Pat215_1.jpg",
    "anon_Pat215_2.jpg",
    "anon_Pat215_3.jpg",
    "anon_Pat215_4.jpg",
    "anon_Pat215_5.jpg",
    
    "anon_Pat220_1.jpg",
    "anon_Pat220_2.jpg",
    "anon_Pat220_3.jpg",
    "anon_Pat220_4.jpg",
    "anon_Pat220_5.jpg",
    
    "anon_Pat238_1.jpg",
    "anon_Pat238_2.jpg",
    "anon_Pat238_3.jpg",
    "anon_Pat238_4.jpg",
    "anon_Pat238_5.jpg",
    
    "anon_Pat246_3.jpg",
    
    "anon_Pat247_1.jpg",
    "anon_Pat247_2.jpg",
    "anon_Pat247_3.jpg",
    "anon_Pat247_4.jpg",
    "anon_Pat247_5.jpg",
    
    "anon_Pat265_1.jpg",
    "anon_Pat265_2.jpg",
    "anon_Pat265_3.jpg",
    "anon_Pat265_4.jpg",
    "anon_Pat265_5.jpg",
    
    "anon_Pat288_1.jpg",
    "anon_Pat288_2.jpg",
    "anon_Pat288_3.jpg",
    "anon_Pat288_4.jpg",
    "anon_Pat288_5.jpg",
    
    "anon_Pat289_1.jpg",
    "anon_Pat289_2.jpg",
    "anon_Pat289_3.jpg",
    "anon_Pat289_4.jpg",
    "anon_Pat289_5.jpg",
    
    "anon_Pat291_1.jpg",
    "anon_Pat291_2.jpg",
    "anon_Pat291_3.jpg",
    "anon_Pat291_4.jpg",
    "anon_Pat291_5.jpg",
    
    "anon_Pat298_3.jpg",
    "anon_Pat298_4.jpg"
  
 ]




extra_crop_files = [
    
   
#     "anon_0302_1r4_1.jpg",
#     "anon_0302_1r4_2.jpg",
#     "anon_0302_1r4_3.jpg",
#     "anon_0302_1r4_4.jpg",
#     "anon_23042026_1.jpg",
#     "anon_23042026_2.jpg",
#     "anon_23042026_3.jpg",
#     "anon_23042026_4.jpg",
#     "anon_220420268_1.jpg",
#     "anon_220420268_2.jpg",
#     "anon_220420268_3.jpg",
#     "anon_220420268_4.jpg",
#     "anon_230420267_1.jpg",
#     "anon_230420267_2.jpg",
#     "anon_230420267_3.jpg",
#     "anon_230420267_4.jpg",
#     "anon_230420267_5.jpg",
#     "anon_230420269_1.jpg",
#     "anon_230420269_2.jpg",
#     "anon_230420269_3.jpg",
#     "anon_230420269_4.jpg",
#     "anon_230420269_5.jpg",
#     "anon_280420269_1.jpg",
#     "anon_280420269_2.jpg",
#     "anon_280420269_3.jpg",
#     "anon_280420269_4.jpg",
#     "anon_280420269_5.jpg",
#     "anon_2304202677_1.jpg",
#     "anon_2304202677_2.jpg",
#     "anon_2304202677_3.jpg",
#     "anon_2304202677_4.jpg",
#     "anon_2304202677_5.jpg",
#     "anon_2904202666_1.jpg",
#     "anon_2904202666_2.jpg",
#     "anon_2904202666_3.jpg",
#     "anon_2904202666_4.jpg",
#     "anon_2904202666_5.jpg",
#     "anon_21042026555_1.jpg",
#     "anon_21042026555_2.jpg",
#     "anon_21042026555_3.jpg",
#     "anon_21042026555_4.jpg",
#     "anon_21042026555_5.jpg",
#     "anon_28042026877_1.jpg",
#     "anon_28042026877_2.jpg",
#     "anon_28042026877_3.jpg",
#     "anon_28042026877_4.jpg",
#     "anon_28042026877_5.jpg",
#     "anon_Gra_2777_1.jpg",
#     "anon_Gra_2777_2.jpg",
#     "anon_Gra_2777_3.jpg",
#     "anon_Gra_2777_4.jpg",
#     "anon_Gra_2777_1666_1.jpg",
#     "anon_Gra_2777_1666_2.jpg",
#     "anon_Gra_2777_1666_3.jpg",
#     "anon_Gra_2777_1666_4.jpg",
#     "anon_Gra_2777_1666_5.jpg",
#     "anon_210420268765_1.jpg",
#     "anon_210420268765_2.jpg",
#     "anon_210420268765_3.jpg",
#     "anon_210420268765_4.jpg",
#     "anon_210420268765_5.jpg",
#     "anon_290420261234_1.jpg",
#     "anon_290420261234_2.jpg",
#     "anon_290420261234_3.jpg",
#     "anon_290420261234_4.jpg",
#     "anon_290420261234_5.jpg"
]

# Choose data with distance remark in bottom left corner
extra_crop_files_distance = [
    "anon_Pat7_2.jpg",
    "anon_Pat9_2.jpg", 
    "anon_Pat48_2.jpg", 
    "anon_Pat56_2.jpg", 
    "anon_Pat60_2.jpg", 
    "anon_Pat65_1.jpg", 
    "anon_Pat67_2.jpg", 
    "anon_Pat71_2.jpg", 
    "anon_Pat75_2.jpg", 
    "anon_Pat87_2.jpg", 
    "anon_Pat99_2.jpg", 
    "anon_Pat103_2.jpg", 
    "anon_Pat121_2.jpg", 
    "anon_Pat130_2.jpg", 
    "anon_Pat138_2.jpg", 
    "anon_Pat140_2.jpg", 
    "anon_Pat158_2.jpg", 
    "anon_Pat160_2.jpg", 
    "anon_Pat163_2.jpg", 
    "anon_Pat182_2.jpg", 
    "anon_Pat184_2.jpg", 
    "anon_Pat190_2.jpg", 
    "anon_Pat196_5.jpg",
    "anon_Pat204_5.jpg", 
    "anon_Pat211_2.jpg", 
    "anon_Pat216_2.jpg", 
    "anon_Pat217_2.jpg", 
    "anon_Pat261_2.jpg", 
    "anon_Pat268_2.jpg", 
    "anon_Pat269_2.jpg", 
    "anon_Pat276_2.jpg", 
    "anon_Pat285_2.jpg", 
    "anon_Pat295_2.jpg", 
    "anon_Pat311_2.jpg", 
    "anon_Pat321_2.jpg", 
    "anon_Pat356_2.jpg", 
    "anon_Pat358_2.jpg", 
    "anon_Pat379_2.jpg", 
    "anon_Pat418_2.jpg"
]


for current_file in glob.iglob(os.path.join(input_folder, "*.jpg")):
    print(current_file)
    filename = os.path.basename(current_file)
    
    im = Image.open(current_file)
    width, height = im.size
    
    # Cut data with different image section separately
    if filename in extra_crop_files_zoom:
        im1 = im.crop((150,50, 800, 600))
        im1.save(os.path.join(output_folder, filename))
        
    elif filename in extra_crop_files:
        im1 = im.crop((0,100, 920, 900))
        im1.save(os.path.join(output_folder, filename))
        
        
    # Cut data with distance remark in bottom left corner separately
    elif filename in extra_crop_files_distance:
        im2 = im.crop((250,100, 950, 900))
        im2.save(os.path.join(output_folder, filename))
        
    else:
        im3 = im.crop((0,100, 950, 900)) # TÜ: to cut pictogram and labels:(0,100, 950, 900)ULM: to anonymize data: (0, 352, width, 1500) - to cut pictogram and labels: (0, 500, 1200, 1500)
        im3.save(os.path.join(output_folder, filename)) # "anon_Pat" + 
    #break


