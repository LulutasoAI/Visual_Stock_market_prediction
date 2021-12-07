from PIL import Image
from Generator import Data_Generator
import os, glob
import sys 
import configparser

class Image_modifier():
    def __init__(self,Safety=True):
        config_ini = configparser.ConfigParser()
        config_ini.read('config.ini', encoding='utf-8')
        self.file = config_ini["Folders"]["Folder"]
        self.Safety = Safety 

    def main(self):
        if self.Safety:
            sys.exit()
        else:
            pass 
        folder = self.file
        up_and_down_paths = glob.glob(os.path.join(folder,"*"))
        for i,up_or_down in enumerate(up_and_down_paths):
            if i == 0:#up?
                picture_paths = glob.glob(os.path.join(up_or_down,"*.png"))
            else:
                picture_paths.extend(glob.glob(os.path.join(up_or_down,"*png")))

        for i,picture_path in enumerate(picture_paths):
            image = Image.open(picture_path)
            if i == 0:
                if image.size!=(800,575): #If already cropped before exit. #
                    print("going to exit")
                    sys.exit()
                w, h = image.size
                box = (w*0.2,h*0.13,w*0.89,h*0.8)
            image.crop(box).save(picture_path)

            
if __name__ == "__main__":
    IM = Image_modifier(False)
    IM.main()
    