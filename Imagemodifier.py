from PIL import Image
from Generator import Data_Generator
import os, glob
import sys 


class Image_modifier():
    def __init__(self,Safety=True):
        self.Safety = Safety 

    def main(self):
        if self.Safety:
            sys.exit()
        else:
            pass 
        generator = Data_Generator() #Just to get the filepath not to make the entire data again by executing the main function.
        folder = generator.file
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
    