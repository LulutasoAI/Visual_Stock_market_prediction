from Generator import Data_Generator
import configparser
from Imagemodifier import Image_modifier
class Data_generator_handler():
    def __init__(self):
        config_ini = configparser.ConfigParser()
        config_ini.read('config.ini', encoding='utf-8')
        self.file = config_ini["Folders"]["Folder"]
        pass 
    def main(self):
        input_OK = False
        while not input_OK:
            try:
                method = int(input("Data retrieval method? 0 for 10 symbols 1 for 501 symbols"))
                input_OK =True
            except:
                method = int(input("input int number please."))
        file_name = self.file
        data_generator= Data_Generator(method)
        data_generator.main()
        
if __name__ == "__main__":
    data_generator_handler = Data_generator_handler()
    data_generator_handler.main()
    image_modifier =  Image_modifier(False)
    image_modifier.main()