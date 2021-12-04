import os
import configparser
import glob
import json
import sys 


class Labeller():

    def __init__(self):
        config_ini = configparser.ConfigParser()
        config_ini.read('config.ini', encoding='utf-8')
        self.datafolderpath = config_ini["Folders"]["folder"]
        self.labelled_data_paths = glob.glob(os.path.join(self.datafolderpath,"*"))
        print(self.labelled_data_paths)
        self.file_type = json.loads(config_ini["File_config"]["file_type"])
        self.isformatted = config_ini["File_config"]["isformatted"]
        if "T" in self.isformatted:
            self.isformatted = True
        else:
            self.isformatted = False
        print(self.file_type,"config file_type {}".format(type(self.file_type)))
        print(self.datafolderpath,"datafolderpath")
        print(self.labelled_data_paths,"labelled_data_paths")

    def label_manifest(self,proper_format=False):
        if proper_format:
            X,Y = self.formatted_labelling_process()
        else:
            X,Y = self.non_formatted_labelling_process()
        return X,Y

    def labelled_data_path_to_individual_paths(self,labelled_data_path,file_extention=".jpg"):
        print(type(file_extention))
        if type(file_extention) == str:
            print("single file_extention detected")
            individual_paths = glob.glob(os.path.join(labelled_data_path, "*", file_extention))
        elif type(file_extention) == list:
            print("multiple file_extentions detected")
            for i,file_ext in enumerate(file_extention):
                if i == 0:
                    print(os.path.join(labelled_data_path, "*", file_ext),"glob to get, when i== 0")
                    individual_paths = glob.glob(os.path.join(labelled_data_path, "*"+file_ext))
                else:
                    individual_paths.extend(glob.glob(os.path.join(labelled_data_path, "*"+file_ext)))
        else:
            print("error")
        return individual_paths

    def non_formatted_labelling_process(self):
        X, Y = [], []
        for i, a_label in enumerate(self.labelled_data_paths):
            individual_paths = self.labelled_data_path_to_individual_paths(a_label,self.file_type)
            for individual_path in individual_paths:
                X.append(individual_path)
                Y.append(i)
        return X,Y

    def formatted_labelling_process(self):
        X, Y = [], []
        print(self.labelled_data_paths,"1, paths from the formatted_labelling_process")
        for i, a_label in enumerate(self.labelled_data_paths):
            individual_paths = self.labelled_data_path_to_individual_paths(a_label,self.file_type)
            print(individual_paths,"individual_paths")
            for individual_path in individual_paths:
                X.append(individual_path)
                Y.append(os.path.basename(a_label))
        return X,Y
    
    def main(self):
        X,Y = self.label_manifest(self.isformatted)
        if len(X) == len(Y):
            return X,Y
        else:
            sys.exit()

if __name__ == "__main__":
    lb = Labeller()
    X,Y = lb.label_manifest(lb.isformatted)
    for i, a in enumerate(X):
        print(a,Y[i])
