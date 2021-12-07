import os 
class Utils():
    def create_folder_if_None_exists(self,name):
            if not os.path.exists(name):
                os.makedirs(name)
                print("The folder named '{}' had not existed so I created it.".format(name))
            else:
                print("The folder named '{}' already exists so nothing was executed.".format(name))
