
#!/usr/bin/env python
import os
import cv2
def save_image(image, path):
    cv2.imwrite(path, image)

class config_loader:
    def __init__(self, configs, namespace):
        self.configs = configs
        self.namespace = namespace
    
    def load_config(self ,key):
        if self.configs[self.namespace].get(key):
            return self.configs[self.namespace][key]
        else:
            print("Key not found in the config file. \n")
            return None