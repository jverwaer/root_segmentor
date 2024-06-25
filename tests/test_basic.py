import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rhsegmentor import dataloader
from rhsegmentor import utils
#dataloader.foo()

r = utils.get_save_fname(fname = "c:\\test\\hallo\\foo.txt",
                     save_dir = "C:\\result",
                     suffix = "TEST.TXT")
print("1)", r)
r = utils.get_save_fname(fname = "c:\\test\\hallo\\foo.txt",
                     save_dir = ".\\my_folder",
                     suffix = "TEST.TXT")
print("2)",r)

r = utils.get_save_fname(fname = "c:\\test\\hallo\\foo.txt",
                     save_dir = None,
                     suffix = "TEST.TXT")
print("3)",r)