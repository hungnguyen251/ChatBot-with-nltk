# ChatterBot
ChatterBot is a Python library that makes it easy to generate automated responses to a user’s input. ChatterBot uses a selection of machine learning algorithms to produce different types of responses. This makes it easy for developers to create chat bots and automate conversations with users. 

# Installation
Python 3.6+ required

python -m venv .venv
source .venv/bin/activate

pip3 install nltk
pip3 install tensorflow
pip3 install numpy

# Basic usage

import nltk
import warnings
 
warnings.filterwarnings("ignore")
#nltk.download() # for downloading packages
import tensorflow as tf
import numpy as np
import random
import string 
