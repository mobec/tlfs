import keras
from dataset import datasets

dataset = datasets.DataSet()
dataset.load(path="/home/mob/Desktop/Dataset")
dataset.train