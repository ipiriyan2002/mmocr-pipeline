from dataset_lib.ing_dataset import IngDataset
from dataset_lib.cord_dataset import CordDataset
from dataset_lib.meme_dataset import MemeDataset
from dataset_lib.glosat_dataset import GlosatDataset


#Ids to Class Names for datasets
ids = dict(
    ing=IngDataset,
    cordv2=CordDataset,
    meme=MemeDataset,
    glosat=GlosatDataset
)

