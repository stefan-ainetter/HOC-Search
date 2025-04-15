import json
import os


from MonteScene.Proposal import Proposal
from MonteScene.constants import NodesTypes

class CategoryProposal(Proposal):
    def __init__(self, id, category_label):

        super().__init__(id, NodesTypes.OTHERNODE)

        self.category_label = category_label


