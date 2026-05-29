__author__ = 'tylin'
from importlib.resources import path
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import networkx as nx
import sys
from collections import defaultdict
from .tokenizer.ptbtokenizer import PTBTokenizer

class COCOEvalCap(object):
    def __init__(self, val_instr_data):
        self.evalImgs = []
        self.eval = {}
        self.gts = defaultdict(list)
        for item in val_instr_data:
            self.gts[str(item['path_id'])].append(item['instruction'])


    def evaluate(self, path2inst):
        for k,v in path2inst.items():
            path2inst[k] = [v]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.gts)
        res = tokenizer.tokenize(path2inst)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
            else:
                self.setEval(score, method)

    def setEval(self, score, method):
        self.eval[method] = score

