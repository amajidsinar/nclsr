from pathlib import Path
import unicodedata
import string
from collections import namedtuple
import torch


class NameDataset():
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)

    def __init__(self, root):
        self.annotations = []
        Annotation = namedtuple('Annotation', ['language', 'name'])
        for txt_file in Path(root).rglob('*.txt'):
            names = self.readlines(str(txt_file))
            for name in names:
                annotation = Annotation(txt_file.stem, name)
                self.annotations.append(annotation)
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        language, name = self.annotations[index]
        name_tensor = self._encode_name(name)
        name_tensor = torch.nn.functional.one_hot(name_tensor, num_classes=self.n_letters)
        name_tensor = name_tensor.reshape(-1, 1, self.n_letters).to(torch.float32)

        return language, name, name_tensor

    def _encode_name(self, name):
        encoded = []
        for char in name:
            encoded.append(self.all_letters.find(char))
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.Tensor(encoded).reshape(1, -1).to(torch.int64)
        return encoded
        
    @classmethod
    def readlines(cls, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [NameDataset.unicode_to_ascii(line) for line in lines]

    @classmethod
    def unicode_to_ascii(cls, unicode_string):
        return ''.join(
        c for c in unicodedata.normalize('NFD', unicode_string)
        if unicodedata.category(c) != 'Mn'
        and c in cls.all_letters
    )