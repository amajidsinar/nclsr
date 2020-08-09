from pathlib import Path
import unicodedata
import string
from collections import namedtuple
import torch


LANG_TO_IDX = {
    "French": 0,
    "Czech": 1,
    "Italian": 2,
    "German": 3,
    "Scottish": 4,
    "Dutch": 5,
    "Greek": 6,
    "Arabic": 7,
    "Spanish": 8,
    "Vietnamese": 9,
    "Irish": 10,
    "Polish": 11,
    "Portuguese": 12,
    "Russian": 13,
    "Japanese": 14,
    "English": 15,
    "Chinese": 16,
    "Korean": 17,
}


class NameDataset:
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)

    def __init__(self, root, mapping=LANG_TO_IDX, max_len=20):

        self.annotations = []
        Annotation = namedtuple("Annotation", ["language", "name"])
        for txt_file in Path(root).rglob("*.txt"):
            names = self.readlines(str(txt_file))
            for name in names:
                annotation = Annotation(txt_file.stem, name)
                self.annotations.append(annotation)

        self.mapping = mapping

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        language, name = self.annotations[index]
        name_tensor = self._encode_name(name)
        language_tensor = torch.Tensor([self.mapping[language]]).to(torch.long)
        # import pdb; pdb.set_trace()

        return language, language_tensor, name, name_tensor

    def collate_fn(self, batch):
        
        languages = []
        language_tensors = []
        names = []
        name_tensors = []
        import pdb; pdb.set_trace()
        for language, language_tensor, name, name_tensor in batch:
            languages.append(language)
            language_tensors.append(language_tensor)
            names.append(name)
            name_tensors.append(name_tensor)

        language_tensors = torch.cat(language_tensors)

        return languages, language_tensors, names, name_tensors

    def _encode_name(self, name):
        encoded = []
        for char in name:
            encoded.append(self.all_letters.find(char))
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.Tensor(encoded).to(torch.int64)
        return encoded

    @classmethod
    def readlines(cls, filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [NameDataset.unicode_to_ascii(line) for line in lines]

    @classmethod
    def unicode_to_ascii(cls, unicode_string):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", unicode_string)
            if unicodedata.category(c) != "Mn" and c in cls.all_letters
        )
