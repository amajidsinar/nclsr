from dataset import NameDataset
from pathlib import Path

def test_instance():
    dataset = NameDataset("data/names")
    for idx, (language, name, target) in enumerate(dataset):
        pass
    assert idx == len(dataset) - 1

def test_language():
    languages = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
    dataset = NameDataset("data/names")
    for language, _, __ in dataset:
        assert (language in languages) == True

def test_name():
    root = "data/names"
    dataset = NameDataset(root)
    for language, name, _ in dataset:
        dataset_txt = f'{Path(root)}/{language}.txt'
        names = dataset.readlines(dataset_txt)
        # import pdb; pdb.set_trace()
        assert (name in names) == True


        

    


