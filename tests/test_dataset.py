from dataset import NameDataset
from pathlib import Path
from torch.utils.data import DataLoader


def test_instance():
    dataset = NameDataset("data/names")
    for idx, (language, _,__, ___) in enumerate(dataset):
        pass
    assert idx == len(dataset) - 1
    assert idx == len(dataset)

def test_language():
    languages = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
    dataset = NameDataset("data/names")
    for language, _, __, ___ in dataset:
        assert (language in languages) == True

# def test_name():
#     root = "data/names"
#     dataset = NameDataset(root)
#     for language, _, name, __ in dataset:
#         dataset_txt = f'{Path(root)}/{language}.txt'
#         names = dataset.readlines(dataset_txt)
#         assert (name in names) == True

def test_loader():
    dataset = NameDataset("data/names")
    loader = DataLoader(dataset, batch_size=4, pin_memory=True, num_workers=4, collate_fn=dataset.collate_fn)
    for idx, (_, __, ___, ____ )in enumerate(loader):
        pass
    import pdb; pdb.set_trace()





        

    


