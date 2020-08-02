from pathlib import Path
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description="Data splitter for name classifier")
parser.add_argument('-i', '--input', type=str, help='path to input', required=True)
parser.add_argument('-o', '--output', type=str, help='path to output', default='dumping')
parser.add_argument('-s', '--training_size', type=float, default=0.8)
parser.add_argument('-r', '--random_seed', type=int, default=69)
args = parser.parse_args()

def write_to_txt(out_path, samples):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as writer:
        for sample in samples:
            writer.write(f'{sample} \n')


for language_txt in Path(args.input).rglob('*.txt'):
    np.random.seed(args.random_seed)
    with open(language_txt, 'r') as reader:
        names = [name.rstrip() for name in reader.readlines()]
        n_name = len(names)
        random_sequence_idx = np.random.permutation(n_name)
        random_sequence_names = np.array(names)[random_sequence_idx]
        n_training_samples = int(n_name * args.training_size)
        random_training_names = random_sequence_names[:n_training_samples]
        random_validation_names = random_sequence_names[n_training_samples:]

        assert random_training_names[-1] != random_validation_names[0]

        train_dest = f'{Path(args.output)}/train/{Path(language_txt).name}'
        write_to_txt(train_dest, random_training_names)

        val_dest = f'{Path(args.output)}/val/{Path(language_txt).name}'
        write_to_txt(val_dest, random_validation_names)
        