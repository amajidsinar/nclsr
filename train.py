from comet_ml import Experiment
import torch
# from data import *
from model import *
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from dataset import NameDataset
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tqdm import tqdm
from torch.utils.data import DataLoader

experiment = Experiment(api_key="9mPrEpU6XpLG2Pc6MO811ca4e", project_name="rnn-name-classifier", disabled=True)

batch_size = 2
n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


from dataset import LANG_TO_IDX

IDX_TO_LANG = {}
for k,v in LANG_TO_IDX.items():
    IDX_TO_LANG[v]=k

def infer(line_tensor):
    # import pdb; pdb.set_trace()
    rnn.eval()
    with torch.no_grad():
        pred = rnn(line_tensor)
        pred = torch.nn.functional.softmax(pred)
        pred_conf, pred_class = torch.max(pred, dim=1)
        pred_class = IDX_TO_LANG[pred_class.item()]
    return pred_class, pred_conf 


rnn = RNN(58, n_hidden, 18)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train(category_tensors, line_tensors):
    rnn.train()
    optimizer.zero_grad()
    output = rnn(line_tensors)
    # import pdb; pdb.set_trace()

    loss = criterion(output, category_tensors)
    loss.backward()

    optimizer.step()

    return output, loss.item()


# def train_batch(category_tensor, line_tensor)

# Keep track of losses for plotting

all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

n_confusion = 10000

train_dataset = NameDataset("data/train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, pin_memory=True, num_workers=4)
val_dataset = NameDataset("data/val")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn, pin_memory=True, num_workers=4)


for epoch in tqdm(range(1, n_epochs + 1)):
    current_loss = 0
    # import pdb; pdb.set_trace()
    print(f'epoch: {epoch}')

    log = {}
    for languages, language_tensors, name, name_tensors in tqdm(train_loader):
        # import pdb; pdb.set_trace()
        output, loss = train(language_tensors, name_tensors)
    # category, line, category_tensor, line_tensor = randomTrainingPair()
    # output, loss = train(category_tensor, line_tensor)
        current_loss += loss

    log['loss']=current_loss

    
    # Print epoch number, loss, name and guess
    # if epoch % print_every == 0:
    #     guess, guess_i = categoryFromOutput(output)
    #     correct = '✓' if guess == category else '✗ (%s)' % category
    #     print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    

    targets = []
    predictions = []
    
    for language, language_tensor, name, name_tensor in val_loader:
        pred_class, _ = infer(name_tensor)
        targets.append(language)
        predictions.append(pred_class)
    prec, rec, fscore, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    
    log['prec'] = prec
    log['rec'] = rec
    log['fscore'] = fscore
    for k, v in log.items():
        print(f' {k}: {v}')
    experiment.log_metrics(log)


torch.save(rnn, 'char-rnn-classification.pt')

