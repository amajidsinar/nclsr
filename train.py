from comet_ml import Experiment
import torch
from data import *
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


experiment = Experiment(api_key="9mPrEpU6XpLG2Pc6MO811ca4e", project_name="rnn-name-classifier")


n_hidden = 128
n_epochs = 100000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

def infer(line_tensor):
    rnn.eval()
    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output = rnn(line_tensor[i])

    return output


rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    rnn.train()
    # hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output = rnn(line_tensor[i])

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

n_confusion = 10000

for epoch in tqdm(range(1, n_epochs + 1)):
    log = {}
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    log['loss']=loss
    
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    

    targets = []
    predictions = []
    val_dataset = NameDataset("data/val")
    for target_language, name, name_tensor in val_dataset:
        output = infer(name_tensor)
        prediction_language, _ = categoryFromOutput(output)
        # print(f'target: {target_language}, prediction: {prediction_language}')
        targets.append(target_language)
        predictions.append(prediction_language)
    prec, rec, fscore, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    
    log['prec'] = prec
    log['rec'] = rec
    log['fscore'] = fscore
    for k, v in log.items():
        print(f' {k}: {v}')
    experiment.log_metrics(log)


torch.save(rnn, 'char-rnn-classification.pt')

