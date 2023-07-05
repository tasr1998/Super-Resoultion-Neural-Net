
from model import SRCNN
from data import SRDataset
import torch
import matplotlib.pyplot as plt
import time
import os
import argparse
from torch import nn
from torchvision.models.vgg import vgg16
from VGGLoss import VGGLoss

#https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


parser = argparse.ArgumentParser(description="Create Image patches from folder and save as npy")
parser.add_argument('--model-save-path', '-s', required=True, help="Folder to save models" )
parser.add_argument('--input-npy', '-o', required=True, help="path of training input npy")
parser.add_argument('--label-npy', '-O', required=True, help="path of training label npy")
parser.add_argument('--epochs', '-n', type=int, default=2000, help="epoch counts")
parser.add_argument('--every', '-N', type=int, default=50,  help="Save model every", )
parser.add_argument('--vgg-loss', '-v',  help="use vgg loss", action="store_true")

args = parser.parse_args()

print(args)

torch.manual_seed(420)

datas = SRDataset(args.input_npy, args.label_npy)
train_len = int(len(datas) * 0.9)
training_set, validation_set = torch.utils.data.random_split(dataset=datas, lengths=[train_len, len(datas) - train_len ])

train_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=True)

num_epochs = args.epochs
save_every = args.every
save_path = args.model_save_path

srcnn = SRCNN().double()
srcnn = srcnn.to("cuda")

criterion = torch.nn.MSELoss()
if(args.vgg_loss):
    criterion = VGGLoss().double()
    criterion.to("cuda")
optim = torch.optim.Adam([
    {"params": srcnn.conv1.parameters()},
    {"params": srcnn.conv2.parameters()},
    {"params": srcnn.conv3.parameters(), "lr":1e-5}
    ],
    lr=1e-4) 



def make_train_step(model, loss_fn, optimizer):
    def train_step(inputs, labels):
        model.train()
        pred = model(inputs)
        loss = loss_fn(labels, pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

train_step = make_train_step(srcnn, criterion, optim)

training_losses = []
validation_losses = []

start = time.time()
cur_epoch = 0
print("Starting epochs")
for epoch in range(0, num_epochs):

    total_training_loss = 0
    batch_count = 0
    for inputs, labels in train_loader:
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        
        total_training_loss += train_step(inputs, labels)
        batch_count += 1

    if(batch_count > 0):
        training_losses.append(total_training_loss/batch_count)
    
    with torch.no_grad():
        val_batch_count = 0

        total_validation_loss = 0 
        for inputs, labels in validation_loader:
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")   

            srcnn.eval()

            pred = srcnn(inputs)
            total_validation_loss += criterion(labels, pred)
            val_batch_count += 1

        if(val_batch_count > 0):    
            validation_losses.append(total_validation_loss/val_batch_count)
    
    if(cur_epoch % save_every == 0):
        print("On epoch {}, saving model".format(cur_epoch))
        torch.save(srcnn.state_dict(), os.path.join(save_path,"model{}.pth".format(cur_epoch)))

    cur_epoch += 1

finish = time.time()
print("Took {} epochs {} seconds, now saving".format(num_epochs, finish-start))
torch.save(srcnn.state_dict(), os.path.join(save_path,"model{}.pth".format(cur_epoch)))

epochs = range(1,num_epochs+1)
plt.plot(epochs, training_losses, 'g', label='Training loss')
plt.plot(epochs, validation_losses, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()