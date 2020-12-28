"""
Name: Jens van Holland
Course: Deep Learning
Assigment: 3
Date: 25-11-2020

The layout of the scipt:
1) import libraries
2) Network Class (Net)
3) Experiments


"""
#libraries
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from plotly.offline import iplot,init_notebook_mode
import torch #cpu only version
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#nn extension
class Net(nn.Module):
    def __init__(self, model = "1"): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding= 1)
        self.pool = nn.MaxPool2d(2) # is used twice
        self.conv2 = nn.Conv2d(16, 4, 3, padding= 1)
        self.fc1 = nn.Linear(196, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 196)
        x = self.fc1(x)
        return x
    
net = Net()

#number of parameters
sum([p.numel() for p in net.parameters()])

0.6*5 + 0.31*8.1
"""
First, one has to compute the mean and std for standardisation.
"""
# transform = transforms.Compose(
#     [transforms.ToTensor()])

# #load in full train set
# trainsetfull = torchvision.datasets.MNIST(root='./data/mnist', train=True,
#                                         download=False, transform=transform)
# # data loader for final run 
# trainfullloader = torch.utils.data.DataLoader(trainsetfull, batch_size=60000,
#                                           shuffle=True, num_workers=2)

# torch.mean(trainsetfull)

# trainfullloader.dataset
# mean, std = 0, 0
# for i, data in enumerate(trainfullloader,0):
#     inputs, labels = data
#     mean += inputs.mean()
#     std += inputs.std()
#mean: 0.1307
#std: 0.3081

"""
Load in the dataset and standardize it.
"""
BATCH_SIZE = 2048
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081, ))]) 
#load in full train set
trainsetfull = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                        download=False, transform=transform)
type(trainsetfull)
# data loader for final run 
trainfullloader = torch.utils.data.DataLoader(trainsetfull, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
#split the set 
trainset, valset = torch.utils.data.random_split(trainsetfull, [55000, 5000])
#load in test set
testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, transform=transform,
                                       download=False)
len(testset)
# data loader for training
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
# data loader for validation
valiloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
# data loader for testing
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=2)
#classes
classes = (0,1,2,3,4,5,6,7,8,9)

"""
I've made some funtions for the second assignment, i will use them again here.
- Accuracy 
- Loss calculation
- Training function
"""

def accuracy(net_, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net_(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct/total
    return accuracy

def calculate_loss(net_, dataloader):
    losses = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net_(inputs)
        loss = criterion(outputs, labels)
        losses += loss
    return losses / len(dataloader.dataset)
        
def train(net_, criterion, scheduler, optimizer, trainload,valload = None,epochs = 10  ):
    #lists for results
    losses_train, losses_val = [],[]
    accuracy_train, accuracy_val = [],[]
    running_loss,accuracy_batch = [], []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch: ", epoch+1)

        acc= accuracy(net_, trainload)
        accuracy_train.append(acc)

        if valload is not None: #if not final run
            acc_val = accuracy(net_, valload)
            accuracy_val.append(acc_val)
            
        for i, data in enumerate(trainload, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_(inputs)

            #batch accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy_batch.append(correct/total)

            #loss
            loss = criterion(outputs, labels)
            running_loss.append(loss.detach().numpy())
            loss.backward()
            

            #optimizer
            optimizer.step()
        #scheduler step
        scheduler.step()    
            
        #final run has no validation set, so printing will be different
        if valload is not None:
            print("Accuracy train: {} --- Accuracy validation: {} --- Mean running loss: {}\n ".format(round(acc,2), round(acc_val,2), sum(running_loss)/(epoch*60000)))
        else: print("Accuracy train: {}\n ".format(round(acc,2)))
    
    # evaluate last epoch
    acc= accuracy(net_, trainload)
    accuracy_train.append(acc)

    if valload is not None: #if not final run
        acc_val = accuracy(net_, valload)
        accuracy_val.append(acc_val)

    print('Finished Training')

    if valload is not None:
        return net_, accuracy_train, accuracy_val, running_loss, accuracy_batch

    else: return net_, accuracy_train, running_loss

def save(net_):
    PATH = './mnist_net.pth'
    torch.save(net_.state_dict(), PATH)

#loss function
criterion =   nn.CrossEntropyLoss()

#mod 1
net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=False)
net, acc_train, acc_val, running_loss, accuracy_batch = train(net, criterion, scheduler, optimizer, trainloader, valiloader)

#mod 2
net2 = Net()
optimizer2 = optim.Adam(net2.parameters(), lr = 0.001)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95, last_epoch=-1, verbose=False)
net2, acc_train2, acc_val2, running_loss2, accuracy_batch2 = train(net2, criterion, scheduler2, optimizer2, trainloader, valiloader)

#mod 3
net3 = Net()
optimizer3 = optim.Adam(net3.parameters(), lr = 0.0001)
scheduler3 = optim.lr_scheduler.ExponentialLR(optimizer3, gamma=0.95, last_epoch=-1, verbose=False)
net3, acc_train3, acc_val3, running_loss3, accuracy_batch3 = train(net3, criterion, scheduler3, optimizer3, trainloader, valiloader)

#mod 4
net4 = Net()
optimizer4 = optim.Adam(net4.parameters(), lr = 0.00001)
scheduler4 = optim.lr_scheduler.ExponentialLR(optimizer4, gamma=0.95, last_epoch=-1, verbose=False)
net4, acc_train4, acc_val4, running_loss4, accuracy_batch4 = train(net4, criterion, scheduler4, optimizer4, trainloader, valiloader)

#mod 5
net5 = Net()
optimizer5 = optim.Adam(net5.parameters(), lr = 0.000001)
scheduler5 = optim.lr_scheduler.ExponentialLR(optimizer5, gamma=0.95, last_epoch=-1, verbose=False)
net5, acc_train5, acc_val5, running_loss5, accuracy_batch5 = train(net5, criterion, scheduler5, optimizer5, trainloader, valiloader)


# plotting different learning curves
x = list(np.arange(0,11))
running_loss
type(acc_train)
fig4 = make_subplots(1,2, horizontal_spacing=0.15, subplot_titles= ("Validation", "Training"))
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_val,
                         name = "Learning rate: 0.01",
                        connectgaps = True, 
                        showlegend=False,
                        line_color = 'olive'),
                        row=1, col=1)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_val2,
                         name = "Learning rate: 0.001",
                        connectgaps = True,
                        showlegend=False,
                        line_color = 'red'),
                        row=1, col=1)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_val3,
                         name = "Learning rate: 0.0001",
                        connectgaps = True,
                        showlegend=False,
                        line_color = 'green'),
                        row=1, col=1)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_val4,
                         name = "Learning rate: 0.00001",
                        connectgaps = True,
                        showlegend=False,
                        line_color = 'blue'),
                        row=1, col=1)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_val5,
                        showlegend=False,
                         name = "Learning rate: 0.000001",
                        connectgaps = True,
                        line_color = 'orange'),
                        row=1, col=1)

fig4.add_trace(go.Scatter(x = x ,
                         y = acc_train,
                         name = "Learning rate: 0.01",
                        connectgaps = True,
                        line_color = 'olive'),
                        row=1, col=2)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_train2,
                         name = "Learning rate: 0.001",
                        connectgaps = True,
                        line_color = 'red'),
                        row=1, col=2)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_train3,
                         name = "Learning rate: 0.0001",
                        connectgaps = True,
                        line_color = 'green'),
                        row=1, col=2)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_train4,
                         name = "Learning rate: 0.00001",
                        connectgaps = True,
                        line_color = 'blue'),
                        row=1, col=2)
fig4.add_trace(go.Scatter(x = x ,
                         y = acc_train5,
                         name = "Learning rate: 0.000001",
                        connectgaps = True,
                        line_color = 'orange'),
                        row=1, col=2)
fig4.update_xaxes(title_text = "Epoch", row = 1, col = 1)
fig4.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0], row =1, col =1)
fig4.update_xaxes(title_text = "Epoch", range = [0,10] ,row = 1, col = 2)
fig4.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0],row =1, col =2)

fig4.update_xaxes(nticks = 12)
fig4.update_layout(
    title = "The validation and training accuracy on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.45,
    xanchor="center",
    x=0.5
)
)


# models for mean and stdev.
net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=False)
net, acc_train, acc_val, running_loss, accuracy_batch = train(net, criterion, scheduler, optimizer, trainloader, valiloader)

net_2 = Net()
optimizer_2 = optim.Adam(net_2.parameters(), lr = 0.01)
scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer_2, gamma=0.95, last_epoch=-1, verbose=False)
net_2, acc_train_2, acc_val_2, running_loss_2, accuracy_batch_2 = train(net_2, criterion, scheduler_2, optimizer_2, trainloader, valiloader)

net_3 = Net()
optimizer_3 = optim.Adam(net_3.parameters(), lr = 0.01)
scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer_3, gamma=0.95, last_epoch=-1, verbose=False)
net_3, acc_train_3, acc_val_3, running_loss_3, accuracy_batch_3 = train(net_3, criterion, scheduler_3, optimizer_3, trainloader, valiloader)



#### the same code for calculation the mean and stdev, it is not pretty but does the job. ######
losses_mean_train = []
losses_sd_train = []
max_train_loss, min_train_loss = [], []

accuracy_mean_train, accuracy_mean_val = [], []
accuracy_sd_train, accuracy_sd_val = [], []
max_train_acc, min_train_acc = [],[]
max_val_acc, min_val_acc = [],[]

accuracy_mean_batch, accuracy_sd_batch = [], []
min_batch, max_batch = [],[]


for i in range(len(acc_train)):


    #mean and sd accuracy
    a_train = [acc_train[i], acc_train_2[i], acc_train_3[i]]
    a_val = [acc_val[i], acc_val_2[i], acc_val_3[i]]

    accuracy_mean_train.append(np.mean(a_train))
    accuracy_mean_val.append(np.mean(a_val))

    accuracy_sd_train.append(np.std(a_train))
    accuracy_sd_val.append(np.std(a_val))

    min_train_acc.append(accuracy_mean_train[i] - accuracy_sd_train[i])
    min_val_acc.append(accuracy_mean_val[i] - accuracy_sd_val[i])

    max_train_acc.append(accuracy_mean_train[i] +accuracy_sd_train[i] )
    max_val_acc.append(accuracy_mean_val[i]+ accuracy_sd_val[i])

for j in range(len(accuracy_batch)):
    #loss
    l_train = [running_loss[j],running_loss_2[j],running_loss_3[j]]
    losses_mean_train.append(np.mean(l_train))
    losses_sd_train.append(np.std(l_train))

    min_train_loss.append(losses_mean_train[j] - losses_sd_train[j])
    max_train_loss.append(losses_mean_train[j] + losses_sd_train[j])
    #accuracy
    a_batch = [accuracy_batch[j], accuracy_batch_2[j], accuracy_batch_3[j]]
    accuracy_mean_batch.append(np.mean(a_batch))
    accuracy_sd_batch.append(np.std(a_batch))

    min_batch.append(accuracy_mean_batch[j] - accuracy_sd_batch[j])
    max_batch.append(accuracy_mean_batch[j] + accuracy_sd_batch[j])


# graph for plotting mean and stdev
x = list(range(1,11))
len(min_batch)
fig3 = make_subplots(1,2, horizontal_spacing=0.15)
step = 270
#x_batch = list(np.arange(0,10, 10/step))

fig3.add_trace(go.Scatter(x = list(range(1, 271)) ,
                         y = min_batch,
                        name = "Training data",
                        connectgaps = True,
                        fill = None,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey"
                        ),
                        row=1, col=1, )
fig3.add_trace(go.Scatter(x = list(range(1, 271)), 
                         y = max_batch,
                         name = "Validation data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey",
                        fill="tonexty"),
                        row=1, col=1)
fig3.add_trace(go.Scatter(x = list(range(1, 271)), 
                         y = accuracy_mean_batch,
                         name = "Batch (size =  2048)",
                        connectgaps = True,
                        line_color = 'black',
                        line_width = 1),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = list(range(1, 271)) ,
                         y = min_batch,
                        name = "Training data",
                        connectgaps = True,
                        fill = None,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey"
                        ),
                        row=1, col=1, )
fig3.add_trace(go.Scatter(x = list(range(1, 271)), 
                         y = max_batch,
                         name = "Validation data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey",
                        fill="tonexty"),
                        row=1, col=1)

fig3.add_trace(go.Scatter(x = list(range(1, 271)), 
                         y = accuracy_mean_batch,
                         name = "Batch (size =  2048)",
                        connectgaps = True,                        
                        showlegend=False,
                        line_color = 'black',
                        line_width = 1),
                        row=1, col=1)


# loss
fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))),
                         y = min_train_loss,
                        name = "data",
                        connectgaps = True,
                        fill = None,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey"
                        ),
                        row=1, col=2, )
fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))), 
                         y = max_train_loss,
                         name = "data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey",
                        fill="tonexty"),
                        row=1, col=2)

fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))), 
                         y = losses_mean_train,
                         name = "Batch (size =  2048)",
                        connectgaps = True,                        
                        showlegend=False,
                        line_color = 'black',
                        line_width = 1),
                        row=1, col=2)

fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))),
                         y = min_train_loss,
                        name = "data",
                        connectgaps = True,
                        fill = None,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey"
                        ),
                        row=1, col=2, )
fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))), 
                         y = max_train_loss,
                         name = "data",
                        connectgaps = True,
                        showlegend=False,
                        line_width = 0,
                        mode= "lines",
                        line_color = "grey",
                        fill="tonexty"),
                        row=1, col=2)

fig3.add_trace(go.Scatter(x = list(range(1,len(losses_mean_train))), 
                         y = losses_mean_train,
                         name = "Batch (size =  2048)",
                        connectgaps = True,                        
                        showlegend=False,
                        line_color = 'black',
                        line_width = 1),
                        row=1, col=2)

fig3.update_xaxes(title_text = "Batch",range=[0, 100], row = 1, col = 1)
fig3.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0], row =1, col =1)
fig3.update_xaxes(title_text = "Batch", range = [0, 100] ,row = 1, col = 2)
fig3.update_yaxes(title_text = "Loss", range=[0.0, 2.5],row =1, col =2)
fig3.update_xaxes(nticks = 20, row=1, col = 1)
fig3.update_xaxes(nticks = 20, row=1, col = 2)
fig3.update_layout(
    title = "The batch accuracy and loss on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
)
)
fig3


#### final run ####

net_final = Net()
optimizer_final = optim.Adam(net_final.parameters(), lr = 0.01)
scheduler_final = optim.lr_scheduler.ExponentialLR(optimizer_final, gamma=0.95, last_epoch=-1, verbose=False)
net_final, acc_train_final,  running_loss_final = train(net_final, criterion, scheduler_final, 
                                                        optimizer_final, trainfullloader, epochs=5)

print("The training accuracy: ", accuracy(net_final, trainfullloader))
print("The test accuracy: ", accuracy(net_final, testloader))

#x = list(np.arange(1,109))
x = list(np.arange(0,11))
running_loss
type(acc_train)
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x = x ,
                         y = acc_train_final,
                         name = "Full training set",
                        connectgaps = True, 
                        line_color = 'blue'))

fig6.update_xaxes(title_text = "Epoch", range = [0,5] )
fig6.update_yaxes(title_text = "Accuracy", range=[0.0, 1.0])

fig6.update_xaxes(nticks = 6)
fig6.update_layout(
    title = "The training accuracy of the final run on the MNIST data",
    title_x=0.5,
    height=450, width=800,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.45,
    xanchor="center",
    x=0.5
)
)