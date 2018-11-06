import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pickle
import torch.functional as F
import numpy as np


class BaseModel(object):

    def __init__(self, params):
        self.model = None
        self.cuda = params['cuda']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.momentum = params['momentum']

        self.data = params['dataset']

    def train(self, params):
        if self.model is None:
            raise("ERROR: no model has been specified")

        # Data loading is manual for now
        dataset = params['dataset']
        with open(dataset, 'rb') as file:
            dataset = pickle.load(file)

        np.random.seed(params['seed'])
        train_set = dataset['train']
        train_labels = dataset['train_labels']
        # shuffling  training data
        idx = np.random.permutation(range(train_set.shape[0]))
        train_set = train_set[idx, :, :, :]
        train_labels = train_labels[idx, :]

        train_set = np.moveaxis(train_set, 3, 1)

        # shuffling validation data
        val_set = dataset['validation']
        val_labels = dataset['validation_labels']

        idx = np.random.permutation(range(val_set.shape[0]))
        val_set = val_set[idx, :, : , :]
        val_labels = val_labels[idx, :]

        val_set = np.moveaxis(val_set, 3, 1)

        # Start training loop
        if self.cuda:
            self.model = self.model.cuda()

        loss_fn = nn.CrossEntropyLoss()
        if self.cuda:
            loss_fn = loss_fn.cuda()

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # remember to set volatile=True for validation input and label
        for epoch in range(self.epochs):
            train_running_loss = 0.0
            validation_running_loss = 0.0
            for i in range(train_set.shape[0]):
                start = i*params['batch_size']
                end = max((i+1)*params['batch_size'], train_set.shape[0])
                inputs, labels = torch.from_numpy(train_set[start:end, :, :, :]), torch.from_numpy(train_labels[start:end, :])

                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = self.model(inputs.float())

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.cpu().item()

                self.model.eval()
                eval_end = False
                for j in range(val_set.shape[0]):
                    if val_set.shape[0] <= params['batch_size']:
                        start = 0
                        end = params['batch_size']
                        eval_end = True
                    else:
                        start = j*params['batch_size']
                        end = (j+1)*params['batch_size']

                    val_inputs, val_labels = torch.from_numpy(val_set[start:end, :, : ,:]), torch.from_numpy(val_labels[start:end, :])
                    if self.cuda:
                        val_set, val_labels = val_set.cuda(), val_labels.cuda()

                    val_inputs, val_labels = Variable(val_inputs, volatile=True), Variable(val_labels, volatile=True)
                    val_outputs = self.model.forward(val_inputs.float())
                    val_loss = loss_fn(val_outputs, val_labels)
                    validation_running_loss += val_loss.cpu().item()
                    if eval_end:
                        break
                self.model.train()

                if i % 100 == 99:
                    print('[%d, %5d] train loss: %.3f' %
                          (epoch + 1, i + 1, train_running_loss / 100))
                    train_running_loss = 0.0

                    print('[%d, %5d] val loss: %.3f' %
                          (epoch + 1, i + 1, validation_running_loss / 100))
                    validation_running_loss = 0.0

                # ADD CHECKPOINTING HERE

        print("Finished training!")
        print("Saving model to %s" %(params['saved_models']+'model_weights.pt'))
        torch.save(self.model.state_dict(), params['saved_models']+'model_weights.pt')

    def predict(self, params):
        self.model.load_state_dict(torch.load(params['saved_models']+'model_weights.pt'))
        self.model.eval()
        raise("Not implemented yet")