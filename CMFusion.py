
FOLDER_NAME = '/home/yz1031/HateMM/'

"""Video classification model part

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from tqdm import tqdm
from sklearn.metrics import *


import torch
import pandas as pd
#import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
#import librosa.display
import matplotlib.pyplot as plt
import tarfile
import torch.nn as nn
import torch.nn.functional as F
import random
import math
def fix_the_random(seed_val = 2021):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

fix_the_random(2024)


class Text_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size,fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.network(xb)


class LSTM(nn.Module):
    def __init__(self, input_emb_size = 768, no_of_frames = 100):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_emb_size, 128)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = self.fc(x)
        return x

class MultiHead(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHead, self).__init__()
        self.head_dim = feature_dim // num_heads
        self.heads = nn.ModuleList([
            nn.Linear(feature_dim // num_heads, feature_dim // num_heads) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(feature_dim, feature_dim)  # Merged processing layer

    def forward(self, x):
        original_size = x.size()
        if x.dim() == 3:
            x = x.reshape(-1, original_size[1] * original_size[2])
        # Split input features to different heads
        split_features = torch.split(x, self.head_dim, dim=1)
        head_outputs = [head(features) for head, features in zip(self.heads, split_features)]
        concatenated = torch.cat(head_outputs, dim=1)
        output = self.output_linear(concatenated)
        return output

import math
class Gate_Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim,dropout_rate=0.3):
        super(Gate_Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(feature_dim, attention_dim))
        self.context_vector = nn.Parameter(torch.Tensor(attention_dim, 1))
        self.gate = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_uniform_(self.attention_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.context_vector, a=math.sqrt(5))

    def forward(self, x):
        # Calculating attention scores
        scores = torch.tanh(x @ self.attention_weights)
        attention_scores = scores @ self.context_vector
        attention_scores = F.softmax(attention_scores, dim=1)

        # Application Gating
        gated_scores = torch.sigmoid(self.gate(x))
        # Applying dropout to the gated scores
        gated_scores = self.dropout(gated_scores)

        # Applying Attention and Gating Scores
        attended = attention_scores * gated_scores * x

        return attended

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, output_dim=1, kernel_size=3):
        super(TemporalAttention, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=output_dim, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # x shape: (batch, feature_dim, seq_len)
        weights = self.conv1d(x)
        weights = F.softmax(weights, dim=2)  # Apply softmax over the temporal dimension
        return weights





class Combined_model(nn.Module):
    def __init__(self, text_model, video_model, audio_model, TemporalAttention,MultiHead,Gate_Attention,num_classes):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.video_model = video_model
        self.text_head = MultiHead(64, 8)
        self.video_head = MultiHead(64, 8)
        self.audio_head = MultiHead(64, 8)
        self.video_attention = TemporalAttention(64)
        self.audio_attention = TemporalAttention(64)
        self.fusion = Gate_Attention(64,8)
        self.fc_output   = nn.Linear(64, num_classes)
    def forward(self, x_text, x_vid, x_audio):
        tex_out = self.text_model(x_text)
        vid_out = self.video_model(x_vid)
        aud_out = self.audio_model(x_audio)
        video_features = vid_out.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        audio_features = aud_out.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        vid_weights = self.video_attention(video_features)
        aud_weights = self.audio_attention(audio_features)
        # 加权求和
        video_features = aud_weights * video_features + video_features
        audio_features = vid_weights * audio_features + audio_features
        video_features = video_features.permute(0, 2, 1)
        audio_features = audio_features.permute(0, 2, 1)
        tex_out = self.text_head(tex_out)
        video_features = self.video_head(video_features)
        audio_features = self.audio_head(audio_features)
        tex_out_weights = self.fusion(tex_out)
        vid_out_weights = self.fusion(video_features)
        aud_out_weights = self.fusion(audio_features)
        combined_features =  tex_out_weights + vid_out_weights + aud_out_weights
        out = self.fc_output(combined_features)
        return out



## ---------------------- Dataloader ---------------------- ##
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels):
        "Initialization"
        self.labels = labels
        self.folders = folders

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_text(self,selected_folder):
        try:
            return torch.tensor(textData[selected_folder]), torch.tensor(vidData[selected_folder]), torch.tensor(audData[selected_folder])
        except KeyError as e:
            return torch.zeros(768), torch.zeros(100,768), torch.zeros(100,40)  # Return None for all if any is missing

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]
        try:
            # Load data
            X_text, X_vid, X_audio = self.read_text(folder)
            y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor
        except:
            with open("Exceptions.txt","a") as f:
                f.write("{}\n".format(folder))
            return None
        return X_text, X_vid, X_audio, y

def evalMetric(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
        recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
        precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
    except:
        return dict({"accuracy": 0, 'mF1Score': 0, 'f1Score': 0, 'auc': 0,'precision': 0, 'recall': 0})
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})



#loading Audio features
import pickle

#with open(FOLDER_NAME+'all_HateXPlainembedding.p','rb') as fp:
with open(FOLDER_NAME+'all_rawBERTembedding.p','rb') as fp:
    textData = pickle.load(fp)

#with open(FOLDER_NAME+'vgg19_audFeatureMap.p','rb') as fp:
#with open(FOLDER_NAME+'MFCCFeatures.p','rb') as fp:
    #audData = pickle.load(fp)


with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

# train, test split
train_list, train_label= allDataAnnotation['train']
val_list, val_label  =  allDataAnnotation['val']
test_list, test_label  =  allDataAnnotation['test']

allVidList = []
allVidLab = []

allVidList.extend(train_list)
allVidList.extend(val_list)
allVidList.extend(test_list)

allVidLab.extend(train_label)
allVidLab.extend(val_label)
allVidLab.extend(test_label)


vidData ={}
missing_files = []
for i in allVidList:
    file_path = os.path.join(FOLDER_NAME,"video_vectors_vit/", f"{i}_vit.p")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
          vidData[i] = np.array(pickle.load(fp))
    else:
        missing_files.append(i)

audData = {}
missing_files_audio = []
for i in allVidList:
    file_path = os.path.join(FOLDER_NAME,"audio_mfcc/", f"{i}.npy")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fp:
            audData[i] = np.load(fp)
            #print(audData[i].shape)
    else:
        missing_files_audio.append(i)


# Audio parameters
input_size_text = 768 #40 #76800 #

input_size_audio = 40 #76800 #


fc1_hidden_audio, fc2_hidden_audio = 128, 128

# training parameters
k = 2            # number of target category
epochs = 40
batch_size = 64
learning_rate = 1e-4
log_interval = 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    all_y_pred = []
    all_y_true = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, (X_text, X_vid, X_aud, y) in enumerate(train_loader):
        # distribute data to device
        X_text, X_vid, X_aud, y = (X_text.float()).to(device), (X_vid.float()).to(device), (X_aud.float()).to(device), y.to(device).view(-1, )

        N_count += X_text.size(0)

        optimizer.zero_grad()
        output = model(X_text, X_vid, X_aud)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device))
        #loss = F.cross_entropy(output, y)
        losses.append(loss.item())

            # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_true.extend(y.cpu().numpy())
        metrics = evalMetric(np.array(all_y_true), np.array(all_y_pred))

        loss.backward()
        optimizer.step()

        # show information
    if (batch_idx + 1) % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))


    return losses, metrics

def validation(model, device, optimizer, test_loader, testingType = "Test"):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X_text, X_vid, X_aud, y in test_loader:
            # distribute data to device
            X_text, X_vid, X_aud, y = (X_text.float()).to(device), (X_vid.float()).to(device), (X_aud.float()).to(device), y.to(device).view(-1, )

            output = model(X_text, X_vid, X_aud)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)

    print("---------------------")
    # try:
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    # except:
    #   metrics = None

    # show information
    print('\nTest set: ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
                len(all_y), test_loss, 100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))

    # # save Pytorch models of best record
    # torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pt'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pt'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))
    #print(testingType + " Len:", len(list(all_y_pred.cpu().data.squeeze().numpy())))
    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy())




# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}
valParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}




with open(FOLDER_NAME+'allFoldDetails.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

allF = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5']
#allF = ['Fold_4']


finalOutputAccrossFold = {}

for fold in allF:
    # train, test split
    train_list, train_label = allDataAnnotation[fold]['train']
    val_list, val_label = allDataAnnotation[fold]['val']
    test_list, test_label = allDataAnnotation[fold]['test']

    combined_test_list = val_list + test_list
    combined_test_label = val_label + test_label

    train_set = Dataset_3DCNN(train_list, train_label)
    test_set = Dataset_3DCNN(combined_test_list, combined_test_label)

    train_loader = data.DataLoader(train_set, **params)
    test_loader = data.DataLoader(test_set, **valParams)

    tex = Text_Model(input_size_text, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
    vid = LSTM().to(device)
    aud = LSTM(40,100).to(device)
    comb = Combined_model(tex, vid, aud,TemporalAttention,MultiHead,Gate_Attention, k).to(device)

    optimizer = torch.optim.Adam(comb.parameters(), lr=learning_rate)   # optimize all cnn parameters


    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    testFinalValue = None
    finalScoreAcc = 0
    prediction = None

    # start training
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, comb, device, train_loader, optimizer, epoch)
        test_loss, test_scores, veTest_pred = validation(comb, device, optimizer, test_loader, 'Test')

        if test_scores['accuracy'] > finalScoreAcc:
            finalScoreAcc = test_scores['accuracy']
            testFinalValue = test_scores
            print("veTest_pred", len(veTest_pred))
            prediction = {'test_list': combined_test_list, 'test_label': combined_test_label, 'test_pred': veTest_pred}
        
        print("====================")
        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores['accuracy'])
        epoch_test_losses.append(test_loss)
        epoch_test_scores.append(test_scores['accuracy'])

        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)

    finalOutputAccrossFold[fold] = {'test': testFinalValue, 'test_prediction': prediction}

with open('foldWiseRes_vit_hateX_audioVGG19_lstm.p', 'wb') as fp:
    pickle.dump(finalOutputAccrossFold, fp)

allValueDict = {}
for fold in allF:
    for val in finalOutputAccrossFold[fold]['test']:
        try:
            allValueDict[val].append(finalOutputAccrossFold[fold]['test'][val])
        except KeyError:
            allValueDict[val] = [finalOutputAccrossFold[fold]['test'][val]]


import numpy as np
for i in allValueDict:
    print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

