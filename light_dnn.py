import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import causal_convolution_layer
class DNNMLPDropout(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropout, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=2)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(128, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))
        z = lay1_out.unsqueeze(1)
        #print(z.shape)
        z_embedding = self.input_embedding(z)
        #print(z_embedding.shape)
        #z_embedding = z_embedding.permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding)
        transformer_embedding_mean = torch.mean(transformer_embedding, 1)
        #print(transformer_embedding_mean.shape)
        emb = self.relu(self.fc2(transformer_embedding_mean))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        # emb = self.dropout(emb)
        # out = self.relu(self.fc5(emb))

        attTmp = self.relu(self.fc_att_1(lay1_out))
        attTmp = self.relu(self.fc_att_2(attTmp))

        att0 = self.sigmoid(self.fc_att_3_label_0(attTmp))
        att1 = self.sigmoid(self.fc_att_3_label_1(attTmp))
        att2 = self.sigmoid(self.fc_att_3_label_2(attTmp))

        fea0 = emb * att0
        fea1 = emb * att1
        fea2 = emb * att2
        fea0emb = self.dropout(fea0)
        fea1emb = self.dropout(fea1)
        fea2emb = self.dropout(fea2)
        out0 = self.relu(self.fc5_0(fea0emb))
        out1 = self.relu(self.fc5_1(fea1emb))
        out2 = self.relu(self.fc5_2(fea2emb))
        out = torch.cat((out0,out1,out2),dim=1)
        #print(out.shape)
        return  out

class DNNMLPDropouthead4(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropouthead4, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(128, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))
        z = lay1_out.unsqueeze(1)
        #print(z.shape)
        z_embedding = self.input_embedding(z)
        #print(z_embedding.shape)
        #z_embedding = z_embedding.permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding)
        transformer_embedding_mean = torch.mean(transformer_embedding, 1)
        #print(transformer_embedding_mean.shape)
        emb = self.relu(self.fc2(transformer_embedding_mean))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        # emb = self.dropout(emb)
        # out = self.relu(self.fc5(emb))

        attTmp = self.relu(self.fc_att_1(lay1_out))
        attTmp = self.relu(self.fc_att_2(attTmp))

        att0 = self.sigmoid(self.fc_att_3_label_0(attTmp))
        att1 = self.sigmoid(self.fc_att_3_label_1(attTmp))
        att2 = self.sigmoid(self.fc_att_3_label_2(attTmp))

        fea0 = emb * att0
        fea1 = emb * att1
        fea2 = emb * att2
        fea0emb = self.dropout(fea0)
        fea1emb = self.dropout(fea1)
        fea2emb = self.dropout(fea2)
        out0 = self.relu(self.fc5_0(fea0emb))
        out1 = self.relu(self.fc5_1(fea1emb))
        out2 = self.relu(self.fc5_2(fea2emb))
        out = torch.cat((out0,out1,out2),dim=1)
        #print(out.shape)
        return  out

class DNNMLPDropouthead6(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropouthead6, self).__init__()
        self.fc1 = torch.nn.Linear(10, 120)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=120, nhead=6)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(120, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(120, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))
        z = lay1_out.unsqueeze(1)
        #print(z.shape)
        z_embedding = self.input_embedding(z)
        #print(z_embedding.shape)
        #z_embedding = z_embedding.permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding)
        transformer_embedding_mean = torch.mean(transformer_embedding, 1)
        #print(transformer_embedding_mean.shape)
        emb = self.relu(self.fc2(transformer_embedding_mean))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        # emb = self.dropout(emb)
        # out = self.relu(self.fc5(emb))

        attTmp = self.relu(self.fc_att_1(lay1_out))
        attTmp = self.relu(self.fc_att_2(attTmp))

        att0 = self.sigmoid(self.fc_att_3_label_0(attTmp))
        att1 = self.sigmoid(self.fc_att_3_label_1(attTmp))
        att2 = self.sigmoid(self.fc_att_3_label_2(attTmp))

        fea0 = emb * att0
        fea1 = emb * att1
        fea2 = emb * att2
        fea0emb = self.dropout(fea0)
        fea1emb = self.dropout(fea1)
        fea2emb = self.dropout(fea2)
        out0 = self.relu(self.fc5_0(fea0emb))
        out1 = self.relu(self.fc5_1(fea1emb))
        out2 = self.relu(self.fc5_2(fea2emb))
        out = torch.cat((out0,out1,out2),dim=1)
        #print(out.shape)
        return  out

class DNNMLPDropouthead8(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropouthead8, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(128, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))
        z = lay1_out.unsqueeze(1)
        #print(z.shape)
        z_embedding = self.input_embedding(z)
        #print(z_embedding.shape)
        #z_embedding = z_embedding.permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding)
        transformer_embedding_mean = torch.mean(transformer_embedding, 1)
        #print(transformer_embedding_mean.shape)
        emb = self.relu(self.fc2(transformer_embedding_mean))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        # emb = self.dropout(emb)
        # out = self.relu(self.fc5(emb))

        attTmp = self.relu(self.fc_att_1(lay1_out))
        attTmp = self.relu(self.fc_att_2(attTmp))

        att0 = self.sigmoid(self.fc_att_3_label_0(attTmp))
        att1 = self.sigmoid(self.fc_att_3_label_1(attTmp))
        att2 = self.sigmoid(self.fc_att_3_label_2(attTmp))

        fea0 = emb * att0
        fea1 = emb * att1
        fea2 = emb * att2
        fea0emb = self.dropout(fea0)
        fea1emb = self.dropout(fea1)
        fea2emb = self.dropout(fea2)
        out0 = self.relu(self.fc5_0(fea0emb))
        out1 = self.relu(self.fc5_1(fea1emb))
        out2 = self.relu(self.fc5_2(fea2emb))
        out = torch.cat((out0,out1,out2),dim=1)
        #print(out.shape)
        return  out

class DNNMLPDropouthead1(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropouthead1, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=1)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(128, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))
        z = lay1_out.unsqueeze(1)
        #print(z.shape)
        z_embedding = self.input_embedding(z)
        #print(z_embedding.shape)
        #z_embedding = z_embedding.permute(2, 0, 1)
        transformer_embedding = self.transformer_decoder(z_embedding)
        transformer_embedding_mean = torch.mean(transformer_embedding, 1)
        #print(transformer_embedding_mean.shape)
        emb = self.relu(self.fc2(transformer_embedding_mean))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        # emb = self.dropout(emb)
        # out = self.relu(self.fc5(emb))

        attTmp = self.relu(self.fc_att_1(lay1_out))
        attTmp = self.relu(self.fc_att_2(attTmp))

        att0 = self.sigmoid(self.fc_att_3_label_0(attTmp))
        att1 = self.sigmoid(self.fc_att_3_label_1(attTmp))
        att2 = self.sigmoid(self.fc_att_3_label_2(attTmp))

        fea0 = emb * att0
        fea1 = emb * att1
        fea2 = emb * att2
        fea0emb = self.dropout(fea0)
        fea1emb = self.dropout(fea1)
        fea2emb = self.dropout(fea2)
        out0 = self.relu(self.fc5_0(fea0emb))
        out1 = self.relu(self.fc5_1(fea1emb))
        out2 = self.relu(self.fc5_2(fea2emb))
        out = torch.cat((out0,out1,out2),dim=1)
        #print(out.shape)
        return  out
class DNNMLPDropoutSRC(torch.nn.Module):
    def __init__(self):
        super(DNNMLPDropoutSRC, self).__init__()
        self.fc1 = torch.nn.Linear(10, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_embedding = causal_convolution_layer.context_embedding(1)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=2)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=2)

        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5 = torch.nn.Linear(128, 3)
        self.fc5_0 = torch.nn.Linear(128, 1)
        self.fc5_1 = torch.nn.Linear(128, 1)
        self.fc5_2= torch.nn.Linear(128, 1)

        self.fc_att_1 = torch.nn.Linear(128, 256)
        self.fc_att_2 = torch.nn.Linear(256, 128)
        self.fc_att_3_label_0 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_1 = torch.nn.Linear(128, 128)
        self.fc_att_3_label_2 = torch.nn.Linear(128, 128)

    def forward(self, x):
        #print(x.shape)
        #x = x.unsqueeze(1)
        lay1_out = self.relu(self.fc1(x))

        emb = self.relu(self.fc2(lay1_out))
        emb = self.relu(self.fc3(emb))
        emb = self.relu(self.fc4(emb))
        emb = self.dropout(emb)
        out = self.relu(self.fc5(emb))
        #print(out.shape)
        return  out


def DNNDROP4(**kwargs):
    model = DNNMLPDropouthead4(**kwargs)

    #model = DNNMLPDropoutSRC(**kwargs)
    return model

def DNNDROP6(**kwargs):
    model = DNNMLPDropouthead6(**kwargs)

    #model = DNNMLPDropoutSRC(**kwargs)
    return model

def DNNDROP8(**kwargs):
    model = DNNMLPDropouthead8(**kwargs)

    #model = DNNMLPDropoutSRC(**kwargs)
    return model

def DNNDROP1(**kwargs):
    model = DNNMLPDropouthead1(**kwargs)

    #model = DNNMLPDropoutSRC(**kwargs)
    return model

def DNNDROP(**kwargs):
    model = DNNMLPDropout(**kwargs)

    #model = DNNMLPDropoutSRC(**kwargs)
    return model


