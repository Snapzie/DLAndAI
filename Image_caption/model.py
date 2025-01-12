import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class CNNModel(nn.Module):
    def __init__(self,embedding_size):
        super(CNNModel,self).__init__()
        resnet = models.resnet152(weights=True)
        module_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features,embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size,momentum=0.01)

    def forward(self,input_images):
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0),-1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features

class LSTModel(nn.Module):
    def __init__(self,embedding_size,hidden_layer_size,vocabulary_size,num_layers,max_seq_len=20):
        super(LSTModel,self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size,embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size,hidden_layer_size,num_layers,batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size,vocabulary_size)
        self.max_seq_len = max_seq_len
    
    def forward(self,input_features,caps,lens):
        embeddings = self.embedding_layer(caps)
        embeddings = torch.cat((input_features.unsqueeze(1),embeddings),1)
        lstm_input = pack_padded_sequence(embeddings,lens,batch_first=True)
        hidden_variables,_ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs
    
    def sample(self,input_features,lstm_states=None):
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs,lstm_states)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))
            _, predicted_outputs = model_outputs.max(1)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)
            lstm_inputs = lstm_inputs.unsqueeze(1)
        sampled_indices = torch.stack(sampled_indices,1)
        return sampled_indices