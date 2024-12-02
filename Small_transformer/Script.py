import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

def preprocess_text(text):
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\.]", "", text)
    words = text.split()
    return words

class Vocabulary:
    def __init__(self, words):
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, word):
        return self.word2idx[word]
    def decode(self, idx):
        return self.idx2word[idx]
    def __len__(self):
        return len(self.word2idx)

class SentenceCompletionDataset(Dataset):
    def __init__(self, words, context_size):
        self.context_size = context_size
        self.data = []
        for i in range(len(words) - context_size):
            context = words[i:i+context_size]
            target = words[i+context_size]
            self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, target = self.data[index]
        return torch.tensor([vocab.encode(word) for word in context]), vocab.encode(target)

class SentenceCompletionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentenceCompletionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output
    
class TransformerSentenceCompletionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, max_len):
        super(TransformerSentenceCompletionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._generate_positional_encoding(max_len, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads, 
                dim_feedforward=dim_feedforward, 
                activation='relu'
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        # Add positional encoding
        x = self.embedding(x) + self.positional_encoding[:seq_len, :].to(x.device)
        # Pass through transformer encoder
        x = self.transformer(x.permute(1, 0, 2))  # Transformer expects (seq_len, batch_size, embedding_dim)
        x = self.fc(x[-1])  # Take output of the last token
        return x

    @staticmethod
    def _generate_positional_encoding(max_len, embedding_dim):
        """Generates positional encoding matrix."""
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

def train_model(model, data_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for contexts, targets in data_loader:
            contexts, targets = contexts.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(contexts)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

def gen_text(model, context, max_len):
    model.eval()
    with torch.no_grad():
        for _ in range(max_len-len(context)):
            context_tensor = torch.tensor([vocab.encode(word) for word in context]).unsqueeze(0).to(device)
            prediction = model(context_tensor)
            predicted_idx = prediction.argmax(dim=1).item()
            word = vocab.decode(predicted_idx)
            context += [word]
            print(' '.join(context))


if __name__ == "__main__":
    # Simulate loading data from a Wikipedia page
    text = open('./genetics.txt').read()
    words = preprocess_text(text)

    # Prepare vocabulary and dataset
    vocab = Vocabulary(list(set(words)))
    context_size = 128  # Number of previous words to consider
    dataset = SentenceCompletionDataset(words, context_size)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define hyperparameters
    embedding_dim = 512
    hidden_dim = 128
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    embedding_dim = 512
    num_heads = 8
    num_layers = 4
    dim_feedforward = 512
    max_len = 128  # Maximum context size

    # Initialize the Transformer model
    # model = TransformerSentenceCompletionModel(vocab_size=len(vocab),embedding_dim=embedding_dim,num_heads=num_heads,num_layers=num_layers,dim_feedforward=dim_feedforward,max_len=max_len).to(device)

    # Initialize model, loss function, and optimizer
    model = SentenceCompletionModel(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=10e-5)

    # Train the model
    train_model(model, data_loader, criterion, optimizer, epochs=200)

    # Save the model
    torch.save(model.state_dict(), "sentence_completion_model.pth")

    # Test the model with an example
    text = 'genetics is the study of'
    context = [s for s in text.split()]
    print(context)
    gen_text(model, context, max_len)
