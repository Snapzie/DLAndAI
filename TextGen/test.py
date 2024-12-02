import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Initialize tokenizer and model configuration
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config()
model = GPT2LMHeadModel(config)
model.to(device)  # Move model to the appropriate device
model.train()  # Set model to training mode

# Example data - For actual training, replace with your dataset
data = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do..."
input_ids = tokenizer.encode(data, add_special_tokens=True, return_tensors="pt")
input_ids = input_ids.to(device)

# Prepare inputs and labels for language modeling
labels = input_ids[:, 1:].contiguous()
inputs = input_ids[:, :-1].contiguous()

# Training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 1000
batch_size = 2

# Training loop
for epoch in range(epochs):
    total_loss = 0
    num_batches = 0
    for i in range(0, inputs.size(1) - batch_size + 1, batch_size):
        batch_inputs = inputs[:, i:i+batch_size]
        batch_labels = labels[:, i:i+batch_size]

        # Forward pass
        outputs = model(batch_inputs)
        logits = outputs.logits

        # Reshape logits and labels to calculate loss
        loss_fct = torch.nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.size(-1))
        batch_labels = batch_labels.view(-1)

        # Compute loss
        loss = loss_fct(logits, batch_labels)

        # Backpropagate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    
    average_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")

# Save the model
model_dir = 'my_trained_model_directory'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Load the trained model for evaluation
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.to(device)  # Ensure the model is on the GPU for evaluation
model.eval()

# Test the model
test_prompt = "Alice was beginning"
encoded_input = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
output_sequences = model.generate(
    input_ids=encoded_input,
    max_length=100,
    num_return_sequences=1,
    temperature=1.0,
    top_p=0.92,
    top_k=50
)
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
