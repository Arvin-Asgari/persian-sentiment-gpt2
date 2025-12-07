# %%


# %%
!pip install torch --upgrade
!pip install transformers --upgrade
!pip install datasets --upgrade
!pip install accelerate --upgrade

# %%
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    return data

# Preprocessing
def preprocess_data(data, tokenizer):
    texts = list(data['text'])
    labels = data['label'].map({'positive': 0, 'neutral': 1, 'negative': 2}).tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings, labels

# Load and preprocess the my dataset
file_path = '/content/data1.txt'
data = load_dataset(file_path)
tokenizer = GPT2Tokenizer.from_pretrained('HooshvareLab/gpt2-fa')

# Set paddings
tokenizer.pad_token = tokenizer.eos_token

encodings, labels = preprocess_data(data, tokenizer)

# Create a dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Spliting the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    encodings['input_ids'], labels, test_size=0.2, random_state=42)

# Rebuild the encodings for train and valid
train_encodings = {
    'input_ids': train_texts,
    'attention_mask': encodings['attention_mask'][:len(train_texts)]
}
val_encodings = {
    'input_ids': val_texts,
    'attention_mask': encodings['attention_mask'][len(train_texts):]
}

# Create the dataset objects
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Load the GPT-2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('HooshvareLab/gpt2-fa', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-sentiment',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjusted batch size to 1
    per_device_eval_batch_size=1,   # Adjusted batch size to 1
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10_000,
    save_total_limit=2,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune 
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./gpt2-sentiment')
tokenizer.save_pretrained('./gpt2-sentiment')


# %%
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Load the fine-tuned model and tokenizer
model_name = '/content/gpt2-sentiment'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name)

# Function to classify sentiment
def classify_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    labels = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return labels[predicted_class]

# Example text
text = "ین کتابخانه خیلی ساکت و آرام است	"

# Classify sentiment
sentiment = classify_sentiment(text, model, tokenizer)
print("Sentiment:", sentiment)



