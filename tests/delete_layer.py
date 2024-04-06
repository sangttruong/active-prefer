from transformers import AutoModelForSequenceClassification
import torch 

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# check if classifier is attrite of model 
if hasattr(model, 'classifier'):
    model.classifier = torch.nn.Identity()

# Print the names of all layers in the model
for name, param in model.named_parameters():
    print(name)

