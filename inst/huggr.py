import torch
import os
import numpy as np
from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer, BertForSequenceClassification, XLMRobertaForMaskedLM, XLMRobertaTokenizer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with_gpu = 'cuda' if torch.cuda.is_available() else None
print(with_gpu)

def bert_download(model_name, path):
  model = BertModel.from_pretrained(model_name)
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)
  
def roberta_download(model_name, path):
  model = XLMRobertaForMaskedLM.from_pretrained(model_name)
  tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)
  
def bert_download(model_name, path):
  model = BertModel.from_pretrained(model_name)
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)

def bert_class_download(model_name, path):
  model = BertForSequenceClassification.from_pretrained(model_name)
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)

def t5_download(model_name, path):
  model = T5ForConditionalGeneration.from_pretrained(model_name)
  tokenizer = T5Tokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)


class huggr_bert():
  def __init__(self, path = None, gpu = True):
    self.dev = with_gpu if gpu else 'cpu:0'
    self.model = BertModel.from_pretrained(path).to(self.dev)
    self.tokenizer = BertTokenizer.from_pretrained(path)
  def get_embedding(self, content):
    sentence_tensors = []
    for text in content:
      input_ids = self.tokenizer.encode(text, add_special_tokens = True, max_length = 512, truncation = True, return_tensors = 'pt').to(self.dev)
      with torch.no_grad():
        output_tuple = self.model(input_ids)
      output = output_tuple[0].squeeze().mean(dim = 0).squeeze().to("cpu").numpy()
      sentence_tensors.append(output)
    return(sentence_tensors)

  # def preprocess(self, text):
  #   new_text = []
  #   for t in text.split(" "):
  #       t = '@user' if t.startswith('@') and len(t) > 1 else t
  #       t = 'http' if t.startswith('http') else t
  #       new_text.append(t)
  #   return " ".join(new_text)

class huggr_roberta():
  def __init__(self, path = None, gpu = True):
    self.dev = with_gpu if gpu else 'cpu:0'
    self.model = XLMRobertaForMaskedLM.from_pretrained(path).to(self.dev)
    self.tokenizer = XLMRobertaTokenizer.from_pretrained(path)
  def get_embedding(self, text):
    sentence_tensors = []
    for t in text:
      encoded_input = self.tokenizer(t, return_tensors='pt').to(self.dev)
      with torch.no_grad():
        features = self.model(**encoded_input)
      features_ = features[0].detach().to("cpu").numpy() 
      features_mean = np.mean(features_[0], axis=0) 
      sentence_tensors.append(features_mean)
    return(sentence_tensors)

class huggr_bert_class():
  def __init__(self, path = None, gpu = True):
    self.dev = with_gpu if gpu else 'cpu:0'
    self.model = BertForSequenceClassification.from_pretrained(path).to(self.dev)
    self.tokenizer = BertTokenizer.from_pretrained(path)
  def classify(self, content):
    sentence_tensors = []
    for text in content:
      input_ids = self.tokenizer.encode(text, add_special_tokens = True, max_length = 512, truncation = True, return_tensors = 'pt').to(self.dev)
      with torch.no_grad():
        output_tuple = self.model(input_ids)
      output = output_tuple[0].to("cpu").tolist()
      sentence_tensors.append(output)
    return(sentence_tensors)

class huggr_t5():
  def __init__(self, path = None, gpu = True):
    self.dev = with_gpu if gpu else 'cpu:0'
    self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.dev)
    self.tokenizer = T5Tokenizer.from_pretrained(path)
  def generate_text(self, task = "translate English to German: ", text = ["", ""], max_length=500, min_length = 1, length_penalty = 1.0, num_beams = 4):
    output = []
    for sent in text:
      inputs = self.tokenizer.encode(task + sent, return_tensors="pt", truncation=False).to(self.dev)
      out = self.model.generate(
        inputs,
        max_length = max_length, 
        min_length = min_length,
        length_penalty = length_penalty, 
        num_beams = num_beams
        )
      output.append(self.tokenizer.decode(out[0]))
    return(output)

