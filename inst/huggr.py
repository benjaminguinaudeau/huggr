import torch
import os
from transformers import BertModel, BertTokenizer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with_gpu = 'cuda' if torch.cuda.is_available() else None
print(with_gpu)

def bert_download(model_name, path):
  model = BertModel.from_pretrained(model_name)
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)


class huggr():
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
