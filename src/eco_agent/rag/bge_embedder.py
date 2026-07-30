import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma

class BGEEmbedder:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        self.model.to("cpu")

    def embed_documents(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cpu")
            embeddings = self.model(**inputs).last_hidden_state[:,0,:].numpy()
        return embeddings.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]
