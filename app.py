from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from dotenv import load_dotenv
load_dotenv('.env')
from transformers import BertTokenizerFast
import pinecone_text
from sentence_transformers import SentenceTransformer
import torch

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

def data_loader(file_sets, ):
    # Đọc dữ liệu từ các file .csv và trả lại toàn bộ dữ liệu
    sum_data = []
    for file_path in file_sets:
        loader = CSVLoader(file_path=f'./{file_path}',
                           
                            encoding='UTF-8')
        data = loader.load()
        sum_data+=data
    return sum_data
file_sets = ['Question_Sheet1.csv']
data = data_loader(file_sets)
# Chia nhỏ dữ liệu
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 0)
texts = text_splitter.split_documents(data)
# Kết nối với index của Pinecone
index_name = 'index1'
index = pinecone.Index(index_name)

# Khởi tạo vector BM25 - sparse vector
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-uncased'
)
def tokenize_func(text):
    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )['input_ids']
    return tokenizer.convert_ids_to_tokens(token_ids)
bm25 = pinecone_text.BM25(tokenize_func)
bm25.fit(data)
# Khởi tạo Dense vectors

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)