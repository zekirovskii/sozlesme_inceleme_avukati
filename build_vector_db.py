"""
- DB işlemleri: build_vector_db.py
    - sözleşme belgesi hazırlama
    - bu belgeyi okuma, metin çıkarma, parçalama (chunk), embedding ve faiss db oluşturma
"""

import os
import fitz
from sentence_transformers import SentenceTransformer # embedding
import faiss # vektör veritabanı
import numpy as np
import pickle # vektör db yi kaydetmek için

# program için dosya olarak .pdf yükleyelim
# .pdften metin donusumu yapacak fonksiyon

def extract_text_from_pdf(pdf_path):
    """
    pdf dosyasından metin okuma
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text = text + page.get_text()

    return text

# print(extract_text_from_pdf("./data/sozlesme_ornegi.pdf"))

# uzun metni daha kucuk parçalara böl (chunk)

def chunk_text(text, max_length = 500):
    """
    metni belirtilen karakter uzunluguna böl
    """

    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_length:
            current += " " + line.strip()
        else:
            chunks.append(current.strip())
            current = line.strip()
    if current:
        chunks.append(current.strip())

    return chunks

# text = extract_text_from_pdf("./data/sozlesme_ornegi.pdf")
# print(chunk_text(text))

pdf_file_path = "./data/sozlesme_ornegi.pdf"

# pdften metin çıkarma
text = extract_text_from_pdf(pdf_file_path)

# metni chunklara bölelim
chunks = chunk_text(text)

# her chunk için embedding yani vektörel temsik oluşturur
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print(embeddings.shape)

# faiss index oluştur
dimension = embeddings.shape[1] # embedding vektör boyutu
index = faiss.IndexFlatL2(dimension) # 12 norm kullanarak benzerlik arama
index.add(np.array(embeddings)) # embeddingleri indexe kaydeder

# faiss indexi ve chunkları kaydet
faiss.write_index(index, "data/sozlesme_ornegi.faiss")
with open("data/sozlesme_ornegi.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index ve chunklar kaydedildi")