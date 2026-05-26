"""
Problem Tanımı: Sözleşme inceleme asistanı
    - Kullanıcının yüklediği bir sözleşme dökümanından bilgi çıkarımı
    - bu içeriği vektörel olarak temsil edelim / embedding
    - faiss ile hızlı arama yapabilen vektör veri tabanı oluştur
    - kullanıcıdan soruları al database e git ve bilgiyi getir, bu bilgi ve kullanıcının sorusu doğrultusunda gemini cevap üretsin

Kullanılan Teknolojiler
    - embedding
    - faiss: hızlı benzerlik araması için db
    - gemini: gemini 2.5 flash

RAG: dil modellerine bilgi desteği sağlayan bir teknik
    - kullanıcı sorusunu alıri ilgil bilgiyi dbden getirir, sonra gemini ile cevap üretir
    - retrieval:
        - kullanıcı sorusunu sorar -> embedding ile vektörleştiririz
        - faiss (db) üzerinden en alakalı içerik (chunk) getirilir
    - Augmention: zenginleştirme, kullanıcı sorusu + prompt + getirilen bilgi
    - Generation: dil modeli ile mantıklı yanıt üretir

Plan/Program:
    - DB işlemleri: build_vector_db.py
        - sözleşme belgesi hazırlama
        - bu belgeyi okuma, metin çıkarma, parçalama (chunk), embedding ve faiss db de depolama
    - Soru cevap sistemi: main.py
        - kullanıcı sorusunu sorar, embedding yapılır, RAG yapılır

pip install google-generativeai python-dotenv sentence-transformers faiss-cpu numpy PyMuPDF
"""

import os
import pickle
import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# initialize gmeini model
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# embeddşng modeli
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# faiss index dosyasını yükle, vektör db
index = faiss.read_index("./data/sozlesme_ornegi.faiss")

# chunklanmıs metin verisi yukle
with open("data/sozlesme_ornegi.pkl","rb") as f:
    chunks = pickle.load(f)

# kullanıcıdan gelen soruları al
while True:

    # kullanıcıdan soru al
    question = input("Sorunuzu giriniz (eng): ")

    if question.lower() in ["quit", "q", "exit"]:
        print("Çıkış yapılıyor")
        break
    
    # kullanıcının sorualrını vektöre çevirelim
    question_embedding = embedding_model.encode([question])

    # faiss veritabanından en yakın 3 chunk aranır ve getirilir
    k = 3 # en yakın 3 chunk
    distances, indices = index.search(np.array(question_embedding),k)

    # bulunan chunklar birleştir, context oluştur
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n ----- \n".join(retrieved_chunks)

    # llm e gönderilecek sistem prompts

    prompt = f"""
                You are a cotract lawyer AI asistant. Based on the contract context below,
                answer the user's question clearly.

                Context:
                {context}

                Question:
                {question}

                Answer:

            """
    # get gemini response
    response = model_gemini.generate_content(prompt)

    print(f"AI: {response.text.strip()}")


"""
FAQ: Software Development Agreement

Q1: Who covers the Google Gemini API and server costs?
A: Unless otherwise specified, all third-party operational costs (API usage fees, cloud infrastructure, etc.) are the responsibility of the Client (YSF A.Ş.). The Contractor is responsible solely for development and integration.

Q2: Does this agreement include post-delivery technical support?
A: No. This agreement covers development only. Once the project is delivered and final payment is made, the contract terminates. Maintenance or support requires a separate, additional agreement.

Q3: Is there a penalty if the project is delayed?
A: The current contract does not include a specific "liquidated damages" clause for delays. If the timeline is critical, both parties may choose to add a penalty clause (e.g., a percentage deduction for every week of delay) before signing.

Q4: How does the Client verify ownership of the code?
A: Upon final payment, the Contractor is contractually obligated to transfer all source code, documentation, and deployment instructions to the Client’s private repository. Once transferred, the Client holds full intellectual property rights.
"""