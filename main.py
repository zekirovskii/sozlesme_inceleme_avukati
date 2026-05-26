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