"""
ChromaDB 문서 적재 유틸리티

다양한 소스의 문서를 ChromaDB에 벡터화하여 저장합니다.
프로덕션 서비스에서 데이터를 미리 적재할 때 사용합니다.

Usage:
    # Python에서 직접 사용
    from rag.ingest import ingest_pdfs
    ingest_pdfs(["data/nlp.pdf"], persist_directory="./chroma_db")

    # CLI에서 사용
    python -m rag.ingest data/nlp.pdf data/transformer.pdf
"""

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Union

# ⚠️ 임베딩 모델은 base.py에서 중앙 관리 (적재/검색 시 동일 모델 보장)
from rag.base import create_embedding_auto as create_embedding


def ingest_pdfs(
    pdf_paths: List[str],
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    PDF 파일들을 ChromaDB에 적재합니다.

    Args:
        pdf_paths: PDF 파일 경로 리스트
        persist_directory: ChromaDB 저장 디렉토리
        collection_name: 컬렉션 이름
        chunk_size: 텍스트 분할 크기
        chunk_overlap: 텍스트 분할 오버랩

    Returns:
        Chroma 벡터스토어 인스턴스
    """
    # 1. 문서 로드
    docs = []
    for path in pdf_paths:
        loader = PDFPlumberLoader(path)
        loaded = loader.load()
        docs.extend(loaded)
        print(f"  📄 {path}: {len(loaded)}페이지 로드")

    print(f"📄 총 {len(docs)}개 페이지 로드 완료")

    # 2. 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    print(f"✂️ 총 {len(split_docs)}개 청크로 분할 완료")

    # 3. ChromaDB에 저장
    embedding = create_embedding()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(
        f"✅ ChromaDB 저장 완료: {persist_directory} "
        f"(collection: {collection_name}, {len(split_docs)}개 청크)"
    )

    return vectorstore


def ingest_documents(
    file_paths: Optional[List[str]] = None,
    documents: Optional[List[Document]] = None,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    텍스트 파일 경로 또는 LangChain Document 리스트를 ChromaDB에 적재합니다.

    Args:
        file_paths: 텍스트 파일 경로 리스트 (.txt, .md 등)
        documents: 이미 로드된 LangChain Document 리스트 (file_paths와 택 1)
        persist_directory: ChromaDB 저장 디렉토리
        collection_name: 컬렉션 이름
        chunk_size: 텍스트 분할 크기
        chunk_overlap: 텍스트 분할 오버랩

    Returns:
        Chroma 벡터스토어 인스턴스

    Usage:
        # 파일 경로로 적재
        ingest_documents(file_paths=["highmath12.txt", "notes.md"])

        # 이미 로드된 Document 객체로 적재
        ingest_documents(documents=[Document(page_content="...", metadata={...})])
    """
    if file_paths is None and documents is None:
        raise ValueError("file_paths 또는 documents 중 하나는 반드시 제공해야 합니다.")

    # 파일 경로가 주어진 경우 → 텍스트 파일 로드
    docs = []
    if file_paths:
        for path in file_paths:
            try:
                loader = TextLoader(path, encoding="utf-8")
                loaded = loader.load()
                docs.extend(loaded)
                print(f"  📄 {path}: {len(loaded)}개 문서 로드")
            except Exception as e:
                print(f"  ⚠️ {path} 로드 실패: {e}")
        print(f"📄 총 {len(docs)}개 문서 로드 완료")

    # 이미 로드된 Document 객체가 주어진 경우
    if documents:
        docs.extend(documents)
        print(f"📄 Document 객체 {len(documents)}개 추가")

    if not docs:
        print("⚠️ 적재할 문서가 없습니다.")
        return None

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    print(f"✂️ 총 {len(split_docs)}개 청크로 분할 완료")

    # ChromaDB에 저장
    embedding = create_embedding()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(
        f"✅ ChromaDB 저장 완료: {persist_directory} "
        f"(collection: {collection_name}, {len(split_docs)}개 청크)"
    )

    return vectorstore


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingest <pdf_path1> [pdf_path2] ...")
        print("Example: python -m rag.ingest data/nlp.pdf data/transformer.pdf")
        sys.exit(1)

    ingest_pdfs(sys.argv[1:])
