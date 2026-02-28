"""
ChromaDB 문서 적재 유틸리티

다양한 소스의 문서를 ChromaDB에 벡터화하여 저장합니다.
프로덕션 서비스에서 데이터를 미리 적재할 때 사용합니다.

Usage:
    # Python에서 직접 사용
    from rag.ingest import ingest_pdfs
    ingest_pdfs(["data/nlp.pdf"], persist_directory="./chroma_db")

    # DB 초기화 후 새 데이터 적재
    from rag.ingest import reset_collection, ingest_pdfs
    reset_collection(persist_directory="./chroma_db", collection_name="my_collection")
    ingest_pdfs(["data/nlp.pdf"], persist_directory="./chroma_db", collection_name="my_collection")

    # CLI에서 사용
    python -m rag.ingest data/nlp.pdf data/transformer.pdf

    # CLI: 초기화 후 적재
    python -m rag.ingest --reset data/nlp.pdf data/transformer.pdf
"""

import shutil
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Union

from rag.base import create_embedding_auto as create_embedding


def reset_collection(
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
):
    """
    ChromaDB 컬렉션을 삭제(초기화)합니다.

    persist_directory 내의 특정 컬렉션만 삭제하거나,
    디렉토리 자체를 제거하여 전체 초기화합니다.
    """
    import chromadb

    persist_path = Path(persist_directory)
    if not persist_path.exists():
        print(f"ℹ️ {persist_directory} 가 존재하지 않습니다. 초기화할 내용이 없습니다.")
        return

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
            print(f"🗑️ 컬렉션 '{collection_name}' 삭제 완료")
        else:
            print(f"ℹ️ 컬렉션 '{collection_name}'이 존재하지 않습니다.")

        remaining = client.list_collections()
        if not remaining:
            del client
            shutil.rmtree(persist_directory, ignore_errors=True)
            print(f"🗑️ 빈 DB 디렉토리 삭제: {persist_directory}")
    except Exception as e:
        print(f"⚠️ 컬렉션 삭제 중 오류, 디렉토리 전체 삭제로 전환: {e}")
        shutil.rmtree(persist_directory, ignore_errors=True)
        print(f"🗑️ DB 디렉토리 전체 삭제 완료: {persist_directory}")


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
    import glob as globmod

    args = sys.argv[1:]
    do_reset = "--reset" in args
    if do_reset:
        args.remove("--reset")

    collection = "my_collection"
    persist_dir = "./chroma_db"

    for i, a in enumerate(args):
        if a == "--collection" and i + 1 < len(args):
            collection = args[i + 1]
            args = args[:i] + args[i + 2:]
            break
    for i, a in enumerate(args):
        if a == "--persist-dir" and i + 1 < len(args):
            persist_dir = args[i + 1]
            args = args[:i] + args[i + 2:]
            break

    expanded = []
    for a in args:
        matched = globmod.glob(a)
        expanded.extend(matched if matched else [a])

    if not expanded:
        print("Usage: python -m rag.ingest [--reset] [--collection NAME] [--persist-dir DIR] <파일들...>")
        print("Example: python -m rag.ingest --reset data/*.pdf")
        sys.exit(1)

    if do_reset:
        print("=" * 50)
        print("🔄 DB 초기화 시작")
        print("=" * 50)
        reset_collection(persist_directory=persist_dir, collection_name=collection)

    pdf_files = [f for f in expanded if f.lower().endswith(".pdf")]
    txt_files = [f for f in expanded if not f.lower().endswith(".pdf")]

    if pdf_files:
        print(f"\n📚 PDF 파일 {len(pdf_files)}개 적재 시작...")
        ingest_pdfs(pdf_files, persist_directory=persist_dir, collection_name=collection)

    if txt_files:
        print(f"\n📚 텍스트 파일 {len(txt_files)}개 적재 시작...")
        ingest_documents(file_paths=txt_files, persist_directory=persist_dir, collection_name=collection)
