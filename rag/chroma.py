"""
ChromaDB 기반 RAG 체인
- 기존에 데이터가 적재된 ChromaDB 컬렉션에 연결하여 검색
- PDF 로딩/분할 과정 없이 바로 벡터스토어 활용
"""

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

from rag.base import RetrievalChain


class ChromaRetrievalChain(RetrievalChain):
    """
    ChromaDB 기반 RetrievalChain

    기존 PDFRetrievalChain과 동일한 인터페이스(retriever, chain)를 제공하되,
    PDF를 매번 로드/분할/임베딩하는 대신 이미 적재된 ChromaDB에 연결합니다.

    Usage:
        rag = ChromaRetrievalChain(
            persist_directory="./chroma_db",
            collection_name="my_collection",
        ).create_chain()

        retriever = rag.retriever
        chain = rag.chain
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "default",
        k: int = 10,
    ):
        super().__init__()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.k = k

    # ── 추상 메서드 구현 (ChromaDB에서는 사용하지 않음) ──────────────
    def load_documents(self, source_uris):
        """ChromaDB는 이미 문서가 저장되어 있으므로 로딩 불필요"""
        return []

    def create_text_splitter(self):
        """ChromaDB는 이미 분할된 문서가 저장되어 있으므로 불필요"""
        return None

    # ── 핵심: create_chain 오버라이드 ─────────────────────────────
    def create_chain(self):
        """
        ChromaDB 기반 RAG 체인 구성

        기존 base.py의 create_chain()은:
          load_documents → split → create_vectorstore → retriever → chain

        이 버전은:
          기존 ChromaDB 연결 → retriever → chain
        """
        # 1. 임베딩 모델 (base.py의 HuggingFaceEmbeddings 재사용)
        embedding = self.create_embedding()

        # 2. 기존 ChromaDB 컬렉션에 연결 (새로 생성하지 않음!)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=embedding,
        )

        doc_count = self.vectorstore._collection.count()
        print(f"📚 ChromaDB 연결 완료: {doc_count}개 문서 (collection: {self.collection_name})")

        if doc_count == 0:
            print("⚠️ 경고: 컬렉션에 문서가 없습니다. rag.ingest로 먼저 문서를 적재하세요.")

        # 3. 리트리버 생성 (base.py 메서드 재사용)
        self.retriever = self.create_retriever(self.vectorstore)

        # 4. LLM 모델 및 프롬프트 (base.py 메서드 재사용)
        model = self.create_model()
        prompt = self.create_prompt_new()

        # 5. 체인 연결
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
                "policy": itemgetter("policy"),
            }
            | prompt
            | model
            | StrOutputParser()
        )

        return self
