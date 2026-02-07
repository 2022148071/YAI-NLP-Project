from abc import ABC, abstractmethod
from operator import itemgetter
import os

# LangChain Core 및 Community
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_classic import hub

# Hugging Face 관련 라이브러리 (OpenAI 대체)
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 5

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        """
        Embeddings 생성
        - OpenAIEmbeddings 대신 HuggingFaceEmbeddings 사용
        - model_name: 한국어 성능이 좋은 모델 (jhgan/ko-sroberta-multitask)
        - device: 'cpu' (Mac은 'mps', NVIDIA GPU는 'cuda'로 변경 가능)
        """
        return HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vectorstore(self, split_docs):
        """VectorStore 생성 (FAISS)"""
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        """Retriever 생성 (MMR 방식 등)"""
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        """
        LLM 모델 생성
        - HuggingFaceEndpoint: 무료 추론 API 사용
        - repo_id: HuggingFaceH4/zephyr-7b-beta (무료, 고성능, 승인 불필요)
        """
        # 환경 변수에서 토큰 확인
        api_token = os.environ.get("HF_API_KEY")
        if not api_token:
            raise ValueError("HF_API_KEY 환경 변수가 설정되지 않았습니다.")

        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            #repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="conversational",
            max_new_tokens=512,
            temperature=0.1,
            huggingfacehub_api_token=api_token
        )
        return ChatHuggingFace(llm=llm)

    def create_prompt(self):
        """Prompt Template 로드"""
        # 기존 프롬프트 사용 (필요시 모델에 맞춰 변경 가능)
        return hub.pull("teddynote/rag-prompt-chat-history")

    def create_prompt_new(self):
        system_template = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        return prompt

    @staticmethod
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])

    def create_chain(self):
        """RAG 체인 구성"""
        # 1. 문서 로드
        docs = self.load_documents(self.source_uri)
        
        # 2. 텍스트 분할
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        
        # 3. 벡터 저장소 및 리트리버 생성
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        
        # 4. 모델 및 프롬프트 준비
        model = self.create_model()
        prompt = self.create_prompt_new()
        
        # 5. 체인 연결
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self