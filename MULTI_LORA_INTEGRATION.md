# Multi-LoRA 통합 작업 컨텍스트

## 목적

기존 단일 LLM(`Qwen/Qwen2.5-14B-Instruct`)으로 답변을 생성하던 RAG 파이프라인에,
**유저 쿼리에 따라 4개의 LoRA 스타일 중 하나를 자동 선택**하여 답변하는 구조를 도입한다.

GPU 사용이 불가능하여 HuggingFace Inference API를 통해 호출해야 한다.

## 아키텍처 변경 요약

```
[변경 전]
쿼리 → _chain (단일 모델) → 답변

[변경 후]
쿼리 → embed(BGE-M3) → centroid 코사인 유사도 비교 → style 선택
     → _chains[style] (스타일별 merged 모델 API) → 답변
```

## 4개 LoRA 스타일

|     스타일    |    설명    | HuggingFace 레포 |
|--------------|----------------------------|----------------------------|
|    direct    | 직접적 답변 | `RiverWon/NeuLoRA-direct` |
|   socratic   | 소크라테스식 질문 유도 | `RiverWon/NeuLoRA-socratic` |
|  scaffolding | 단계적 힌트 제공 | `RiverWon/NeuLoRA-scaffolding` |
|   feedback   | 피드백 기반 | `RiverWon/NeuLoRA-feedback` |

- base model: `Qwen/Qwen2.5-14B-Instruct`
- LoRA 원본: `marimmo/multi-lora` (adapter_config: r=16, alpha=32)
- merge 방식: safetensors 샤드별 수학적 merge (W_merged = W_base + B@A * alpha/r)
- 각 레포는 29.6GB, 8 shards, Qwen2ForCausalLM 구조

## 라우터

- 파일: `LangGraph/router_model.json` (출처: `marimmo/multi-lora/router/router_model.json`)
- 방식: hash 기반 centroid classifier
- 동작: 쿼리를 BGE-M3로 임베딩 → 4개 스타일 centroid와 코사인 유사도 비교 → 최대 유사도 스타일 선택
- GPU 불필요 (numpy 연산만)

## 완료된 작업

1. LoRA 4개를 base model에 merge하여 개별 HuggingFace 레포에 업로드 완료
2. `router_model.json`을 `LangGraph/` 폴더에 배치 완료
3. `rag/base.py` 수정 완료:
   - `STYLE_MODELS` 상수 추가
   - `create_model(self, model_name=None)` 파라미터 추가
   - API 분기에서 `repo_id = model_name or ANSWER_MODEL` 적용
4. `rag/chroma.py` 부분 수정:
   - `create_chain()` 내부에서 `self.create_model(model_name=self.model_name)` 호출
5. `LangGraph/LangGraph.py` 부분 수정:
   - `import numpy as np` 추가
   - `LORA_ROUTER_PATH`, `STYLE_MODELS` 상수 추가
   - `GraphState`에 `style` 필드 추가
   - `_init_lora_router()`, `route_style()` 함수 추가
   - `initialize()`에 `_init_lora_router()` 호출 추가
   - `llm_answer()`에 `route_style()` 호출 추가
   - `requirements.txt`에 `numpy` 추가

## 현재 코드에 남아있는 버그 (수정 필요)

### 버그 1: `LangGraph.py` 145~146번 줄 — 변수 선언 문법 오류

**현재 (잘못됨):**
```python
_chains = Dict[str, Any] = {}
_centroids = Dict[str, list] = {}
```

**수정:**
```python
_chains: Dict[str, Any] = {}
_centroids: Dict[str, list] = {}
```

`=` → `:` (타입 힌트 문법)

### 버그 2: `LangGraph.py` `_init_rag_chain()` — for 루프 누락

**현재 (260~280번 줄, 잘못됨):**
```python
def _init_rag_chain(...):
    global _retriever, _chain, _answer_model_used   # _chain은 삭제된 변수
    ...
    rag = ChromaRetrievalChain(
        ...,
        model_name = model_name,    # model_name 미정의
    ).create_chain()
    if _retriever is None:
        _retriever = rag.retriever
    _chains[style] = rag.chain      # style 미정의
```

**수정:**
```python
def _init_rag_chain(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_MAIN,
    k: int = 10,
):
    global _retriever, _chains
    _log("🚀 스타일별 RAG 체인 생성 시작...")

    for style, model_name in STYLE_MODELS.items():
        _log(f"  ⏳ {style} 체인 생성 중... ({model_name})")
        rag = ChromaRetrievalChain(
            persist_directory=persist_directory,
            collection_name=collection_name,
            k=k,
            model_name=model_name,
        ).create_chain()

        if _retriever is None:
            _retriever = rag.retriever
        _chains[style] = rag.chain
        _log(f"  ✅ {style} 체인 생성 완료")

    _log(f"✅ 전체 RAG 체인 생성 완료: {list(_chains.keys())}")
```

### 버그 3: `LangGraph.py` `llm_answer()` — 여전히 `_chain` 사용

**현재 (576번 줄, 잘못됨):**
```python
style = route_style(question)
_log(f"🎯 LoRA 스타일 선택: {style}")
try:
    response = _chain.invoke(...)      # _chain은 삭제된 변수
```

**수정:**
```python
style = route_style(question)
_log(f"🎯 LoRA 스타일 선택: {style}")
chain = _chains.get(style) or _chains.get("direct")
try:
    response = chain.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history,
            "policy": policy,
        }
    )
```

반환값에도 `style` 추가:
```python
return GraphState(
    answer=response,
    style=style,
    messages=[("user", question), ("assistant", response)],
)
```

### 버그 4: `rag/chroma.py` `__init__` — `model_name` 파라미터 누락

**현재 (31~41번 줄, 잘못됨):**
```python
def __init__(
    self,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    k: int = 10,
):                                      # model_name 파라미터가 없음
    super().__init__()
    self.persist_directory = persist_directory
    self.collection_name = collection_name
    self.k = k
    self.model_name = model_name        # NameError 발생
```

**수정:**
```python
def __init__(
    self,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    k: int = 10,
    model_name: str | None = None,
):
    super().__init__()
    self.persist_directory = persist_directory
    self.collection_name = collection_name
    self.k = k
    self.model_name = model_name
```

## 관련 파일 목록

| 파일 | 역할 |
|---|---|
| `LangGraph/LangGraph.py` | 메인 파이프라인 (LangGraph 기반) |
| `rag/base.py` | RetrievalChain 추상 클래스, create_model(), STYLE_MODELS |
| `rag/chroma.py` | ChromaDB 기반 RAG 체인 구현 |
| `LangGraph/router_model.json` | LoRA 스타일 라우터 centroid 벡터 |
| `requirements.txt` | 프로젝트 의존성 |

## HuggingFace 리소스

| 리소스 | URL |
|---|---|
| LoRA 원본 (어댑터) | https://huggingface.co/marimmo/multi-lora |
| 라우터 JSON | https://huggingface.co/marimmo/multi-lora/resolve/main/router/router_model.json |
| merged: direct | https://huggingface.co/RiverWon/NeuLoRA-direct |
| merged: socratic | https://huggingface.co/RiverWon/NeuLoRA-socratic |
| merged: scaffolding | https://huggingface.co/RiverWon/NeuLoRA-scaffolding |
| merged: feedback | https://huggingface.co/RiverWon/NeuLoRA-feedback |
