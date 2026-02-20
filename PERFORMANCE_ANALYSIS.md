# LLM API 호출 vs 로컬 GPU 실행 시간 차이 분석

## 📊 개요

이 문서는 현재 프로젝트에서 **HuggingFace API 호출 방식**과 **로컬 GPU(3090) 실행 방식** 간의 시간적 차이점을 분석합니다.

---

## 🔍 현재 프로젝트 구조

### 사용 중인 모델
- **라우팅/판단/요약 LLM**: `Qwen/Qwen2.5-14B-Instruct` (14B 파라미터)
- **답변 생성 LLM**: `Qwen/Qwen2.5-14B-Instruct` (14B 파라미터)
- **임베딩**: `BAAI/bge-m3` (로컬/API 선택 가능)

### 현재 구현 방식
- **LLM**: `HuggingFaceEndpoint` → API 호출 방식
- **임베딩**: `create_embedding_auto()` → 환경변수로 로컬/API 선택

---

## ⏱️ 시간 차이 분석

### 1. 초기화 단계 (최초 1회)

#### API 호출 방식
```
초기화 시간: ~0.5-2초
- HuggingFace 로그인: ~0.1-0.5초
- HuggingFaceEndpoint 객체 생성: ~0.1-0.3초
- 네트워크 연결 확인: ~0.3-1.2초
```

#### 로컬 GPU 방식 (3090)
```
초기화 시간: ~10-30초 (최초 실행 시)
- 모델 다운로드: ~5-15분 (최초 1회만, 이후 캐시 사용)
- 모델 로딩 (VRAM): ~5-15초
- CUDA 초기화: ~1-3초
- 메모리 할당: ~2-5초
```

**차이점**:
- **최초 실행**: 로컬이 훨씬 느림 (모델 다운로드 필요)
- **이후 실행**: 로컬이 약간 느림 (모델 로딩 필요)

---

### 2. 단일 추론 요청 (Single Inference)

#### API 호출 방식
```
전체 소요 시간: ~2-8초
├─ 네트워크 지연 (요청 전송): ~0.1-0.5초
├─ API 서버 처리: ~1-5초
│  ├─ 모델 로딩 (서버 측): 이미 로드됨
│  ├─ 토큰 생성: ~1-4초 (토큰 수에 비례)
│  └─ 응답 포맷팅: ~0.1-0.3초
└─ 네트워크 지연 (응답 수신): ~0.1-0.5초
```

**변동 요인**:
- 네트워크 상태 (지연 시간)
- API 서버 부하
- 무료 티어의 경우 큐 대기 시간 추가 (~1-5초)

#### 로컬 GPU 방식 (3090)
```
전체 소요 시간: ~0.5-3초
├─ GPU 메모리 접근: ~0.01-0.05초
├─ 토큰 생성: ~0.3-2초 (토큰 수에 비례)
│  ├─ Forward pass: ~0.1-0.5초/토큰
│  └─ 512 토큰 생성 시: ~0.3-2초
└─ 결과 반환: ~0.01-0.05초
```

**변동 요인**:
- 생성 토큰 수
- 배치 크기
- GPU 메모리 상태

**차이점**:
- **짧은 응답 (50-100 토큰)**: 로컬이 **2-5배 빠름**
- **중간 응답 (200-300 토큰)**: 로컬이 **3-8배 빠름**
- **긴 응답 (500+ 토큰)**: 로컬이 **5-15배 빠름**

---

### 3. LangGraph 파이프라인 전체 실행 시간

현재 프로젝트의 LangGraph 워크플로우는 다음과 같은 노드들을 포함합니다:

```
START → contextualize → retrieve_or_not
         ├─ retrieve → relevance_check → llm_answer
         └─ llm_answer → save_memory → END
```

#### API 호출 방식 (전체 파이프라인)
```
전체 소요 시간: ~8-25초
├─ contextualize 노드: ~2-6초
│  └─ LLM 호출 (판단/재작성): ~2-6초
├─ retrieve 노드: ~0.1-0.5초
│  └─ 벡터 검색 (로컬): ~0.1-0.5초
├─ relevance_check 노드: ~2-5초
│  └─ LLM 호출 (관련성 평가): ~2-5초
├─ llm_answer 노드: ~3-10초
│  └─ LLM 호출 (답변 생성): ~3-10초
└─ save_memory 노드: ~1-4초 (조건부)
   └─ LLM 호출 (요약): ~1-4초
```

**총 LLM 호출 횟수**: 2-4회 (경로에 따라)

#### 로컬 GPU 방식 (전체 파이프라인)
```
전체 소요 시간: ~2-8초
├─ contextualize 노드: ~0.5-2초
│  └─ LLM 추론: ~0.5-2초
├─ retrieve 노드: ~0.1-0.5초
│  └─ 벡터 검색 (로컬): ~0.1-0.5초
├─ relevance_check 노드: ~0.5-1.5초
│  └─ LLM 추론: ~0.5-1.5초
├─ llm_answer 노드: ~0.5-3초
│  └─ LLM 추론: ~0.5-3초
└─ save_memory 노드: ~0.5-1.5초 (조건부)
   └─ LLM 추론: ~0.5-1.5초
```

**차이점**:
- **전체 파이프라인**: 로컬이 **3-5배 빠름**
- **네트워크 지연 누적 효과**: API 방식은 각 호출마다 네트워크 지연 발생

---

### 4. 동시 요청 처리 (Concurrent Requests)

#### API 호출 방식
```
동시 처리 능력: 제한적
- 무료 티어: 동시 요청 제한 (보통 1-2개)
- 유료 티어: 동시 요청 가능하나 비용 증가
- 각 요청마다 네트워크 지연 발생
```

#### 로컬 GPU 방식 (3090)
```
동시 처리 능력: 배치 처리 가능
- 단일 GPU에서 배치 크기 조절 가능
- 배치 크기 4-8: 약 1.5-2배 속도 향상
- 네트워크 지연 없음
```

**차이점**:
- **동시 요청이 많은 경우**: 로컬이 **5-10배 빠름**
- **배치 처리 최적화**: 로컬에서 더 효과적

---

## 📈 성능 비교 요약

| 항목 | API 호출 방식 | 로컬 GPU (3090) | 개선율 |
|------|--------------|----------------|--------|
| **초기화 (최초)** | 0.5-2초 | 10-30초 | - |
| **초기화 (이후)** | 0.5-2초 | 5-15초 | - |
| **단일 추론 (50 토큰)** | 2-4초 | 0.3-0.8초 | **3-5배 빠름** |
| **단일 추론 (500 토큰)** | 5-10초 | 0.5-2초 | **5-10배 빠름** |
| **전체 파이프라인** | 8-25초 | 2-8초 | **3-5배 빠름** |
| **동시 요청 처리** | 제한적 | 배치 처리 가능 | **5-10배 빠름** |
| **네트워크 의존성** | 필수 | 없음 | - |
| **오프라인 사용** | 불가능 | 가능 | - |

---

## 🎯 로컬 GPU 전환 시 예상 개선 효과

### 1. 사용자 경험 개선
- **응답 시간**: 평균 **3-5배 단축**
- **대기 시간 감소**: 네트워크 지연 제거
- **안정성 향상**: API 서버 장애 영향 없음

### 2. 비용 절감
- **API 호출 비용**: 제로
- **무료 티어 제한**: 없음
- **사용량 제한**: 없음

### 3. 프라이버시 향상
- **데이터 전송**: 로컬 처리로 외부 전송 없음
- **민감 정보**: 로컬에서만 처리

---

## ⚠️ 로컬 GPU 전환 시 고려사항

### 1. 하드웨어 요구사항
- **VRAM**: Qwen2.5-14B-Instruct는 약 **28-32GB VRAM** 필요
- **3090 (24GB)**: **양자화(Quantization) 필수**
  - 8-bit 양자화: ~14GB VRAM
  - 4-bit 양자화: ~8GB VRAM

### 2. 구현 변경 필요사항
- `HuggingFaceEndpoint` → `ChatHuggingFace` (로컬 모델)
- 모델 로딩 코드 추가
- 양자화 설정 추가

### 3. 성능 최적화
- **Flash Attention**: 메모리 효율성 향상
- **KV Cache**: 반복 추론 속도 향상
- **배치 처리**: 동시 요청 처리 최적화

---

## 🔧 로컬 GPU 전환 구현 예시

### 현재 코드 (API 방식)
```python
def _init_chat_model():
    llm = HuggingFaceEndpoint(
        repo_id=ROUTER_MODEL,
        task="text-generation",
        temperature=0.7,
        max_new_tokens=512,
    )
    _chat_hf = ChatHuggingFace(llm=llm)
```

### 변경 후 코드 (로컬 GPU)
```python
def _init_chat_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from langchain_huggingface import ChatHuggingFace
    
    model_name = ROUTER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 양자화 적용 (3090 24GB에 맞춤)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,  # 8-bit 양자화
        torch_dtype=torch.float16,
    )
    
    llm = ChatHuggingFace.from_model_id(
        model_id=model_name,
        task="text-generation",
        model_kwargs={
            "device_map": "auto",
            "load_in_8bit": True,
        },
    )
    _chat_hf = llm
```

---

## 📝 결론

### API 호출 방식의 장점
- ✅ 초기 설정 간단
- ✅ 하드웨어 요구사항 낮음
- ✅ 모델 업데이트 자동 반영

### 로컬 GPU 방식의 장점
- ✅ **응답 속도 3-5배 향상**
- ✅ 네트워크 지연 제거
- ✅ 비용 절감 (장기적)
- ✅ 오프라인 사용 가능
- ✅ 프라이버시 향상

### 권장사항
**3090 GPU를 사용할 수 있는 환경이라면 로컬 GPU 방식으로 전환하는 것을 강력히 권장합니다.**

특히:
- **대화형 애플리케이션**: 응답 속도가 사용자 경험에 직접적 영향
- **빈번한 요청**: 네트워크 지연 누적 효과가 큼
- **프라이버시 중요**: 민감한 문서 처리 시

---

## 📚 참고 자료

- [LangChain HuggingFace Integration](https://python.langchain.com/docs/integrations/llms/huggingface)
- [Transformers Quantization Guide](https://huggingface.co/docs/transformers/quantization)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
