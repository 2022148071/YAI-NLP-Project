# Data 폴더

PDF 파일을 저장하는 폴더입니다.

## 파일 업로드 방법

### 방법 1: Python 스크립트 사용 (권장)
```python
from copy_pdf import copy_pdf_with_korean_name

# 한글 파일명이 포함된 PDF 복사
copy_pdf_with_korean_name("원본파일경로/한글파일명.pdf")
```

### 방법 2: 터미널에서 직접 복사
```powershell
# PowerShell에서
Copy-Item "원본경로\한글파일명.pdf" -Destination "data\"
```

### 방법 3: 드래그 앤 드롭 (영문 파일명)
- 파일명을 영문으로 변경 후 드래그 앤 드롭
