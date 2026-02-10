def format_docs(docs):
    # return "\n".join(
    #     [
    #         f" {doc.page_content} {doc.metadata['source']} {int(doc.metadata['page'])+1} "
    #         for doc in docs
    #     ]
    # )
    """
    검색된 문서 리스트를 프롬프트에 넣기 좋은 문자열로 변환.

    기존 구현은 PDF 전용 메타데이터(특히 'page')가 있다고 가정해서
    doc.metadata['page'] 접근 중 KeyError가 발생할 수 있었다.

    이 버전은:
    - 'source'가 없으면 file_path / path / "" 순으로 대체
    - 'page'가 없으면 페이지 정보를 생략
    - page가 있으면 1-based 페이지 번호로 표시
    """
    lines = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        source = (
            meta.get("source")
            or meta.get("file_path")
            or meta.get("path")
            or ""
        )

        page = meta.get("page", None)
        if page is not None:
            try:
                page_str = str(int(page) + 1)  # 0-based → 1-based
            except Exception:
                page_str = str(page)
        else:
            page_str = ""

        line = f" {doc.page_content} {source} {page_str} "
        lines.append(line)

    return "\n".join(lines)


def format_searched_docs(docs):
    return "\n".join(
        [
            f" {doc['content']} {doc['url']} "
            for doc in docs
        ]
    )


def format_task(tasks):
    # 결과를 저장할 빈 리스트 생성
    task_time_pairs = []

    # 리스트를 순회하면서 각 항목을 처리
    for item in tasks:
        # 콜론(:) 기준으로 문자열을 분리
        task, time_str = item.rsplit(":", 1)
        # '시간' 문자열을 제거하고 정수로 변환
        time = int(time_str.replace("시간", "").strip())
        # 할 일과 시간을 튜플로 만들어 리스트에 추가
        task_time_pairs.append((task, time))

    # 결과 출력
    return task_time_pairs

