## 🧩 노드 구성

| 노드(Label) | 주요 속성(Properties) | 설명 |
|--------------|----------------------|------|
| `:Trace` | `id` | 하나의 실행 흐름(트레이스)을 의미 |
| `:Summary` | `text`, `embedding` | 트레이스 요약 내용 및 임베딩 벡터 |
| `:Technique` | `id` | MITRE ATT&CK 기술 ID (`T1055` 등) |
| `:Process` | `guid` | 실행된 프로세스 식별자 |
| `:File` | `name` | 접근/로드된 파일 이름 |
| `:IpAddress` | `address` | 외부 통신 대상 IP |
| `:RegistryKey` | `path` | 접근 또는 수정된 레지스트리 키 경로 |

---

## 🔗 관계(Relationships)

| 관계 타입 | 방향 | 의미 |
|-------------|--------|------|
| `[:HAS_SUMMARY]` | `(:Trace) → (:Summary)` | 트레이스의 요약을 연결 |
| `[:MENTIONS]` | `(:Summary) → (:Technique)` | 요약에서 특정 공격 기술을 언급 |
| `[:INDICATES_TECHNIQUE]` | `(:Trace) → (:Technique)` | 트레이스가 특정 MITRE 기법을 나타냄 |
| `[:PARTICIPATED_IN]` | `(:Process)`, `(:File)`, `(:IpAddress)` → `(:Trace)` | 트레이스에 참여한 개체들 |
| `[:MODIFIED_KEY]` | `(:Process) → (:RegistryKey)` | 프로세스가 레지스트리 키를 수정 |
| *(또 다른)* `[:PARTICIPATED_IN]` | `(:File) → (:Trace)` | 파일이 해당 트레이스에서 사용됨 |

---

## 분석 흐름

### 1. 트레이스 입력
- JSON 형식 트레이스 파일 또는 Kafka 등 실시간 스트림에서 입력 가능
- 예시 파일 경로: `C:\Users\KISIA\Downloads\data\~trace-xxxxx.json`

### 2. 트레이스 분석
- `create_summary_context()` → 트레이스 이벤트 그룹화 & 요약 컨텍스트 생성
- `summarize_trace_with_llm()` → LLM 호출, 자연어 요약 반환

### 3. 유사 트레이스 검색 및 구조적 분석
- `find_similar_traces()` → Neo4j DB에서 의미적 유사도 상위 트레이스 선택
- 구조적 유사성 분석 → 공통 엔티티 및 공격 기술(TTP) 확인

### 4. 간접 연결 탐색
- Neo4j 그래프에서 엔티티 간 최단 경로 탐색 (1~2 hops)
- 공격 연관성 및 간접 관계 확인

[
  {
    "e1_name": "svchost.exe",
    "e2_name": "malware.exe",
    "hops": 2,
    "path_nodes": [
      "Process:svchost.exe",
      "User:NT AUTHORITY\\SYSTEM",
      "Process:malware.exe"
    ]
  }
]
<br>
svchost.exe → malware.exe가 직접 연결되어 있지 않아도, 사용자 계정 NT AUTHORITY\SYSTEM을 통해 간접적으로 연결됨을 확인 가능. 이를 통해 공격자가 정상 프로세스를 이용해 악성 파일을 실행했을 가능성을 탐지할 수 있음.

### 5. 대응 제안 생성
- `generate_mitigation_prompt()` → LLM 프롬프트 구성
- LLM 호출 → 단계별 대응 방안 생성 (탐지/격리, 네트워크 차단, 로그 모니터링, 예방 전략)

## 실행 방법
kafka 토픽: trace_analysis

- producer.py 파일 실행 시 'trace_analysis' 토픽으로 실시간 스트리밍
