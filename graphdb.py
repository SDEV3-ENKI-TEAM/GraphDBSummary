import json
import os
from neo4j import GraphDatabase
import numpy as np
from llm import llm
from collections import Counter
from sentence_transformers import SentenceTransformer

# --- Neo4j 연결 정보 ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "PW")
DATABASE = "neo4j"
driver = GraphDatabase.driver(URI, auth=AUTH)

PROMPT_FILE = "C:\\Users\\KISIA\\Desktop\\Enki\\Neo4j\\summary_prompt.md"

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def create_summary_context(trace_data):
    context_lines = []

    def get_tag_value(tags, key, default=None):
        for tag in tags:
            if tag.get('key') == key:
                return tag.get('value')
        return default

    sigma_alerts = Counter()
    process_flows = Counter()
    network_events = Counter()
    file_events = Counter()
    registry_events = Counter()

    for span in trace_data.get('spans', []):
        tags = span.get('tags', [])
        event_name = get_tag_value(tags, 'EventName', '')
        process_image = get_tag_value(tags, 'Image')
        process_name = os.path.basename(process_image or 'N/A')

        # Sigma 룰 탐지
        rule_title = get_tag_value(tags, 'sigma.rule_title')
        if rule_title:
            sigma_alerts[f"규칙: {rule_title}, 프로세스: {process_name}"] += 1

        # 프로세스 생성 흐름
        if "ProcessCreate" in event_name:
            parent_image = get_tag_value(tags, 'ParentImage')
            if parent_image and process_image:
                process_flows[f"'{os.path.basename(parent_image)}'가 '{process_name}'를 실행"] += 1

        # 네트워크/파일/레지스트리 이벤트
        if "NetworkConnect" in event_name:
            dest_ip = get_tag_value(tags, 'DestinationIp')
            dest_port = get_tag_value(tags, 'DestinationPort')
            if dest_ip and dest_port:
                network_events[f"[네트워크] '{process_name}'가 '{dest_ip}:{dest_port}'로 연결"] += 1
        elif "FileCreate" in event_name:
            target_file = get_tag_value(tags, 'TargetFilename')
            if target_file:
                file_events[f"[파일] '{process_name}'가 '{target_file}' 파일을 생성"] += 1
        elif "RegistryValueSet" in event_name:
            target_object = get_tag_value(tags, 'TargetObject')
            if target_object:
                registry_events[f"[레지스트리] '{process_name}'가 '{target_object}' 키 값을 수정"] += 1

    # 컨텍스트 생성
    if sigma_alerts:
        context_lines.append("### Sigma Rule 탐지 요약:")
        for item, count in sigma_alerts.most_common():
            context_lines.append(f"- {item} ({count}회)")
    if process_flows:
        context_lines.append("\n### 주요 프로세스 생성 흐름:")
        for item, count in process_flows.most_common():
            context_lines.append(f"- {item} ({count}회)")
    if network_events or file_events or registry_events:
        context_lines.append("\n### 기타 주요 이벤트:")
        for item, count in network_events.most_common(5):
            context_lines.append(f"- {item} ({count}회)")
        for item, count in file_events.most_common(5):
            context_lines.append(f"- {item} ({count}회)")
        for item, count in registry_events.most_common(5):
            context_lines.append(f"- {item} ({count}회)")

    return "\n".join(context_lines)


def summarize_trace_with_llm(trace_input, prompt_template):
    if isinstance(trace_input, str):
        with open(trace_input, 'r', encoding='utf-8-sig') as f:
            trace_data = json.load(f)
    else:
        trace_data = trace_input

    summary_context = create_summary_context(trace_data)
    final_prompt = prompt_template.replace("[분석할 JSON 데이터가 여기에 삽입됩니다]", summary_context)

    try:
        response = llm.invoke(final_prompt)
        raw_content = response.content
        if not raw_content.strip():
            return {"error": "LLM으로부터 빈 응답을 받았습니다."}

        cleaned_content = raw_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split('\n', 1)[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit('\n', 1)[0]

        analysis_result = json.loads(cleaned_content.strip())
        return analysis_result
    except json.JSONDecodeError:
        return {"error": "LLM이 유효한 JSON을 반환하지 않았습니다.", "raw_response": raw_content}
    except Exception as e:
        return {"error": f"LLM 호출 중 오류 발생: {e}"}


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_similar_traces(driver, summary_text, top_k=5):
    with driver.session(database=DATABASE) as session:
        all_summaries = session.run("""
            MATCH (t:Trace)-[:HAS_SUMMARY]->(s:Summary)
            RETURN 
                coalesce(t.traceId, t.`traceId:ID(Trace)`) AS trace_id, 
                s.embedding AS embedding
        """)

        summary_embedding = embedding_model.encode(summary_text)
        similarities = []

        for record in all_summaries:
            trace_id = record['trace_id']
            emb = record['embedding']
            sim = cosine_similarity(summary_embedding, emb)
            similarities.append({'trace_id': trace_id, 'similarity': sim})

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]


def generate_mitigation_prompt(summary_result, structural_similarity, indirect_connections):
    """
    LLM에게 악성 행위 대응 방안을 요청하는 프롬프트 생성
    """
    summary_text = summary_result.get('summary', '')

    similar_entities = set()
    similar_techniques = set()
    for s in structural_similarity:
        similar_entities.update(s['common_entities'])
        similar_techniques.update(s['common_techniques'])

    for c in indirect_connections:
        similar_entities.add(c['e1_name'])
        similar_entities.add(c['e2_name'])

    prompt = f"""
    당신은 보안 전문가입니다. 아래 트레이스 분석 정보를 바탕으로 기업 환경에서 발견된 악성 행위에 대한 
    실제 대응 방안을 구체적으로 제안해주세요.

    [트레이스 요약]
    {summary_text}

    [연관 엔티티]
    {', '.join(similar_entities)}

    [연관 공격 기술 / TTP]
    {', '.join(similar_techniques)}

    [요청]
    1. 탐지된 악성 프로세스 및 파일 격리 방법
    2. 네트워크 차단 및 외부 통신 통제 방안
    3. 로그/시스템 모니터링 강화 방법
    4. 향후 유사 공격 예방 전략
    5. 실무자가 바로 적용 가능한 단계별 대응 권장

    응답은 JSON 또는 마크다운 형식으로 작성하고, 단계별로 번호를 붙여 상세히 설명해주세요.
    언어는 반드시 한국어로 응답하세요.
    """
    return prompt


def analyze_structural_similarity_no_db(driver, new_trace, prompt_template, top_k=5):
    # LLM 요약
    summary_result = summarize_trace_with_llm(new_trace, prompt_template)
    if 'error' in summary_result:
        return summary_result
    summary_text = summary_result.get('summary', '')

    # 의미적 유사 트레이스 검색
    top_similar_traces = find_similar_traces(driver, summary_text, top_k=top_k)
    similar_ids = [t['trace_id'] for t in top_similar_traces]

    print(f"\n🔍 의미적 유사도 상위 {top_k} 트레이스: {similar_ids}\n")

    # 구조적 유사성 분석
    with driver.session(database=DATABASE) as session:
        res = session.run("""
            MATCH (t:Trace)-[:HAS_SUMMARY]->(s:Summary)
            WHERE t.traceId IN $trace_ids
            OPTIONAL MATCH (s)-[:MENTIONS]->(e)
            OPTIONAL MATCH (s)-[:INDICATES_TECHNIQUE]->(tech)
            RETURN t.traceId AS trace_id, 
                   collect(DISTINCT coalesce(e.name, e.filePath, e.address, e.keyPath)) AS entities,
                   collect(DISTINCT tech.name) AS techniques
        """, trace_ids=similar_ids)

        trace_json = new_trace if isinstance(new_trace, dict) else json.load(open(new_trace, 'r', encoding='utf-8-sig'))
        new_entities = set([e['value'] for e in trace_json.get('key_entities', [])])
        new_techniques = set([t['name'] for t in trace_json.get('attack_techniques', [])])

        comparisons = []
        for record in res:
            db_entities = set(record['entities'])
            db_techniques = set(record['techniques'])
            common_entities = new_entities & db_entities
            common_techniques = new_techniques & db_techniques
            comparisons.append({
                'trace_id': record['trace_id'],
                'common_entities': list(common_entities),
                'common_techniques': list(common_techniques),
                'entity_match_count': len(common_entities),
                'technique_match_count': len(common_techniques)
            })

        comparisons.sort(key=lambda x: (x['entity_match_count'], x['technique_match_count']), reverse=True)

    # 간접 연결 탐색
    with driver.session(database=DATABASE) as session:
        query = """
            UNWIND $trace_ids AS trace_id
            MATCH (t:Trace {traceId: trace_id})-[:HAS_SUMMARY]->(:Summary)-[:MENTIONS]->(e)
            WITH collect(DISTINCT e) AS groupEntities
            UNWIND groupEntities AS e1
            UNWIND groupEntities AS e2
            WITH e1, e2 WHERE id(e1) < id(e2)
            MATCH path = shortestPath((e1)-[*..2]-(e2))
            RETURN e1.name AS e1_name, e2.name AS e2_name,
                   length(path) AS hops,
                   [n IN nodes(path) | labels(n)[0] + ':' + coalesce(n.name, n.filePath, n.address, '')] AS path_nodes
            LIMIT 50
        """
        indirect_connections = session.run(query, trace_ids=similar_ids)
        indirect_connections = [r.data() for r in indirect_connections]

    # 대응 제안 생성
    mitigation_prompt = generate_mitigation_prompt(summary_result, comparisons, indirect_connections)
    mitigation_response = llm.invoke(mitigation_prompt)

    return {
        'summary': summary_result,
        'semantic_top_traces': top_similar_traces,
        'structural_similarity': comparisons,
        'indirect_connections': indirect_connections,
        'mitigation_suggestions': mitigation_response.content
    }


if __name__ == "__main__":
    # 요약 프롬프트 읽기
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"오류: '{os.path.abspath(PROMPT_FILE)}' 파일을 찾을 수 없습니다.")
        exit()

    # 분석할 trace 경로 -> kafka 연동 필요
    trace_path = "C:\\Users\\KISIA\\Downloads\\data\\~trace-0c19072a66a94fe548636d7b50a06bef.json"

    results = analyze_structural_similarity_no_db(driver, trace_path, prompt_template, top_k=5)
    print(json.dumps(results, ensure_ascii=False, indent=2))
