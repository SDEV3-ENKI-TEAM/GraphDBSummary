import json
import os
import uuid
from neo4j import GraphDatabase
from tqdm import tqdm

# --- 설정 변수 ---
MERGED_SUMMARY_FILE = "merged_summaries.json"

# --- Neo4j 연결 정보 ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "20011226")
DATABASE = "neo4j"

def model_llm_summary_in_db(driver, summary_data):
    """
    JSON에서 읽어온 하나의 요약+임베딩 데이터를 기존 DB에 MERGE하여 추가합니다.
    """
    trace_id = summary_data.get("traceid")
    summary_text = summary_data.get("summary")
    key_entities = summary_data.get("key_entities", [])
    attack_techniques = summary_data.get("attack_techniques", [])
    # [수정] 파일에서 미리 계산된 임베딩을 가져옵니다. 없으면 임시 벡터 사용.
    summary_embedding = summary_data.get("embedding", [0.0] * 128)

    if not (trace_id and summary_text):
        return

    summary_id = str(uuid.uuid4())

    with driver.session(database=DATABASE) as session:
        # 1. Summary, Technique 노드를 생성하고 기존 Trace에 연결
        session.run("""
            // 잘못된 속성 이름으로 저장된 경우를 대비해 두 가지 케이스 모두 확인
            MATCH (trace:Trace) WHERE trace.traceId = $trace_id OR trace.`traceId:ID(Trace)` = $trace_id
            
            // 요약 노드 생성
            CREATE (summary:Summary {id: $summary_id, text: $summary_text, embedding: $embedding})
            
            // Trace와 Summary를 연결
            MERGE (trace)-[:HAS_SUMMARY]->(summary)

            // Technique 노드들을 생성하고 Summary에 연결
            WITH summary
            UNWIND $techniques AS tech
            FOREACH (id IN CASE WHEN tech.id IS NOT NULL THEN [tech.id] ELSE [] END |
                MERGE (t:Technique {id: id}) ON CREATE SET t.name = tech.name
                MERGE (summary)-[:INDICATES_TECHNIQUE]->(t)
            )
        """, {
            "trace_id": trace_id, "summary_id": summary_id,
            "summary_text": summary_text, "embedding": summary_embedding,
            "techniques": attack_techniques
        })

        # 2. Key Entity들을 찾아 Summary와 MENTIONS 관계로 연결
        for entity in key_entities:
            entity_type = entity.get("type")
            entity_value = entity.get("value")
            if not (entity_type and entity_value):
                continue
            
            if entity_type == "Process":
                session.run("""
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (p:Process {name: $process_name})
                    MERGE (s)-[:MENTIONS]->(p)
                """, {"summary_id": summary_id, "process_name": entity_value})
            
            elif entity_type == "File":
                session.run("""
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (f:File {filePath: $file_path})
                    MERGE (s)-[:MENTIONS]->(f)
                """, {"summary_id": summary_id, "file_path": entity_value})
            
            elif entity_type == "IP Address":
                session.run("""
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (i:IpAddress {address: $ip_address})
                    MERGE (s)-[:MENTIONS]->(i)
                """, {"summary_id": summary_id, "ip_address": entity_value})

            elif entity_type == "User":
                session.run("""
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (u:User {name: $user_name})
                    MERGE (s)-[:MENTIONS]->(u)
                """, {"summary_id": summary_id, "user_name": entity_value})

            elif entity_type == "Registry Key":
                session.run("""
                    MATCH (s:Summary {id: $summary_id})
                    MATCH (r:RegistryKey {keyPath: $key_path})
                    MERGE (s)-[:MENTIONS]->(r)
                """, {"summary_id": summary_id, "key_path": entity_value})

def main():
    """메인 실행 함수"""
    # 1. 미리 생성된 요약+임베딩 JSON 파일을 로드합니다.
    try:
        with open(MERGED_SUMMARY_FILE, 'r', encoding='utf-8-sig') as f:
            all_summaries = json.load(f)
        print(f"✅ '{MERGED_SUMMARY_FILE}' 파일 로드 성공. 총 {len(all_summaries)}개의 요약 정보를 찾았습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{os.path.abspath(MERGED_SUMMARY_FILE)}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"❌ 오류: '{MERGED_SUMMARY_FILE}' 파일이 올바른 JSON 형식이 아닙니다 (BOM 또는 중간 중단 여부 확인).")
        return

    # 2. 데이터베이스에 연결하고, 각 요약 정보를 DB에 추가합니다.
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            print("Neo4j Desktop 연결 성공!")
            
            # [수정] JSON 파일이 리스트 형식일 경우를 처리
            if isinstance(all_summaries, list):
                print(f"총 {len(all_summaries)}개의 요약 정보를 DB에 추가합니다...")
                for summary_data in tqdm(all_summaries, desc="Enriching DB with Summaries"):
                    if summary_data and "error" not in summary_data:
                        model_llm_summary_in_db(driver, summary_data)
            # [수정] JSON 파일이 딕셔너리 형식일 경우를 처리
            elif isinstance(all_summaries, dict):
                print(f"총 {len(all_summaries)}개의 요약 정보를 DB에 추가합니다...")
                for trace_id, summary_data in tqdm(all_summaries.items(), desc="Enriching DB with Summaries"):
                    if summary_data and "error" not in summary_data:
                         # 딕셔너리 형식일 경우 trace_id를 직접 전달
                        summary_data['traceid'] = trace_id
                        model_llm_summary_in_db(driver, summary_data)

        print(f"\n✅ 작업 완료! 모든 요약 정보가 데이터베이스에 추가/업데이트되었습니다.")

    except Exception as e:
        print(f"\n❌ 데이터베이스 작업 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

