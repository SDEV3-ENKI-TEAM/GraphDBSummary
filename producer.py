from kafka import KafkaProducer
import json
from graphdb import analyze_structural_similarity_no_db
import os
from neo4j_driver import driver

driver = driver
PROMPT_FILE = "C:\\Users\\KISIA\\Desktop\\Enki\\Neo4j\\summary_prompt.md"
TRACE_PATH = "C:\\Users\\KISIA\\Downloads\\data\\T1018.json"
KAFKA_TOPIC = "trace_analysis"
KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]  # Kafka 브로커 주소

if __name__ == "__main__":
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"오류: '{os.path.abspath(PROMPT_FILE)}' 파일을 찾을 수 없습니다.")
        exit()

    # 분석
    results = analyze_structural_similarity_no_db(driver, TRACE_PATH, prompt_template, top_k=5)

    # Kafka Producer 초기화
    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"], 
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # 전송할 데이터 구성
    message = {
        "summary": results.get("summary", {}),
        "structural_similarity": results.get("structural_similarity", []),
        "indirect_connections": results.get("indirect_connections", []),
        "mitigation_suggestions": results.get("mitigation_suggestions", "")
    }

    # Kafka 전송
    producer.send(KAFKA_TOPIC, message)
    producer.flush()  # 전송 완료까지 기다림
    print(f" 분석 결과를 Kafka 토픽 '{KAFKA_TOPIC}'로 전송했습니다.")
