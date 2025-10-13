import json
import csv
import os
from tqdm import tqdm

def process_trace_directory(input_dir, output_dir='output_csvs_aggregated'):

    def get_tag_value(tags, key, default=None):
        for tag in tags:
            if tag.get('key') == key:
                return tag.get('value')
        return default

    traces, users, files, ips, registry_keys = set(), set(), set(), set(), set()
    processes_data = {} 

    rel_participated_process, rel_participated_user, rel_participated_file = set(), set(), set()
    rel_participated_ip, rel_participated_registry = set(), set()
    rel_spawned, rel_created_file, rel_connects_to, rel_modified_key, rel_executed_by = set(), set(), set(), set(), set()

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"총 {len(file_list)}개의 트레이스 파일을 처리합니다...")

    for filename in tqdm(file_list, desc="Processing Traces"):
        file_path = os.path.join(input_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                trace_data = json.load(f)
            trace_id = trace_data.get('traceID')
            if not trace_id:
                continue

            traces.add((trace_id,))

            for span in trace_data.get('spans', []):
                tags = span.get('tags', [])
                proc_guid = get_tag_value(tags, 'ProcessGuid')
                if not proc_guid:
                    continue
                
                if proc_guid not in processes_data:
                    processes_data[proc_guid] = {'guid': proc_guid}
                
                image_path = get_tag_value(tags, 'Image')
                if image_path:
                    processes_data[proc_guid]['imagePath'] = image_path
                    processes_data[proc_guid]['name'] = os.path.basename(image_path)
                
                rel_participated_process.add((proc_guid, trace_id))
                
                proc_user = get_tag_value(tags, 'User')
                if proc_user:
                    processes_data[proc_guid]['user'] = proc_user 
                    users.add((proc_user,))
                    rel_executed_by.add((proc_guid, proc_user))
                    rel_participated_user.add((proc_user, trace_id))

                event_name_lower = get_tag_value(tags, 'EventName', '').lower()

                if 'processcreate' in event_name_lower:
                    processes_data[proc_guid].update({
                        'pid': get_tag_value(tags, 'ProcessId'),
                        'commandLine': get_tag_value(tags, 'CommandLine', ''),
                    })
                    
                    parent_guid = get_tag_value(tags, 'ParentProcessGuid')
                    if parent_guid:
                        if parent_guid not in processes_data:
                            processes_data[parent_guid] = {'guid': parent_guid}
                        parent_image = get_tag_value(tags, 'ParentImage')
                        if parent_image:
                           processes_data[parent_guid]['imagePath'] = parent_image
                           processes_data[parent_guid]['name'] = os.path.basename(parent_image)
                        
                        parent_user = get_tag_value(tags, 'ParentUser')
                        if parent_user:
                            processes_data[parent_guid]['user'] = parent_user 
                            users.add((parent_user,))
                            rel_executed_by.add((parent_guid, parent_user))
                            rel_participated_user.add((parent_user, trace_id))

                        rel_participated_process.add((parent_guid, trace_id))
                        rel_spawned.add((parent_guid, proc_guid))

                elif 'filecreate' in event_name_lower:
                    target_file = get_tag_value(tags, 'TargetFilename')
                    if target_file:
                        files.add((target_file,))
                        rel_participated_file.add((target_file, trace_id))
                        rel_created_file.add((proc_guid, target_file))

                elif 'networkconnect' in event_name_lower:
                    dest_ip = get_tag_value(tags, 'DestinationIp')
                    if dest_ip and dest_ip != '-':
                        ips.add((dest_ip,))
                        rel_participated_ip.add((dest_ip, trace_id))
                        rel_connects_to.add((proc_guid, dest_ip))

                elif 'registryvalueset' in event_name_lower:
                    target_object = get_tag_value(tags, 'TargetObject')
                    if target_object:
                        registry_keys.add((target_object,))
                        rel_participated_registry.add((target_object, trace_id))
                        rel_modified_key.add((proc_guid, target_object))
        except Exception as e:
            print(f"\n파일 처리 중 오류 발생 ({filename}): {e}")

    os.makedirs(output_dir, exist_ok=True)

    def write_csv(filename, header, data):
        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            if data:
                writer.writerows(sorted(list(data)))

    print("\nCSV 파일 저장 시작...")
    
    processes_list = [
        (
            p['guid'], p.get('pid'), p.get('commandLine', ''), 
            p.get('user', ''), p.get('name', 'Unknown'), p.get('imagePath', '')
        ) for p in processes_data.values()
    ]

    write_csv('nodes_traces.csv', ['traceId:ID(Trace)'], traces)
    write_csv('nodes_users.csv', ['name:ID(User)'], users)
    write_csv('nodes_processes.csv', ['guid:ID(Process)', 'pid:int', 'commandLine', 'user', 'name', 'imagePath'], processes_list)
    write_csv('nodes_files.csv', ['filePath:ID(File)'], files)
    write_csv('nodes_ips.csv', ['address:ID(IP)'], ips)
    write_csv('nodes_registry_keys.csv', ['keyPath:ID(Registry)'], registry_keys)
    
    write_csv('rels_process_participated.csv', [':START_ID(Process)', ':END_ID(Trace)'], rel_participated_process)
    write_csv('rels_user_participated.csv', [':START_ID(User)', ':END_ID(Trace)'], rel_participated_user)
    write_csv('rels_file_participated.csv', [':START_ID(File)', ':END_ID(Trace)'], rel_participated_file)
    write_csv('rels_ip_participated.csv', [':START_ID(IP)', ':END_ID(Trace)'], rel_participated_ip)
    write_csv('rels_registry_participated.csv', [':START_ID(Registry)', ':END_ID(Trace)'], rel_participated_registry)
    write_csv('rels_executed_by.csv', [':START_ID(Process)', ':END_ID(User)'], rel_executed_by)
    write_csv('rels_spawned.csv', [':START_ID(Process)', ':END_ID(Process)'], rel_spawned)
    write_csv('rels_created_file.csv', [':START_ID(Process)', ':END_ID(File)'], rel_created_file)
    write_csv('rels_connects_to.csv', [':START_ID(Process)', ':END_ID(IP)'], rel_connects_to)
    write_csv('rels_modified_key.csv', [':START_ID(Process)', ':END_ID(Registry)'], rel_modified_key)
    
    print(f"CSV 파일 생성이 완료되었습니다. '{output_dir}' 폴더를 확인하세요.")


if __name__ == "__main__":
    directory_to_process = 'C:\\Users\\KISIA\\Downloads\\data' 
    
    if not os.path.isdir(directory_to_process):
        print(f"❌ 디렉터리를 찾을 수 없습니다: '{directory_to_process}'")
    else:
        process_trace_directory(directory_to_process)
