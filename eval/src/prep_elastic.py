# import argparse
# import glob
# import time
# import csv
# from tqdm import tqdm
# from beir.retrieval.search.lexical.elastic_search import ElasticSearch
# from elasticsearch.helpers import streaming_bulk

# def build_elasticsearch(
#     beir_corpus_file_pattern: str,
#     index_name: str,
# ):
#     beir_corpus_files = glob.glob(beir_corpus_file_pattern)
#     print(f'#files {len(beir_corpus_files)}')
#     config = {
#         'hostname': 'http://localhost:9200',
#         'index_name': index_name,
#         'keys': {'title': 'title', 'body': 'body'},
#         'timeout': 100,
#         'retry_on_timeout': True,
#         'maxsize': 24,
#         'number_of_shards': 'default',
#         'language': 'english',
#     }
#     es = ElasticSearch(config)

#     # create index
#     print(f'create index {index_name}')
#     es.delete_index()
#     time.sleep(5)
#     es.create_index()

#     # generator
#     def generate_actions():
#         for beir_corpus_file in beir_corpus_files:
#             with open(beir_corpus_file, 'r') as fin:
#                 reader = csv.reader(fin, delimiter='\t')
#                 header = next(reader)  # skip header
#                 for row in reader:
#                     _id, text, title = str(row[0]), row[1], row[2]
#                     es_doc = {
#                         '_id': _id,
#                         '_op_type': 'index',
#                         'refresh': 'wait_for',
#                         config['keys']['title']: title,
#                         config['keys']['body']: text,
#                     }
#                     yield es_doc

 
#     # index
#     progress = tqdm(unit='docs')
#     es.bulk_add_to_index(
#         generate_actions=generate_actions(),
#         progress=progress)




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default=None, help='input file')
#     parser.add_argument("--index_name", type=str, default=None, help="index name")
#     args = parser.parse_args()
#     build_elasticsearch(args.data_path, index_name=args.index_name)

import argparse
import glob
import time
import csv
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

def build_elasticsearch_optimized(
    beir_corpus_file_pattern: str,
    index_name: str,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    
    # 直接连接 Elasticsearch
    es = Elasticsearch(
        ['http://localhost:9200'],
        timeout=100,
        max_retries=10,
        retry_on_timeout=True
    )

    # 删除旧索引
    if es.indices.exists(index=index_name):
        print(f'Deleting existing index {index_name}')
        es.indices.delete(index=index_name)
        time.sleep(2)

    # 创建索引，优化设置用于批量导入
    print(f'Creating index {index_name} with optimized settings')
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,  # 导入时不需要副本
            "refresh_interval": "-1",  # 禁用自动刷新，极大提升性能
            "index.translog.durability": "async",  # 异步提交
            "index.translog.flush_threshold_size": "1gb"
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "english"
                },
                "body": {
                    "type": "text",
                    "analyzer": "english"
                }
            }
        }
    }
    es.indices.create(index=index_name, body=index_settings)
    print("Index created")

    # 生成器
    def generate_actions():
        for beir_corpus_file in beir_corpus_files:
            print(f"Processing file: {beir_corpus_file}")
            with open(beir_corpus_file, 'r', encoding='utf-8') as fin:
                reader = csv.reader(fin, delimiter='\t')
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                
                for row in reader:
                    if len(row) < 3:
                        continue
                    
                    _id, text, title = str(row[0]), row[1], row[2]
                    yield {
                        '_index': index_name,
                        '_id': _id,
                        'title': title,
                        'body': text,
                    }

    # 使用 parallel_bulk 进行并行索引（更快）
    print("Starting parallel bulk indexing...")
    progress = tqdm(unit='docs', desc="Indexing")
    success_count = 0
    error_count = 0
    errors = []
    
    try:
        for ok, response in parallel_bulk(
            client=es,
            actions=generate_actions(),
            thread_count=4,  # 并行线程数
            chunk_size=1000,  # 每批1000个文档
            max_chunk_bytes=15728640,  # 15MB per batch
            raise_on_error=False,
            raise_on_exception=False,
            request_timeout=120,
        ):
            progress.update(1)
            if ok:
                success_count += 1
            else:
                error_count += 1
                if len(errors) < 10:
                    errors.append(response)
        
        progress.close()
        
        # 恢复正常设置并刷新
        print("\nRestoring index settings...")
        es.indices.put_settings(
            index=index_name,
            body={
                "index": {
                    "refresh_interval": "1s",
                    "number_of_replicas": 1,
                    "translog.durability": "request"
                }
            }
        )
        
        print("Forcing refresh...")
        es.indices.refresh(index=index_name)
        
        # 打印统计
        count = es.count(index=index_name)['count']
        print(f"\n{'='*50}")
        print(f"Indexing Summary:")
        print(f"  Documents in index: {count}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {error_count}")
        print(f"{'='*50}")
        
        if errors:
            print("\nFirst few errors:")
            for err in errors[:3]:
                print(err)
                
    except Exception as e:
        progress.close()
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='input file')
    parser.add_argument("--index_name", type=str, required=True, help="index name")
    args = parser.parse_args()
    build_elasticsearch_optimized(args.data_path, index_name=args.index_name)