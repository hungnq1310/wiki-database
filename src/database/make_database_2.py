import os
import sys
from tqdm.auto import tqdm
import pandas as pd
import math
import glob
import json
from typing import List, Union

from transformers import (
        DPRContextEncoderTokenizer,
        DPRContextEncoder,
        ViltLayer
        )
import datasets
import torch

from pymilvus import connections, utility
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.retriever_model import (
         get_ctx_embd
         )

import logging
import dotenv
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Connection
DBNAME=os.getenv("DBNAME", "wiki_33m_milvus")
HOST=os.getenv("HOST", "localhost")
PORT=os.getenv("PORT", "19530")
USER=os.getenv("USER", "root")
PWD=os.getenv("PWD", "Milvus")
# Default Table
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
# Custom
TB_CLIENT=os.getenv("TB_CLIENT", "client_tb")
# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3


def create_milvus_connection(host, port) -> None:
    """Create a database to contain a data table
    """
    try:
        # Create a Milvus connection
        logger.info(f"\nCreate connection...")
        connections.connect(host=host, port=port)
        logger.info(f"\nList connections:")
        logger.info(connections.list_connections())
    except Exception as e:
        logger.error(f"""Error while connecting to Milvus Server: {e}, 
                     please check that Milvus is runing in Docker Desktop""")

def create_wiki_table() -> Collection:
    """Create a table contains wiki snippets passages
    """
    try:
        # define each type of each filed (columns)
        id = FieldSchema(
                name="document_id",
                dtype=DataType.INT64,
                is_primary=True,
        )
        title = FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=10000,
                # The default value will be used if this field is left empty during data inserts or upserts.
                # The data type of `default_value` must be the same as that specified in `dtype`.
                default_value="Unknown"
        )
        name = FieldSchema(
                name="name",
                dtype=DataType.VARCHAR,
                max_length=10000,
                # The default value will be used if this field is left empty during data inserts or upserts.
                # The data type of `default_value` must be the same as that specified in `dtype`.
                default_value="Unknown"
        )
        passage_text = FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=50000,
                # The default value will be used if this field is left empty during data inserts or upserts.
                # The data type of `default_value` must be the same as that specified in `dtype`.
                default_value="Unknown"
        )
        embedding_text = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=384,  
        )

        # Define schema for collection
        schema = CollectionSchema(
                fields=[id, name, title, passage_text, embedding_text],
                description="Wiki Table",
                enable_dynamic_field=True
        )
        
        collection = Collection(
                name=TB_WIKI,
                schema=schema,
                using='default'
        )
        print(f"{TB_WIKI} table is created successfully.")
        return collection

    except Exception as e:
        logger.error(f"Error while creating Wiki Table: {e}")

def get_entity_num(collection: Collection):
    logger.info("\nThe number of entity:", collection.num_entities)
    return collection.num_entities


def _insert_entites(collection: Collection,
                    batch_ids: List[int],
                    batch_names: List[str],
                    batch_titles: List[str],
                    batch_contents: List[str],
                    batch_embs: List[List[float]],
                   ):
        """
        Example:
                task_id = utility.do_bulk_insert(
                        collection_name="book",
                        files=["test.json"]
                )
        """
        # list all files existed in save path
        # do bulk insert
        entities = [
                batch_ids,
                batch_names,  
                batch_titles,
                batch_contents,
                batch_embs
        ]

        insert_result = collection.insert(entities)
        collection.flush()
        print(f"Number of entities in Milvus: {collection.num_entities}")


def _insert_bulk(collection: Collection,
            save_path= str,
           ):
        """
        Example:
                task_id = utility.do_bulk_insert(
                        collection_name="book",
                        files=["test.json"]
                )
        """
        # list all files existed in save path
        # do bulk insert
        task_id  = utility.do_bulk_insert(
                collection_name=collection.name,
                files=["samples.json"]
        )
        print("Status: ", task_id)

def _store_to_json(save_path: str,
                   batch_ids: List[int],
                   batch_titles: List[str],
                   batch_names: List[str],
                   batch_contents: List[str],
                   batch_embeds: List[List[float]],
                   device: str):
    format_insert_data = {"rows": []}
    for e_id, e_name, e_title, e_contents, e_embed in zip(batch_ids, batch_names, batch_titles, batch_contents, batch_embeds):
        # optimize colllection.entities
        format_insert_data["rows"].append({
            "document_id": e_id,  
            "name": e_name,
            "title": e_title,
            "text": e_contents,
            "embedding": e_embed
        })
    with open(os.path.join(save_path, f'samples.json'), "a+") as f:
        json.dump(format_insert_data, f)
    logger.info("Store success")

def _store_client_to_json(save_path: str,
                   batch_ids: List[int],
                   batch_titles: List[str],
                   batch_doamins: List[str],
                   batch_contents: List[str],
                   batch_embeds: List[List[float]],
                   device: str):
    format_insert_data = {"rows": []}
    for e_id, e_name, e_title, e_contents, e_embed in zip(batch_ids, batch_doamins, batch_titles, batch_contents, batch_embeds):
        # optimize colllection.entities
        format_insert_data["rows"].append({
            "document_id": e_id,  
            "domain": e_name,
            "title": e_title,
            "text": e_contents,
            "embedding": e_embed
        })
    with open(os.path.join(save_path, f'batch_{batch_ids[0]}.json'), "w+") as f:
        json.dump(format_insert_data, f)
    logger.info("Store success")
 

def insert_knowledges(
        saved_path: str,
        collection: Collection,
        context_encoder: DPRContextEncoder,
        context_tokenizer: DPRContextEncoderTokenizer,
        snippets: datasets.iterable_dataset.IterableDataset,
        device: torch.device,
        n_gpus: int, 
        gpu_index: int,
        batch: int,
        limit_samples: int
        )->None:
        """Insert wiki snippets or knowledge to wiki table

        Args:
                context_encoder: a model that encodes data to embedding
                context_tokenizer: a tokenizer that creates input for `context_encoder`
                snippets: dataset object that contains wikipedia snippet passages
        """
    
        logger.info(f"Starting inserting knowledge to {TB_WIKI}")

        current_id = collection.num_entities

        batch_ids = []
        batch_titles = []
        batch_names = []
        batch_contents = []
        for idx, article in tqdm(enumerate(iter(snippets))):
                if idx == limit_samples:
                     break
        #     if idx%n_gpus != gpu_index: 
        #         continue

                if idx < current_id:
                        continue
                batch_ids.append(idx)
                batch_titles.append(str(article["section_title"]))
                batch_names.append(str(article["article_title"]))
                batch_contents.append(str(article["passage_text"]))

                if len(batch_contents) == batch:
                        assert len(batch_titles) == len(batch_names) == len(batch_contents), \
                                f"len(batch_titles): {len(batch_titles)} "\
                                f"len(batch_names): {len(batch_names)}"\
                                f"len(batch_contents): {len(batch_contents)}"
                        batch_embeds = get_ctx_embd(
                                model_encoder=context_encoder,
                                tokenizer=context_tokenizer,
                                text=batch_contents,
                                device="cuda:0"
                        )
                        _insert_entites(
                                collection=collection,
                                batch_ids=batch_ids,
                                batch_titles=batch_titles,
                                batch_names=batch_names,
                                batch_contents=batch_contents,
                                batch_embs=batch_embeds,
                        )
                        
                        # _store_to_json(
                        #         save_path=saved_path,
                        #         batch_ids=batch_ids,
                        #         batch_titles=batch_titles,
                        #         batch_names=batch_names,
                        #         batch_contents=batch_contents,
                        #         batch_embeds=batch_embeds,
                        #         device=device
                        # )
                        
                        batch_ids.clear()
                        batch_titles.clear()
                        batch_names.clear()
                        batch_contents.clear()
                        batch_embeds.clear()

        if batch_contents:
                assert len(batch_titles) == len(batch_names) == len(batch_contents), \
                        f"len(batch_titles): {len(batch_titles)} "\
                        f"len(batch_names): {len(batch_names)}"\
                        f"len(batch_contents): {len(batch_contents)}"

                batch_embeds = get_ctx_embd(
                        model_encoder=context_encoder,
                        tokenizer=context_tokenizer,
                        text=batch_contents,
                        device="cuda:0"
                )
                _insert_entites(
                        collection=collection,
                        batch_ids=batch_ids,
                        batch_titles=batch_titles,
                        batch_names=batch_names,
                        batch_contents=batch_contents,
                        batch_embs=batch_embeds,
                        )
                # _store_to_json(
                #         save_path=saved_path,
                #         batch_ids=batch_ids,
                #         batch_titles=batch_titles,
                #         batch_names=batch_names,
                #         batch_contents=batch_contents,
                #         batch_embeds=batch_embeds,
                #         device=device
                # )

        logger.info(f"Insert knowledges to {TB_WIKI} successfully")
#     except Exception as e:
#         logger.error(f"Failed inserting knowledge into {TB_WIKI}: {e}")


def insert_client_knowledges(
        saved_path: str,
        collection: Collection, 
        context_encoder: DPRContextEncoder,
        context_tokenizer: DPRContextEncoderTokenizer,
        device: torch.device,
        batch: int
        )->None:
    """Insert client's knowledge to table

    Args:
        context_encoder: a model that encodes data to embedding
        context_tokenizer: a tokenizer that creates input for `context_encoder`
        snippets: dataset object that contains wikipedia snippet passages
    """
    logger.info(f"Starting inserting knowledge to {TB_CLIENT}")

    # load file after preprocessing
    files = glob.glob(os.path.join(saved_path, "*.json"))
    one_sample = json.load(open(files[0], "r"))
    if "embedding" in one_sample['rows'][0]:
        task_id = utility.do_bulk_insert(
                        collection_name=collection.name,
                        files=files
                )
        return 

    try:
        
        current_id = collection.num_entities
        batch_ids = []
        batch_titles = []
        batch_domains = []
        batch_contents = []
        for idx, file in tqdm(enumerate(files)):
            load_data = json.load(open(file, "r"))
            if idx < current_id:
                continue
            batch_ids.append(idx)
            batch_titles.append(load_data["row"][0]["title"])
            batch_domains.append(load_data["row"][0]["domain"])
            batch_contents.append(load_data["row"][0]["content"])
            if len(batch_contents) == batch:
                assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                        f"len(batch_titles): {len(batch_titles)} "\
                        f"len(batch_domains): {len(batch_domains)}"\
                        f"len(batch_contents): {len(batch_contents)}"
                batch_embeds = get_ctx_embd(
                    model_encoder=context_encoder,
                    tokenizer=context_tokenizer,
                    text=batch_contents,
                    device="cuda:0"
                )
                batch_embeds = [(list(batch_embeds[i, :].cpu().detach().numpy().reshape(-1)))
                    for i in range(batch_embeds.size(0))
                    ]
                _store_client_to_json(
                        save_path=saved_path,
                        batch_ids=batch_ids,
                        batch_titles=batch_titles,
                        batch_doamins=batch_domains,
                        batch_contents=batch_contents,
                        batch_embs=batch_embeds,
                        device=device
                )
                
                batch_ids = []
                batch_titles.clear()
                batch_domains.clear()
                batch_contents.clear()

        if batch_contents:
            assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                    f"len(batch_titles): {len(batch_titles)} "\
                    f"len(batch_names): {len(batch_domains)}"\
                    f"len(batch_contents): {len(batch_contents)}"

            batch_embeds = get_ctx_embd(
                    model_encoder=context_encoder,
                    tokenizer=context_tokenizer,
                    text=batch_contents,
                    device="cuda:0"
                )
            batch_embeds = [(list(batch_embeds[i, :].cpu().detach().numpy().reshape(-1)))
                    for i in range(batch_embeds.size(0))
                    ]
            _store_client_to_json(
                        save_path=saved_path,
                        batch_ids=batch_ids,
                        batch_titles=batch_titles,
                        batch_doamins=batch_domains,
                        batch_contents=batch_contents,
                        batch_embs=batch_embeds,
                        device=device
                )
        # insert
        _insert(collection=collection,
                save_path=saved_path)

        logger.info(f"Insert knowledges to {TB_CLIENT} successfully")

    except Exception as e:
        logger.error(f"Failed inserting knowledge into {TB_CLIENT}: {e}")

# def create_index(
#         tb_name: str,
#         num_data: int,
#     ) -> None:
#     """Create index for embedding column

#     Args:
#         num_data: number of data or number of rows in the table
#     """
#     try:
#         print("Creating index")
#         connection = psycopg2.connect(dbname=PGDBNAME,
#                                       host=PGHOST,
#                                       port=PGPORT,
#                                       user=PGUSER,
#                                       password=PGPWD)
#         connection.autocommit = True

#         cursor = connection.cursor()
#         nlist =  round(2*math.sqrt(num_data))

#         create_index_cluster_cmd = f'''
#                 SET maintenance_work_mem TO '14 GB';
#                 CREATE INDEX ON {tb_name} USING ivfflat (embedd vector_ip_ops) WITH (lists = {nlist});
#                 '''
#         create_index_default_cmd = f'''
#                 CREATE INDEX ON {tb_name} USING ivfflat (embedd vector_ip_ops);
#                 '''
#         try:
#             logger.info(f"Creating index with {nlist} cluster")
#             cursor.execute(create_index_cluster_cmd)
#         except:
#             logger.error(f"Created index clustering on {tb_name} was failed, try default settings")
#             cursor.execute(create_index_default_cmd)
#         logger.info("Create index successfully")
#         if connection:
#             cursor.close()
#             connection.close()
#             logger.info("PostgreSQL connection is closed")

#     except (Exception, psycopg2.Error) as err:
#         logger.error("Error while connecting to PostgreSQL", err)

def create_index(collection: Collection, 
                 filed_name: str = "embedding"):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    logger.info("\nCreated index:\n{}".format(collection.index().params))

def search(collection: Collection, 
           vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"}
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        logger.info("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            logger.info("Top {}: {}".format(j, res))
