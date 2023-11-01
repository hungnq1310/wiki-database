import os
import sys
from tqdm.auto import tqdm
import glob
import json
from typing import List, Union

from sentence_transformers import SentenceTransformer
import datasets

from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.retriever_model import (
    get_ctx_emb_sbert,
    )

import logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Default Table
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
# Custom
TB_CLIENT=os.getenv("TB_CLIENT", "client_tb")
# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16


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

def create_wiki_table(model_dim: int) -> Collection:
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
            dim=model_dim,  
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


def _insert_entites(
        collection: Collection,
        batch_ids: List[int],
        batch_names: List[str],
        batch_titles: List[str],
        batch_contents: List[str],
        batch_embs: List[List[float]],
        ):
    """
    Example:
        data_samples = [[feature_1], [feature_2], [feature_3]]
        collection.insert(data_samples)
    """
    # concatenate to primal list
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


def using_sbert_encode_multi_gpu(
        collection: Collection,
        model: SentenceTransformer,
        snippets: datasets.iterable_dataset.IterableDataset,
        target_devices: List[str],
        batch_insert: int,
        limit_samples: int
        ) -> None:
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
        if idx < current_id:
            continue
        batch_ids.append(idx)
        batch_titles.append(str(article["section_title"]))
        batch_names.append(str(article["article_title"]))
        batch_contents.append(str(article["passage_text"]))

        if len(batch_contents) == batch_insert:
            assert len(batch_titles) == len(batch_names) == len(batch_contents), \
                f"""
                len(batch_titles): {len(batch_titles)}, 
                len(batch_names): {len(batch_names)}, 
                len(batch_contents): {len(batch_contents)}
                """
            batch_embeds = get_ctx_emb_sbert(
                model=model,
                batch_contents=batch_contents,
                target_devices=target_devices
            )
            _insert_entites(
                collection=collection,
                batch_ids=batch_ids,
                batch_titles=batch_titles,
                batch_names=batch_names,
                batch_contents=batch_contents,
                batch_embs=batch_embeds,
            )
            # clear
            batch_ids.clear()
            batch_titles.clear()
            batch_names.clear()
            batch_contents.clear()
    
    # insert the remaining samples when not having enough to collect a batch
    if batch_contents:
        assert len(batch_titles) == len(batch_names) == len(batch_contents), \
            f"""
            len(batch_titles): {len(batch_titles)}, 
            len(batch_names): {len(batch_names)}, 
            len(batch_contents): {len(batch_contents)}
            """
        batch_embeds = get_ctx_emb_sbert(
            model=model,
            batch_contents=batch_contents,
            target_devices=target_devices
        )
        _insert_entites(
            collection=collection,
            batch_ids=batch_ids,
            batch_titles=batch_titles,
            batch_names=batch_names,
            batch_contents=batch_contents,
            batch_embs=batch_embeds,
        )
    # log
    logger.info(f"Insert knowledges to {TB_WIKI} successfully")
    

def insert_client_knowledges(
        saved_path: str,
        collection: Collection, 
        model: SentenceTransformer,
        target_devices: List[str],
        batch: int
        ) -> None:
    """Insert client's knowledge to table

    Args:
        context_encoder: a model that encodes data to embedding
        context_tokenizer: a tokenizer that creates input for `context_encoder`
        snippets: dataset object that contains wikipedia snippet passages
    """
    logger.info(f"Starting inserting knowledge to {TB_CLIENT}")

    # load file after preprocessing
    files = glob.glob(os.path.join(saved_path, "*.json"))

    try:
        current_id = collection.num_entities
        batch_ids = []
        batch_titles = []
        batch_domains = []
        batch_contents = []
        for idx, file in tqdm(enumerate(files)):
            load_data = json.load(open(file, "r"))
            if not isinstance(load_data, dict):
                raise ValueError("Sample of file is not a dictionary")
            if idx < current_id:
                continue
            # collect batch samples
            batch_ids.append(idx)
            batch_titles.append(load_data.get("title"))
            batch_domains.append(load_data.get("domains"))
            batch_contents.append(load_data.get("content"))
            # insert batch
            if len(batch_contents) == batch:
                assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                        f"""len(batch_titles): {len(batch_titles)}, 
                        len(batch_domains): {len(batch_domains)},  
                        len(batch_contents): {len(batch_contents)}"""
                # get embddings from model sbert
                batch_embeds = get_ctx_emb_sbert(
                    model=model,
                    batch_contents=batch_contents,
                    target_devices=target_devices,
                )
                # insert entities, batch_names <- batch_domains
                _insert_entites(
                    collection=collection,
                    batch_ids=batch_ids,
                    batch_titles=batch_titles,
                    batch_names=batch_domains,
                    batch_contents=batch_contents,
                    batch_embs=batch_embeds,
                )
                
                batch_ids.clear()
                batch_titles.clear()
                batch_domains.clear()
                batch_contents.clear()

        # continue insert whether batch_contents is not None
        if batch_contents:
            assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                f"""len(batch_titles): {len(batch_titles)}, 
                len(batch_domains): {len(batch_domains)},  
                len(batch_contents): {len(batch_contents)}"""

            # get embddings from model sbert
            batch_embeds = get_ctx_emb_sbert(
                model=model,
                batch_contents=batch_contents,
                target_devices=target_devices,
            )
            # insert entities, batch_names <- batch_domains
            _insert_entites(
                collection=collection,
                batch_ids=batch_ids,
                batch_titles=batch_titles,
                batch_names=batch_domains,
                batch_contents=batch_contents,
                batch_embs=batch_embeds,
            )
        # log
        logger.info(f"Insert knowledges to {TB_CLIENT} successfully")

    except Exception as e:
        logger.error(f"Failed inserting knowledge into {TB_CLIENT}: {e}")


def build_indexs(collection: Collection, 
                 filed_name: str = "embedding"):
    """Building index for column search  
    
    Please check milvus's documents to customize parameters: https://milvus.io/docs/build_index.md
    Args
        collection: A table using to build
        filed_name: A column using to build
    """
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    logger.info("\nCreated index:\n{}".format(collection.index().params))

def search(
        collection: Collection, 
        data_embed: Union[List[float], List[List[float]]],
        column_search: str = "embedding",  
        limit_docs: int = 3,
        ):
    """Insert client's knowledge to table

    To customize search_params, please check documents of milvus: https://milvus.io/docs/search.md
    Args:
        collection: A table using to search
        data_embed: one or list of embedding outputs which encoding from model(**)
        column_search: Search on column which already building indexes, default to search on `embedding` column
        limit_docs: Number of docs return per sample
    """
    # Feel free to custom metric type and n_probe
    search_params = {
        "metric_type": _METRIC_TYPE, 
        "ignore_growing": False, 
        "params": {"nprobe": _NPROBE}
    }
    # Search
    results = collection.search(
        data=data_embed,
        anns_field=column_search,
        param=search_params,
        limit=limit_docs,
        expr=None)
    
    # print test
    for i, result in enumerate(results):
        logger.info("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            logger.info("Top {}: {}".format(j, res))

    return results
