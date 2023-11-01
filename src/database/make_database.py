import os
import sys
from tqdm.auto import tqdm
import pandas as pd
import psycopg2
from psycopg2 import Error
import math
from transformers import (
        DPRContextEncoderTokenizer,
        DPRContextEncoder,
        ViltLayer
        )
import datasets
from typing import (
        List
        )

import torch

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

PGDBNAME=os.getenv("PGDBNAME", "wiki_pgvector")
PGHOST=os.getenv("PGHOST", "localhost")
PGPORT=os.getenv("PGPORT", "5432")
PGUSER=os.getenv("PGUSER", "wiki_ad")
PGPWD=os.getenv("PGPWD", "55235")
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
TB_CLIENT=os.getenv("TB_CLIENT", "client_tb")
BATCH=int(os.getenv("BATCH", 16))

def create_postgres_db() -> None:
    """Create a database to contain a data table
    """
    try:
        connection = psycopg2.connect(
                host=PGHOST,
                port=PGPORT,
                dbname="postgres",
                user=PGUSER,
                password=PGPWD
                )
        connection.autocommit = True
        cursor = connection.cursor()
        sql = '''CREATE DATABASE {};'''.format(PGDBNAME)
        cursor.execute(sql)

        logger.info(f"Database {PGDBNAME} is created successfully.")

        if connection:
            cursor.close()
            connection.close()

    except (Exception, Error) as e:
        logger.error(f"Error while connecting to PostgreSQL: {e}")

def create_wiki_table() -> None:
    """Create a table contains wiki snippets passages
    """
    try:
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)
        connection.autocommit = True

        cursor = connection.cursor()
        sql = f'''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE {TB_WIKI} (
                id SERIAL PRIMARY KEY,
                title TEXT,
                name TEXT,
                content TEXT,
                embedd vector(128));
                '''

        cursor.execute(sql)
        logger.info(f"{TB_WIKI} is created successfully.")

        if connection:
            cursor.close()
            connection.close()
            logger.info("PostgreSQL connection is closed.")

    except (Exception, Error) as err:
        logger.error(f"Error while connecting to PostgreSQL: {err}")

def create_client_table() -> None:
    """Create a table contains client's knowledges
    """
    try:
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)
        connection.autocommit = True

        cursor = connection.cursor()
        sql = f'''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE {TB_CLIENT} (
                id SERIAL PRIMARY KEY,
                title TEXT,
                domain TEXT,
                content TEXT,
                embedd vector(128));
                '''

        cursor.execute(sql)
        logger.info(f"{TB_CLIENT} is created successfully.")

        if connection:
            cursor.close()
            connection.close()
            logger.info("PostgreSQL connection is closed.")

    except (Exception, Error) as err:
        logger.error(f"Error while connecting to PostgreSQL: {err}")

def count_row(tb_name: str) -> int:
    """Count the number of row in a table

    It helps us to keep inserting knowledge to the table

    Args:
        tb_name: name of table
    """
    try:
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)
        connection.autocommit = True

        cursor = connection.cursor()
        sql = f'''
                SELECT count(*) FROM {tb_name}
                '''

        cursor.execute(sql)
        result = cursor.fetchone()[0]

        if connection:
            cursor.close()
            connection.close()
            logger.info("PostgreSQL connection is closed.")
        return result

    except (Exception, psycopg2.Error) as err:
        logger.error(f"Error while connecting to PostgreSQL: {err}")
        return -1

def insert_knowledges(
        context_encoder: DPRContextEncoder,
        context_tokenizer: DPRContextEncoderTokenizer,
        snippets: datasets.iterable_dataset.IterableDataset,
        device: torch.device,
        n_gpus: int, 
        gpu_index: int
        )->None:
    """Insert wiki snippets or knowledge to wiki table

    Args:
        context_encoder: a model that encodes data to embedding
        context_tokenizer: a tokenizer that creates input for `context_encoder`
        snippets: dataset object that contains wikipedia snippet passages
    """
    logger.info(f"Starting inserting knowledge to {TB_WIKI}")

    try:
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)

        cursor = connection.cursor()
        def _insert(
                batch_titles: List[str],
                batch_names: List[str],
                batch_contents: List[str],
                device: torch.device
                ) -> None:

            passage_embd = get_ctx_embd(
                    model_encoder=context_encoder,
                    tokenizer=context_tokenizer,
                    text=batch_contents,
                    device=device
                    )
            embd = [str(list(passage_embd[i, :].cpu().detach().numpy().reshape(-1)))
                    for i in range(passage_embd.size(0))
                    ]
            values =  list(zip(batch_titles, batch_names, batch_contents, embd))
            args = ','.join(cursor.mogrify("(%s,%s,%s,%s)", i).decode('utf-8') for i in values)
            sql_insert_query = f"""
                    INSERT INTO {TB_WIKI} (title, name, content, embedd)
                    VALUES"""
            cursor.execute(sql_insert_query + (args))
            connection.commit()
        current_id = count_row(tb_name=TB_WIKI)
        batch_titles = []
        batch_names = []
        batch_contents = []
        for idx, article in tqdm(enumerate(iter(snippets))):
            if idx%n_gpus != gpu_index: 
                continue

            if idx < current_id:
                continue
            batch_titles.append(str(article["section_title"]))
            batch_names.append(str(article["article_title"]))
            batch_contents.append(str(article["passage_text"]))
            if len(batch_contents) == BATCH:
                assert len(batch_titles) == len(batch_names) == len(batch_contents), \
                        f"len(batch_titles): {len(batch_titles)} "\
                        f"len(batch_names): {len(batch_names)}"\
                        f"len(batch_contents): {len(batch_contents)}"
                _insert(
                        batch_titles=batch_titles,
                        batch_names=batch_names,
                        batch_contents=batch_contents,
                        device=device
                )
                batch_titles.clear()
                batch_names.clear()
                batch_contents.clear()

        if batch_contents:
            assert len(batch_titles) == len(batch_names) == len(batch_contents), \
                    f"len(batch_titles): {len(batch_titles)} "\
                    f"len(batch_names): {len(batch_names)}"\
                    f"len(batch_contents): {len(batch_contents)}"

            _insert(
                    batch_titles=batch_titles,
                    batch_names=batch_names,
                    batch_contents=batch_contents,
                    device=device
            )


        logger.info(f"Insert knowledges to {TB_WIKI} successfully")

        if connection:
            cursor.close()
            connection.close()
    except (Exception, Error) as e:
        logger.error(f"Failed inserting knowledge into {TB_WIKI}: {e}")


def insert_client_knowledges(
        context_encoder: DPRContextEncoder,
        context_tokenizer: DPRContextEncoderTokenizer,
        snippets: pd.DataFrame,
        device: torch.device,
        )->None:
    """Insert client's knowledge to table

    Args:
        context_encoder: a model that encodes data to embedding
        context_tokenizer: a tokenizer that creates input for `context_encoder`
        snippets: dataset object that contains wikipedia snippet passages
    """
    logger.info(f"Starting inserting knowledge to {TB_CLIENT}")

    try:
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)

        cursor = connection.cursor()
        def _insert(
                batch_titles: List[str],
                batch_domains: List[str],
                batch_contents: List[str],
                device: torch.device
                ) -> None:

            passage_embd = get_ctx_embd(
                    model_encoder=context_encoder,
                    tokenizer=context_tokenizer,
                    text=batch_contents,
                    device=device
                    )
            embd = [str(list(passage_embd[i, :].cpu().detach().numpy().reshape(-1)))
                    for i in range(passage_embd.size(0))
                    ]
            values =  list(zip(batch_titles, batch_domains, batch_contents, embd))
            args = ','.join(cursor.mogrify("(%s,%s,%s,%s)", i).decode('utf-8') for i in values)
            sql_insert_query = f"""
                    INSERT INTO {TB_CLIENT} (title, domain, content, embedd)
                    VALUES"""
            cursor.execute(sql_insert_query + (args))
            connection.commit()
        current_id = count_row(tb_name=TB_CLIENT)
        batch_titles = []
        batch_domains = []
        batch_contents = []
        for idx, article in tqdm(enumerate(snippets.iterrows())):
            if idx < current_id:
                continue
            batch_titles.append(str(article[1]["Title"]))
            batch_domains.append(str(article[1]["Domain"]))
            batch_contents.append(str(article[1]["Content"]))
            if len(batch_contents) == BATCH:
                assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                        f"len(batch_titles): {len(batch_titles)} "\
                        f"len(batch_domains): {len(batch_domains)}"\
                        f"len(batch_contents): {len(batch_contents)}"
                _insert(
                        batch_titles=batch_titles,
                        batch_domains=batch_domains,
                        batch_contents=batch_contents,
                        device=device
                )
                batch_titles.clear()
                batch_domains.clear()
                batch_contents.clear()

        if batch_contents:
            assert len(batch_titles) == len(batch_domains) == len(batch_contents), \
                    f"len(batch_titles): {len(batch_titles)} "\
                    f"len(batch_names): {len(batch_domains)}"\
                    f"len(batch_contents): {len(batch_contents)}"

            _insert(
                    batch_titles=batch_titles,
                    batch_domains=batch_domains,
                    batch_contents=batch_contents,
                    device=device
            )


        logger.info(f"Insert knowledges to {TB_CLIENT} successfully")

        if connection:
            cursor.close()
            connection.close()
    except (Exception, Error) as e:
        logger.error(f"Failed inserting knowledge into {TB_CLIENT}: {e}")


def create_index(
        tb_name: str,
        num_data: int,
    ) -> None:
    """Create index for embedding column

    Args:
        num_data: number of data or number of rows in the table
    """
    try:
        print("Creating index")
        connection = psycopg2.connect(dbname=PGDBNAME,
                                      host=PGHOST,
                                      port=PGPORT,
                                      user=PGUSER,
                                      password=PGPWD)
        connection.autocommit = True

        cursor = connection.cursor()
        nlist =  round(2*math.sqrt(num_data))

        create_index_cluster_cmd = f'''
                SET maintenance_work_mem TO '14 GB';
                CREATE INDEX ON {tb_name} USING ivfflat (embedd vector_ip_ops) WITH (lists = {nlist});
                '''
        create_index_default_cmd = f'''
                CREATE INDEX ON {tb_name} USING ivfflat (embedd vector_ip_ops);
                '''
        try:
            logger.info(f"Creating index with {nlist} cluster")
            cursor.execute(create_index_cluster_cmd)
        except:
            logger.error(f"Created index clustering on {tb_name} was failed, try default settings")
            cursor.execute(create_index_default_cmd)
        logger.info("Create index successfully")
        if connection:
            cursor.close()
            connection.close()
            logger.info("PostgreSQL connection is closed")

    except (Exception, psycopg2.Error) as err:
        logger.error("Error while connecting to PostgreSQL", err)

