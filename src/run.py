import os
import logging
import torch

from configs.arguments import Arguments
from database import make_database
from data import make_data
from database.make_database import create_milvus_connection
from pymilvus import utility, Collection
from sentence_transformers import SentenceTransformer

import dotenv

dotenv.load_dotenv()

DBNAME=os.getenv("DBNAME", "wiki_33m_milvus")
HOST=os.getenv("HOST", "localhost")
PORT=os.getenv("PORT", "19530")
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
TB_CLIENT=os.getenv("TB_CLIENT", "client_tb")
BATCH=int(os.getenv("BATCH", 10000))

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def main():
    arguments = Arguments()
    args = arguments.parse()

    # VALIDATE
    # row = 17553713 if args.dataset_version == "wiki40b_en_100_0" else 33849898
    max_rows = None
    if args.max_rows:
        max_rows = args.max_rows
    elif args.dataset_version == "wiki40b_en_100_0":
        max_rows = 17553713
    else: 
        max_rows = 33849898

    # upper case metric_type and index_type    
    index_type = args.index_type.upper()
    metrics_type = args.metrics_type.lower()

    # connect milvus server
    create_milvus_connection(
        host=args.host or HOST,
        port=args.port or PORT,
    )

    # start to create
    if not args.just_create_index:
        # Init collection
        collection = None

        # Initialize model
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer(args.model_name_or_path)
        # specify target_devices for multi_gpus/gpu/cpu
        target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        # get dim of output's embedding
        sentences = ["only for get embeddings"]
        embeddings = model.encode(sentences)
        model_dim = int(embeddings.shape[-1])

        # check if create collection
        if args.init_tb:
            if utility.has_collection(args.tbname or TB_WIKI):
                print("Drop collection")
                utility.drop_collection(args.tbname or TB_WIKI)
            print("Create collection")
            # create collection with dim of model's embedding
            collection = make_database.create_wiki_table(
                model_dim=model_dim
            )
        elif not utility.has_collection(args.tbname or TB_WIKI): 
            raise ValueError("Not having table wiki, please initialize first with `--init_tb` arguements")
        else: 
            collection = Collection(args.tbname or TB_WIKI)  
        
        print("Start downloading dataset")
        wiki_snippets = make_data.download_dataset(
            dataset_name=args.dataset_name,
            dataset_version = args.dataset_version,
            streaming=args.streaming
        )

        print("Start inserting knowledges")
        make_database.using_sbert_encode_multi_gpu(
            collection=collection,
            model=model,
            snippets=wiki_snippets,
            target_devices=target_devices, 
            batch_insert=args.batch_insert or BATCH,
            limit_samples=max_rows
        )
        print("building index for tables")
        make_database.build_indexs(
            collection=collection,
            filed_name="embedding",
            index_type=index_type,
            metric_type=metrics_type,
        )
    else:
        # logger.warning("Only index initialization is perfomed, make sure your table is filled up with data.")
        print("Only index initialization is perfomed, make sure your table is filled up with data.")
        collection = Collection(TB_WIKI or args.tbname)  
        collection.release()
        collection.drop_index(
            filed_name="embedding",
        )
        make_database.build_indexs(
            collection=collection,
            filed_name="embedding",
            index_type=index_type,
            metric_type=metrics_type,
        )

if __name__=="__main__":
    main()


