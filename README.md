# wiki-database
## Setup Environment
```bash
conda create -n wiki python==3.8 -y
conda activate wiki
pip install -r requirements.txt
```

## Setup Milvus server
### Requirement
Milvus server run on Docker server, please see these following links first:
1. https://docs.docker.com/engine/install/ 
2. https://milvus.io/docs/prerequisite-docker.md

### Install Milvus
Download `milvus-standalone-docker-compose.yml` and save it as `docker-compose.yml` manually, or with the following command:
```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

### Start Milvus
* In the same directory as the `docker-compose.yml` file, start up Milvus by running:
```
sudo docker-compose up -d
```

* Now check if same directory as the `docker-compose.yml` file has folder `volumes`, or run the following command:
```
sudo docker compose ps
```

* Connect to Milvus: 
```
docker port milvus-standalone 19530/tcp
```

## .env
- Please access `.env` file to modify database information in order to connect to the database
```bash
DBNAME="wiki_33m_milvus"
HOST="localhost"
PORT="19530"
TB_WIKI="wiki_tb"
TB_CLIENT="client_tb"
BATCH=200
```
## Implement
### Multi process (hotfix)
- If you using gpus, please specify `CUDA_VISIBLE_DEVICES`
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/run.py
```

### Create Wikipedia database
- if there is no database exists, run:
```bash
pthon src/run.py --init_db
```
- if database already exists (you have your own database) and just want to create the table, run:
```bash
python src/run.py --init_tb
```
- if database and table exist, just want to insert passages, run:
```bash
python src/run.py
```
- This script automatically create Wikipedia database with ~17M data. If you want to create Wikipedia database with ~33M data. add use argument: `--dataset_version wikipedia_en_100_0`
- When you have your own table filled up with data before and just want to create index, run:
```bash
python src/run.py --just_create_index \
                  --client_data_path </path/to/client/knowledge/csv>
```
### Create Client database
MSD knowledge can be download [here](https://drive.google.com/file/d/1S2i325zIv13O1IVoDj9jib9bsCUv20Pt/view?usp=sharing)
- if there is no database exists, run:
```bash
pthon src/run_client.py --init_db \
                        --client_data_path </path/to/client/knowledge/csv>
```
- if database already exists (you have your own database) and just want to create the table, run:
```bash
python src/run_client.py --init_tb \
                        --client_data_path </path/to/client/knowledge/csv>
```
- if database and table exist, just want to insert passages, run:
```bash
python src/run_client.py --client_data_path </path/to/client/knowledge/csv>
```
- When you have your own table filled up with data before and just want to create index, run:
```bash
python src/run.py --just_create_index \
                  --client_data_path </path/to/client/knowledge/csv>
```