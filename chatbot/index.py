import os
import json
import glob
import logging
from pathlib import Path
from openai import OpenAI
from beaver import BeaverDB, Document
from chatbot.config import load

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def index():
    """
    Executes the data indexing process for the Chatbot using structured logging.

    This function performs the following steps:
    1.  Loads the application configuration.
    2.  Initializes the OpenAI client for generating embeddings.
    3.  Initializes the BeaverDB database connection.
    4.  Locates JSON data files in the configured directory.
    5.  Iterates through each JSON file and processes its content.
    6.  For each record, concatenates 'name' and 'description' to form the embedding text.
    7.  Generates vector embeddings using the configured model.
    8.  Stores the document, embedding, and metadata into a BeaverDB collection named after the source file.

    The function uses the logging module to report progress, warnings, and errors
    instead of standard output printing.
    """
    try:
        config = load()
    except FileNotFoundError:
        config = load("config.yaml")

    logger.info(f"Configuration loaded. DB: {config.db}")
    logger.info(f"Index folder: {config.index_files_folder}")

    client = OpenAI(
        base_url=config.embedding.base_url,
        api_key=config.embedding.api_key or "lm-studio"
    )

    db = BeaverDB(config.db)

    json_path = Path(config.index_files_folder)

    if not json_path.exists():
        logger.warning(f"Directory {json_path} does not exist. Creating it...")
        json_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Please place your .json files in {json_path}/")
        return

    json_files = glob.glob(str(json_path / "*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {json_path}/.")
        return

    logger.info(f"Files found: {[os.path.basename(f) for f in json_files]}")

    for file_path in json_files:
        filename = os.path.basename(file_path)
        collection_name = os.path.splitext(filename)[0]

        logger.info(f"Processing collection: '{collection_name}' from {filename}...")
        collection = db.collection(collection_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON {filename}: {e}")
            continue

        if not isinstance(data, list):
            logger.error(f"File {filename} must contain a list of objects.")
            continue

        count = 0
        for item in data:
            name = item.get("name", "")
            description = item.get("description", "")

            if not name and not description:
                continue

            text_to_embed = f"{name}. {description}"

            try:
                response = client.embeddings.create(
                    input=[text_to_embed],
                    model=config.embedding.model
                )
                embedding_vector = response.data[0].embedding

                doc = Document(
                    embedding=embedding_vector,
                    body=item,
                )

                collection.index(doc)
                count += 1
            except Exception as e:
                logger.error(f"Error indexing item '{name}': {e}")

        logger.info(f"Completed {collection_name}: {count} documents indexed.")

    db.close()
    logger.info("Indexing process finished.")

if __name__ == "__main__":
    index()