from rag.rag import GraphEmbeddingRAG
from models.infomax import DeepGraphInfomaxModel
from models.node2vec import Node2VecModel
from config.config import get_config
import warnings


def main():
    config = get_config("./src/config/config.yaml")
    print(config)

    neo4j_config = config["neo4j"]
    llama_config = config["llama"]

    rag_dgi = GraphEmbeddingRAG(**{**neo4j_config, **llama_config}, model_class=Node2VecModel)

    rag_dgi.fetch_graph_data(max_nodes=1000, max_rels=2000)
    rag_dgi.create_embeddings()
    rag_dgi.add_to_vector_store()
    rag_dgi.init_index_and_query_engine()

    rag_dgi.run_prompt("Who follows a person first named Catherine and last named Jones?")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
