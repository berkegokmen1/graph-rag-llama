from models.node2vec import Node2VecModel
from models.emb import PrecomputedEmbedding
from utils.json import CustomJSONEncoder
from models.llama import get_llama_model
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.schema import TextNode

# from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, set_global_tokenizer
from neo4j import GraphDatabase
import networkx as nx
import torch
from tqdm import tqdm
import json


class GraphEmbeddingRAG:
    def __init__(
        self,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llama_model,
        embedding_dim=64,
        model_class=Node2VecModel,
    ):

        # print(f"Neo4j URI: {neo4j_uri}, User: {neo4j_user}, Password: {neo4j_password}")
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.llama_model = llama_model
        self.embedding_dim = embedding_dim
        self.embedding_model = model_class(embedding_dim)
        self.graph = None
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.index = None
        self.query_engine = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._init_vector_store()
        self._init_llm()

    def _init_vector_store(self):
        self.vector_store = Neo4jVectorStore(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            embedding_dimension=self.embedding_dim,
            index_name="graph_vector_index",
            node_label="GraphNode",
        )

    def _init_llm(self):
        self.llm, self.tokenizer = get_llama_model(self.llama_model, self.device)

    def fetch_graph_data(self, max_nodes=None, max_rels=None):
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        G = nx.Graph()

        with driver.session() as session:
            self._fetch_nodes(session, G, max_nodes)
            self._fetch_relationships(session, G, max_rels)

        driver.close()
        self.graph = G
        return G

    def _fetch_nodes(self, session, G, max_nodes=None):
        batch_size = 1000
        offset = 0
        while True:
            result = session.run(
                f"MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props "
                f"SKIP {offset} LIMIT {batch_size}"
            )
            batch = list(result)
            if not batch:
                break

            if max_nodes is not None and offset >= max_nodes:
                break

            for record in batch:
                node_id = record["id"]
                labels = record["labels"]
                props = record["props"]
                props.pop("label", None)
                G.add_node(node_id, label=labels[0], **props)
            offset += batch_size

    def _fetch_relationships(self, session, G, max_rels=None):
        batch_size = 1000
        offset = 0
        while True:
            result = session.run(
                f"MATCH ()-[r]->() RETURN id(startNode(r)) AS source, id(endNode(r)) AS target, type(r) AS type "
                f"SKIP {offset} LIMIT {batch_size}"
            )
            batch = list(result)
            if not batch:
                break

            if max_rels is not None and offset >= max_rels:
                break

            for record in batch:
                G.add_edge(record["source"], record["target"], type=record["type"])
            offset += batch_size

    def create_embeddings(self):
        if self.graph is None:
            raise ValueError("Graph data not fetched. Call fetch_graph_data() first.")

        self.embedding_model.process_graph(self.graph, self.device)
        model = self.embedding_model.create_model(self.device)
        self.embedding_model.train_model(model, self.device)
        self.embeddings = self.embedding_model.generate_embeddings(model, self.device)

    def add_to_vector_store(self):
        if self.graph is None or self.embeddings is None:
            raise ValueError(
                "Graph data or embeddings not created. Call fetch_graph_data() and create_embeddings() first."
            )

        batch_size = 1000
        nodes = list(self.graph.nodes(data=True))
        for i in tqdm(range(0, len(nodes), batch_size), desc="Adding to vector store"):
            batch = nodes[i : i + batch_size]
            nodes_to_add = []
            for node in batch:
                node_id, node_data = node
                embedding = self.embeddings[node_id].tolist()

                json_safe_props = json.loads(json.dumps(node_data, cls=CustomJSONEncoder))

                text = f"Node ID: {node_id}, Label: {json_safe_props['label']}, "
                text += ", ".join([f"{k}: {v}" for k, v in json_safe_props.items() if k != "label"])

                metadata = {
                    "node_id": node_id,
                    "label": json_safe_props["label"],
                }
                metadata.update({k: v for k, v in json_safe_props.items() if k != "label"})

                text_node = TextNode(text=text, embedding=embedding, metadata=metadata)
                nodes_to_add.append(text_node)

            self.vector_store.add(nodes_to_add)

    def init_index_and_query_engine(self):
        embed_model = PrecomputedEmbedding(self.embeddings)

        self.index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=embed_model)
        self.query_engine = self.index.as_query_engine(llm=self.llm)

    def run_prompt(self, prompt):
        response = self.query_engine.query(prompt)
        print("Response:", response)
