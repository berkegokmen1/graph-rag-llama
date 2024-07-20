from src.models.node2vec import Node2VecModel
from src.models.base import PrecomputedEmbedding
from src.utils.json import CustomJSONEncoder
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.schema import TextNode

# from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex
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
        embedding_dim=64,
        model_class=Node2VecModel,
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding_dim = embedding_dim
        self.embedding_model = model_class(embedding_dim)
        self.graph = None
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.index = None
        self.query_engine = None

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
        self.llm = Ollama(model="llama3:instruct", request_timeout=360.0)
        Settings.llm = self.llm

    def fetch_graph_data(self):
        driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        G = nx.Graph()

        with driver.session() as session:
            self._fetch_nodes(session, G)
            self._fetch_relationships(session, G)

        driver.close()
        self.graph = G
        return G

    def _fetch_nodes(self, session, G):
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
            for record in batch:
                node_id = record["id"]
                labels = record["labels"]
                props = record["props"]
                G.add_node(node_id, label=labels[0], **props)
            offset += batch_size

    def _fetch_relationships(self, session, G):
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
            for record in batch:
                G.add_edge(record["source"], record["target"], type=record["type"])
            offset += batch_size

    def create_embeddings(self):
        if self.graph is None:
            raise ValueError("Graph data not fetched. Call fetch_graph_data() first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.embedding_model.process_graph(self.graph, device)
        model = self.embedding_model.create_model(device)
        self.embedding_model.train_model(model, device)
        self.embeddings = self.embedding_model.generate_embeddings(model, device)

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

        # Initialize embed Settings
        Settings.embed_model = embed_model
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.query_engine = self.index.as_query_engine()

    def run_prompt(self, prompt):
        response = self.query_engine.query(prompt)
        print("Response:", response)
