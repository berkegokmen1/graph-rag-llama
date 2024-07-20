from neo4j import GraphDatabase


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]


# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# def fetch_graph_data():
#     with driver.session() as session:
#         result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
#         nodes = set()
#         edges = []
#         for record in result:
#             nodes.add(record["n"].id)
#             nodes.add(record["m"].id)
#             edges.append((record["n"].id, record["m"].id))
#         return nodes, edges


# nodes, edges = fetch_graph_data()
# print(f"Nodes: {nodes}")
# print(f"Edges: {edges}")
