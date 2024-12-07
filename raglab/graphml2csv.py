import os
import networkx as nx
import pandas as pd


def construct_node_df(nx_graph) -> pd.DataFrame:
    nodes_data = []
    for node, data in nx_graph.nodes(data=True):
        data['id'] = node
        nodes_data.append(data)

    node_df = pd.DataFrame(nodes_data)

    node_df['entity_id'] = node_df.index

    # Change the order of columns
    cols = node_df.columns.tolist()
    cols = [cols[-1], cols[-2], *cols[:-2]]
    node_df = node_df[cols]

    # Rename columns
    node_df = node_df.rename(columns={
        'id': 'entity_name',
        'source_id': 'entity_chunk_id',
        'description': 'entity_description',
    })

    return node_df


def construct_edge_df(nx_graph, node_df) -> pd.DataFrame:
    edge_data = []
    for u, v, data in nx_graph.edges(data=True):
        data['source'] = u
        data['target'] = v
        edge_data.append(data)

    edge_df = pd.DataFrame(edge_data)

    # Build the node entity2id mapping
    entity2id = node_df.set_index('entity_name')['entity_id'].to_dict()
    edge_df['source_entity_id'] = edge_df['source'].map(entity2id)
    edge_df['target_entity_id'] = edge_df['target'].map(entity2id)

    # Merge source and target columns with node_df
    edge_df = edge_df.merge(node_df[['entity_id', 'entity_name', 'entity_type', 'entity_description',
                            'entity_chunk_id']], left_on='source_entity_id', right_on='entity_id', how='left')
    # edge_df = edge_df.drop(columns=['source', 'entity_id'])
    # edge_df = edge_df.rename(columns={'entity_name': 'source'})

    edge_df = edge_df.merge(node_df[['entity_id', 'entity_name', 'entity_type', 'entity_description',
                            'entity_chunk_id']], left_on='target_entity_id', right_on='entity_id', how='left')
    # edge_df = edge_df.drop(columns=['target', 'entity_id'])
    # edge_df = edge_df.rename(columns={'entity_name': 'target'})

    # # Change the order of columns
    # cols = edge_df.columns.tolist()
    # cols = [cols[-2], cols[-1], *cols[:-2]]
    # edge_df = edge_df[cols]

    # Rename columns
    edge_df = edge_df.rename(columns={
        'weight': 'relation_weight',
        'description': 'relation_description',
        'keywords': 'relation_keywords',
    })

    # Reorder columns
    edge_df = edge_df[['source_entity_id', 'target_entity_id', 
                       'relation_weight', 'relation_description', 'relation_keywords',
                       'entity_name_x', 'entity_type_x', 'entity_description_x', 'entity_chunk_id_x',
                       'entity_name_y', 'entity_type_y', 'entity_description_y', 'entity_chunk_id_y',]]

    return edge_df


if __name__ == '__main__':
    FILE_HOME = '../datasets/UltraDomain/working_dir/cs/'
    graph_path = f"{FILE_HOME}graph_chunk_entity_relation.graphml"
    nx_graph = nx.read_graphml(graph_path)
    nodes_df = construct_node_df(nx_graph)
    edges_df = construct_edge_df(nx_graph, nodes_df)

    if not os.path.exists(f"{FILE_HOME}csv/"):
        os.makedirs(f"{FILE_HOME}csv/")

    nodes_df.to_csv(f"{FILE_HOME}csv/nodes.csv", index=False)
    edges_df.to_csv(f"{FILE_HOME}csv/edges.csv", index=False)
