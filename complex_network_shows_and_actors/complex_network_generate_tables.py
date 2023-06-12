import pandas as pd
from tqdm import tqdm
from itertools import combinations


def create_nodes_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    actors = {}
    
    for _, row in df.iterrows():
        cast = row['cast']
        if isinstance(cast, str):
            cast = cast.split(', ')
            for actor in cast:
                if actor in actors:
                    actors[actor] += 1
                else:
                    actors[actor] = 1
    
    nodes = pd.DataFrame({'Id': list(actors.keys()), 'Label': list(actors.keys()), 'NumShows': list(actors.values())})
    nodes = nodes.head(1000)
    nodes.to_csv(output_file, index=False)
    print("Nodes CSV file has been created successfully.")


def create_edges_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    actor_edges = {}
    
    for _, row in df.iterrows():
        actors = row['cast']
        
        if isinstance(actors, str):
            actors = actors.split(', ')
            
            for actor in actors:
                if actor not in actor_edges:
                    actor_edges[actor] = set()
                
                actor_edges[actor].update(actors)
    
    edges_data = []
    
    for actor, connected_actors in tqdm(actor_edges.items()):
        for connected_actor in connected_actors:
            if actor != connected_actor:
                edges_data.append([actor, connected_actor])
    
    edges_df = pd.DataFrame(edges_data, columns=['Source', 'Target'])
    edges_df = edges_df.head(5000)
    edges_df.to_csv(output_file, index=False)
    print("Edges CSV file has been created successfully.")


if __name__ == "__main__":
    input_file = 'dataset/netflix_titles.csv'
    nodes_output_file = 'complex_network_shows_and_actors/nodes.csv'
    edges_output_file = 'complex_network_shows_and_actors/edges.csv'
    create_nodes_csv(input_file, nodes_output_file)
    create_edges_csv(input_file, edges_output_file)
