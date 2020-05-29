import spotipy
import spotipy.util as util
import math
import numpy as np
import pandas as pd
import networkx as nx
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
from collections import defaultdict


ANDREW = '1258447710'
ARNAV  = '67th4pl1pr8noy9kg5p19llnf'

SPOTIPY_CLIENT_ID=''
SPOTIPY_CLIENT_SECRET=''
MIN_RANK = 50

username = ANDREW
scope = 'user-top-read'
token = util.prompt_for_user_token(username,
                        scope,
                        client_id=SPOTIPY_CLIENT_ID,
                        client_secret=SPOTIPY_CLIENT_SECRET,
                        redirect_uri='http://localhost:8080')

if token:
    sp = spotipy.Spotify(auth=token)
else:
    raise RuntimeError("Cannot get token")


def get_ratings(path):
    ratings = []
    with open(path, 'r') as f:
        for rating in f.readlines():
            ratings.append(int(rating.strip()))
    return ratings

      
def add_to_table(table, results):
    for i, item in enumerate(results['items']):
        track = item['track']
        row = [track['name'], track['id'], [artist['name'] for artist in track['artists']], [artist['id'] for artist in track['artists']], track['album']['name'], track['album']['id']]
        table.append(row)
    return table
      
  
def get_table(playlist):
    results = sp.playlist(playlist['id'], fields='tracks,next')
    tracks = results['tracks']
    table = add_to_table([], tracks)
    while tracks['next']:
        tracks = sp.next(tracks)
        table = add_to_table(table, tracks)
    return table


def get_feature_df(track_table, feature_data):
    features = ['acousticness', 'danceability', 'energy', 'loudness', 'speechiness', 'valence', 'tempo', 'mode', 'key', 'time_signature']
    names = ['track_name', 'artist_names']
    feature_table = []
    for i in range(len(track_table)):
        track = track_table[i]
        track_features = feature_data[i]
        row_names = [track[0], ", ".join(track[2])]
        row_features = [track_features[feature] for feature in features]
        feature_table.append(row_names + row_features)
    
    feature_df = pd.DataFrame(feature_table, columns=(names + features))
    return feature_df


def flatten(deep_list):
    return [x for sub in deep_list for x in sub]


def get_data(playlist):
    table = get_table(playlist)
    table_df = pd.DataFrame(table, columns=['track_name', 'track_id', 'artist_names', 'artist_ids', 'album_name', 'album_id'])
    id_groups = np.array_split(table_df['track_id'].values, math.ceil(len(table_df) / 100))
    features = flatten([sp.audio_features(id_group) for id_group in id_groups])
    feature_df = get_feature_df(table, features)
    return table_df, feature_df


def print_col(playlist_data, col):
    for entry in playlist_data[col]:
        print(entry)


def get_related_artists(artist_id):
    related_artists = sp.artist_related_artists(artist_id)
    related_table = []
    for artist in related_artists['artists']:
        related_table.append([artist['name'], artist['popularity'], artist['id']])
    related_df = pd.DataFrame(related_table, columns=['artist', 'pop', 'id'])
    return related_df


def get_top_artists(time_range):
    top_artists = sp.current_user_top_artists(limit=50, time_range=time_range)
    remove_artists = ['Lata Mangeshkar', 'Alka Yagnik', 'Amitabh Bachchan', 'Josh A', 'Prince', 'KILLY']
    top_table = []
    for artist in top_artists['items']:
        top_table.append([artist['name'], artist['popularity'], artist['id']])
    top_df = pd.DataFrame(top_table, columns=['artist', 'pop', 'id'])
    top_df = top_df.loc[~top_df['artist'].isin(remove_artists)].reset_index(drop=True)
    return top_df


# Parameters
include_related_edges = False
weight_threshold = 200
weight_baseline = 10
size_baseline = 30
time_range = 'medium_term'

def get_weight(rank):
    return -rank + MIN_RANK + weight_baseline
def get_size(rank):
    return -rank + MIN_RANK + size_baseline

nodes = []
edges = []

# top_df_short = get_top_artists('short_term')
# top_df_med = get_top_artists('medium_term')
# top_df_long = get_top_artists('long_term')
# top_df_combined = top_df_short.append(top_df_med).append(top_df_long).drop_duplicates()

top_df = get_top_artists(time_range)
top_names = [name for name in top_df['artist'].values]
new_artist_edges = defaultdict(lambda: [])
new_artist_ids = {}
weights = defaultdict(lambda: 0)
ranks = {}

for (rank, row) in top_df.iterrows():
    current_name = row['artist']
    ranks[current_name] = rank
    for related_artist in sp.artist_related_artists(row['id'])['artists']:
        related_name = related_artist['name']
        related_edge = {'data': {'source': current_name, 'target': related_name}}
        if related_name in top_names:
            if {'data': {'source': related_name, 'target': current_name}} not in edges:
                edges.append(related_edge)
        else:
            new_artist_edges[related_name].append(related_edge)
            if related_name not in new_artist_ids:
                new_artist_ids[related_name] = related_artist['id']

for edge in edges:
    (u, v) = (edge['data']['source'], edge['data']['target'])
    weights[u] += get_weight(ranks[v])
    weights[v] += get_weight(ranks[u])

for (new_name, new_edges) in new_artist_edges.items():
    for edge in new_edges:
        (u, v) = (edge['data']['source'], edge['data']['target'])
        # assert(edge['data']['source'] in top_names and edge['data']['target'] not in top_names)
        weights[v] += get_weight(ranks[u])

    if weights[new_name] >= weight_threshold:
        nodes.append({'data': {'id': new_name, 'label': new_name + " " + str(weights[new_name]), 'size': get_size(MIN_RANK), 'weight': weights[new_name]}, 'classes': 'new'})
        edges += new_edges

for (rank, name) in enumerate(top_names):
    nodes.append({'data': {'id': name, 'label': name + " " + str(weights[name]), 'size': get_size(rank), 'weight': weights[name]}, 'classes': 'top'})
 

if include_related_edges:
    node_names = [node['data']['id'] for node in nodes]
    for name in node_names:
        if name not in top_names:
            for related_artist in sp.artist_related_artists(new_artist_ids[name])['artists']:
                related_name = related_artist['name']
                if related_name in node_names and {'data': {'source': related_name, 'target': name}} not in edges:
                    edges.append({'data': {'source': name, 'target': related_name}})




cose_medium = {
    'name': 'cose',
    'animate': False,
    'randomize': True, 
    'edgeElasticity': 400, 
    'nodeRepulsion': 400000,
    'nodeOverlap': 400000,
    'gravity': 40,
    # 'idealEdgeLength': 100,
    'componentSpacing': 150,
    'nodeDimensionsIncludeLabels': False
}


cose_medium2 = {
    'name': 'cose',
    'animate': False,
    'randomize': True, 
    'edgeElasticity': 400, 
    'nodeRepulsion': 4000000,
    'nodeOverlap': 4000000,
    'gravity': 40,
    'componentSpacing': 250,
    'nodeDimensionsIncludeLabels': False
}

cose_medium3 = {
    'name': 'cose',
    'animate': False,
    'randomize': True, 
    'edgeElasticity': 200, 
    'nodeRepulsion': 2000000,
    'nodeOverlap': 2000000,
    'gravity': 200,
    'componentSpacing': 250,
    'nodeDimensionsIncludeLabels': True
}


cose_long = {
    'name': 'cose',
    'animate': False,
    'randomize': True, 
    'edgeElasticity': 400, 
    'nodeRepulsion': 4000000,
    'nodeOverlap': 4000000,
    'gravity': 40,
    # 'idealEdgeLength': 100,
    'componentSpacing': 20,
    'nodeDimensionsIncludeLabels': False
}


graph = cyto.Cytoscape(
    id='artist-graph',
    layout=cose_medium3,
    style={'width': '1280px', 'height': '800px'},
    elements=(nodes + edges),
    stylesheet=[
        {
            'selector': 'node',
            'style': {
                # 'label': 'data(weight)',
                'width': 'data(size)',
                'height': 'data(size)',
                'label': 'data(label)',
                # 'background-color': '#555',
                # 'text-outline-color': '#555',
                'text-outline-width': '2px',
                'color': '#fff',
                'text-valign': 'center',
                'text-halign': 'center'
                #'background-fit': 'contain'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': '2px'
            }
        },
        {
            'selector': '.selected-node',
            'style': {
                'background-color': '#f00',
                'text-outline-color': '#f00'
            }
        },
        {
            'selector': '.selected-edge',
            'style': {
                'line-color': '#f00'
            }
        },
        {
            'selector': '.top',
            'style': {
                'background-color': '#555',
                'text-outline-color': '#555'
            }
        },
        {
            'selector': '.new',
            'style': {
                'background-color': '#00f',
                'text-outline-color': '#00f'
            }
        }
    ]
)


app = dash.Dash(__name__)
app.layout = html.Div([
    graph
])


@app.callback(Output('artist-graph', 'elements'),
                [Input('artist-graph', 'mouseoverNodeData')],
                [State('artist-graph', 'elements')])
def onHoverNode(data, elements):
    if data:
        for edge in elements:
            if 'source' in edge['data']:
                if edge['data']['source'] == data['id'] or edge['data']['target'] == data['id']:
                    edge['classes'] = 'selected-edge'
                elif 'classes' in edge and edge['classes'] == 'selected-edge':
                    edge['classes'] = ''
        for node in elements:
            if node['data']['id'] == data['id']:
                node['classes'] = 'selected-node'
            elif 'classes' in node and node['classes'] == 'selected-node':
                if node['data']['id'] in top_names:
                    node['classes'] = 'top'
                else:
                    node['classes'] = 'new'

    return elements





if __name__ == '__main__':
    app.run_server(debug=True)
