import spotipy
import spotipy.util as util
import math
import numpy as np
import pandas as pd
import networkx as nx
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from collections import defaultdict



app = dash.Dash(
    __name__, meta_tags=[{'name': 'viewport', 'content': 'width=device-width'}]
)


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
    raise RuntimeError('Cannot get token')


      
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
        row_names = [track[0], ', '.join(track[2])]
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
default_weight_threshold = 200
weight_baseline = 10
size_baseline = 30
default_time_range = 'medium_term'
# top_names = [] # fix this by not having to use in the node hover update (modify CSS styling)
# choosing_seeds = False
seeds = []
# adjacency_list = {}
artist_ids = {}

def get_weight(rank):
    return -rank + MIN_RANK + weight_baseline
def get_size(rank):
    return -rank + MIN_RANK + size_baseline


def get_graph_elements(time_range, threshold):
    nodes = []
    edges = []

    top_df = get_top_artists(time_range)
    top_names = [name for name in top_df['artist'].values]
    new_artist_edges = defaultdict(lambda: [])
    
    weights = defaultdict(lambda: 0)
    ranks = {}

    for (rank, row) in top_df.iterrows():
        current_name = row['artist']
        ranks[current_name] = rank
        artist_ids[current_name] = row['id']
        # adjacency_list[current_name] = []
        for related_artist in sp.artist_related_artists(row['id'])['artists']:
            related_name = related_artist['name']
            related_edge = {'data': {'source': current_name, 'target': related_name}, 'classes': ''}
            if related_name in top_names:
                if len([edge for edge in edges if edge['data']['source'] == related_name and edge['data']['target'] == current_name]) == 0:
                    edges.append(related_edge)                    
            else:
                new_artist_edges[related_name].append(related_edge)
                if related_name not in artist_ids:
                    artist_ids[related_name] = related_artist['id']

    for edge in edges:
        (u, v) = (edge['data']['source'], edge['data']['target'])
        weights[u] += get_weight(ranks[v])
        weights[v] += get_weight(ranks[u])

    for (new_name, new_edges) in new_artist_edges.items():
        for edge in new_edges:
            (u, v) = (edge['data']['source'], edge['data']['target'])
            # assert(edge['data']['source'] in top_names and edge['data']['target'] not in top_names)
            weights[v] += get_weight(ranks[u])

        if weights[new_name] >= threshold:
            nodes.append({'data': {'id': new_name, 'label': new_name,
                'size': get_size(MIN_RANK), 'weight': weights[new_name], 'activation': 0, 'selected': False, 'top': False}, 'classes': 'new'})
            edges += new_edges

    for (rank, name) in enumerate(top_names):
        nodes.append({'data': {'id': name, 'label': name, 'size': get_size(rank),
            'weight': weights[name], 'activation': 0, 'selected': False, 'top': True}, 'classes': 'top'})
 

    if include_related_edges:
        node_names = [node['data']['id'] for node in nodes]
        for name in node_names:
            if name not in top_names:
                for related_artist in sp.artist_related_artists(artist_ids[name])['artists']:
                    related_name = related_artist['name']
                    if related_name in node_names and {'data': {'source': related_name, 'target': name}} not in edges:
                        edges.append({'data': {'source': name, 'target': related_name}})

    return nodes + edges







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
    'gravity': 40,
    'componentSpacing': 200,
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
    style={'width': '100%', 'height': '100vh'},
    elements=[],
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
                # 'font-size': '24px',
                'color': '#fff',
                'text-valign': 'center',
                'text-halign': 'center'
                #'background-fit': 'contain'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': '2px',
                'opacity': 0.5
            }
        },
        # {
        #     'selector': '[activation = 0]',
        #     'style': {
        #         'background-color': 'yellow',
        #         'text-outline-color': 'yellow'
        #     }
        # },
        {
            'selector': '.selected-edge',
            'style': {
                'line-color': '#f00',
                'width': '4px',
                'z-index': 99
            }
        },
        # {
        #     'selector': '.selected-node',
        #     'style': {
        #         'background-color': '#f00',
        #         'text-outline-color': '#f00'
        #     }
        # },
        {
            'selector': '.selected-node',
            'style': {
                'background-color': '#f00000',
                'text-outline-color': '#f00000'
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
                'background-color': 'green',
                'text-outline-color': 'green'
            }
        },
        {
            'selector': '[activation >= 1]',
            'style': {
                'background-color': '#33C3F0',
                'text-outline-color': '#000'
            }
        }
    ]
)


expansion_marks = {}
for i in range(0, 4):
    expansion_marks[i] = {'label': str(i)}

exclude_dropdown_options = []

app.layout = html.Div(
    className='row',
    children=[
        html.Div(
            className='left-panel',
            children=[
                html.Div(
                    id='div-header',
                    children=[
                        html.H3(id='title-header', children='Artist Graph')
                    ]
                ),
                html.Div(
                    id='div-graph-options',
                    children=[
                        html.Label(id='time-range-label', htmlFor='time-range-dropdown', children='Time Range'),
                        dcc.Dropdown(
                            id='time-range-dropdown',
                            options=[
                                {'label': 'Short (4 weeks)', 'value': 'short_term'},
                                {'label': 'Medium (6 months)', 'value': 'medium_term'},
                                {'label': 'Long (all-time)', 'value': 'long_term'}
                            ],
                            value='medium_term',
                            searchable=False,
                            clearable=False
                        ),
                        html.Label(id='new-artist-threshold-label', htmlFor='new-artist-threshold-slider', children='New artist threshold'),
                        dcc.Slider(
                            id='new-artist-threshold-slider',
                            min=0,
                            max=1000,
                            step=50,
                            value=300,
                            tooltip={'always_visible': False, 'placement': 'right'}
                        ),
                        html.Label(id='artist-search-label', htmlFor='artist-search-dropdown', children='Artist Search'),
                        dcc.Dropdown(
                            id='artist-search-dropdown',
                            options=[],
                            placeholder='Search for an artist...'
                        ),
                        html.Ul(id='artist-tracks-list', children=[]),
                        # html.Div(
                        #     id='artist-tracks',
                        #     children=[
                        #         html.P(id='artist-tracks-header'),
                                
                        #     ],
                        # )
                    ]
                )
            ]
        ),
        html.Div(
            className='center-panel',
            children=[
                graph
            ]
        ),
        html.Div(
            className='right-panel',
            children=[
                # html.Div(
                #     id='playlist-header',
                #     children=[
                #         html.H3(id='playlist-title-header', children='Playlist Maker')
                #     ]
                # ),
                html.Button('New Playlist', id='new-playlist-btn', className='button-primary'),
                html.Div(
                    id='playlist-maker',
                    children=[
                        html.Label(id='playlist-size-label', htmlFor='playlist-size-slider', children='Playlist Size'),
                        dcc.Slider(
                            id='playlist-size-slider',
                            min=25,
                            max=200,
                            step=25,
                            value=50,
                            tooltip={'always_visible': False, 'placement': 'right'}
                        ),
                        html.Label(id='expansion-label', htmlFor='expansion-slider', children='Expansion Depth'),
                        dcc.Slider(
                            id='expansion-slider',
                            min=0,
                            max=3,
                            step=1,
                            value=1,
                            marks=expansion_marks
                        ),
                        dcc.Checklist(
                           id='exclude-playlist-tracks',
                           options=[
                               {'label': 'Exclude tracks in your current playlists', 'value': 1}
                           ]
                        ),
                        html.H6('Seeds', id='seed-header', className=''),
                        html.Div(id='seed-list', children=[]),
                        html.Button('Initialize', id='initialize-btn', className='button-primary'),
                        # html.Button('Cancel', id='cancel-btn', className='button-danger'),
                        html.Br(),
                        html.Div(
                            id='playlist-view',
                            className='display-none',
                            children=[
                                dcc.Input(
                                    id='playlist-name',
                                    type='text',
                                    placeholder='Name your playlist...'
                                ),
                                html.Ul(id='playlist', children=[]),
                                html.Button('Save to Spotify', id='save-playlist-btn', className='button-primary')
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)





def get_search_suggestions(text):
    query = "\"" + text.replace(" ", "+") + "\""
    results = sp.search(query, type='artist')
    return [{'label': artist['name'], 'value': artist['id']} for artist in results['artists']['items']]


def get_artist_tracks(artist_id):
    results = sp.artist_top_tracks(artist_id)
    return [html.Li(
        children=[
            html.P(className='track-name', children=track['name']),
            html.P(className='track-artists', children=', '.join([artist['name'] for artist in track['artists']])),
            html.Audio(className='track-audio', src=track['preview_url'], controls='controls')
        ]) for track in results['tracks']]


@app.callback(
    Output('artist-search-dropdown', 'options'),
    [
        Input('artist-search-dropdown', 'search_value'),
        Input('artist-search-dropdown', 'value')
    ],
    [State('artist-search-dropdown', 'options')])
def update_search_suggestions(search_value, value, options):
    if search_value:
        return get_search_suggestions(search_value)
    elif value:
        return [option for option in options if option['value'] == value]
    else:
        return options
    



def old_class(elem):
    if 'label' in elem['data']:
        if elem['data']['top']:
            return 'top'
        else:
            return 'new'
    else:
        return ''


@app.callback(
    [
        Output('artist-graph', 'elements'),
        Output('seed-list', 'children'),
        Output('artist-search-dropdown', 'value')
    ],
    [
        Input('time-range-dropdown', 'value'),
        Input('new-artist-threshold-slider', 'value'),
        Input('artist-graph', 'tapNodeData')
        # Input('expansion-slider', 'value')
    ],
    [
        State('artist-graph', 'elements'),
        State('seed-list', 'children'),
        State('artist-search-dropdown', 'value'),
        State('seed-header', 'className')
    ])
def update_artist_graph(time_range, threshold, node_data, elements, seed_list, artist_search_value, seed_header_class):
    ctx = dash.callback_context
    print("context: {}".format(ctx.triggered))
    trigger = ctx.triggered[0]
    if trigger['value'] == None or trigger['prop_id'] in ['new-artist-threshold-slider.value', 'time-range-dropdown.value']:
        return get_graph_elements(time_range, threshold), seed_list, artist_search_value
    elif trigger['prop_id'] == 'artist-graph.tapNodeData':
        seed = [node for node in elements if node['data']['id'] == node_data['id']][0]
        # print("Seed: {}".format(seed))
        choosing_seeds = (seed_header_class == 'choosing-seeds')
        print("choosing_seeds: {}".format(choosing_seeds))
        if choosing_seeds:
            if len([elem for elem in seed_list if elem['props']['children'] == seed['data']['label']]) > 0:
                seed_list = [elem for elem in seed_list if elem['props']['children'] != seed['data']['label']]
                # seed_list.remove(seed['data']['label'])
                seed['data']['activation'] = 0
            else:
                new_elem = html.P(seed['data']['label'], id=seed['data']['label'])
                # print(new_elem == seed['data']['label'])
                seed_list.append(new_elem)
                # print("adding {}".format(new_elem))
                # print("id: {}".format(new_elem.id))
                # print(seed_list)
                seed['data']['activation'] = 1
            return elements, seed_list, artist_search_value
        else:
            if seed['data']['selected'] == False:
                # seed['classes'] += ' selected-node'
                seed['data']['selected'] = True
                seed['classes'] = 'selected-node'
                for elem in elements:
                    if 'source' in elem['data']:
                        if elem['data']['source'] == node_data['id'] or elem['data']['target'] == node_data['id']:
                            elem['classes'] = 'selected-edge'
                        elif elem['classes'] == 'selected-edge':
                            elem['classes'] = old_class(elem)
                    elif 'label' in elem['data'] and elem['data']['id'] != seed['data']['id'] and elem['data']['selected'] == True:
                        elem['data']['selected'] = False
                        elem['classes'] = old_class(elem)
                        # print("UNSELECT {}".format(elem))
                return elements, seed_list, artist_ids[seed['data']['id']]
            else:
                seed['data']['selected'] = False
                seed['classes'] = old_class(seed)
                # print("UNSELECT {}".format(seed))
                for elem in elements:
                    if 'source' in elem['data'] and elem['classes'] == 'selected-edge':
                        elem['classes'] = old_class(elem)
                return elements, seed_list, ''
        # print("Selected:")
        # for node in elements:
        #     if 'selected' in node['data'] and node['data']['selected']:
        #         print(node)
    else:
        return elements, seed_list, artist_search_value


@app.callback(
    Output('artist-tracks-list', 'children'),
    [
        Input('artist-search-dropdown', 'value')
    ])
def update_artist_tracks(artist_id):
    if artist_id:
        return get_artist_tracks(artist_id)


@app.callback(
    [
        Output('playlist-maker', 'className'),
        Output('seed-header', 'className'),
        Output('playlist-view', 'className'),
        Output('playlist', 'children')
    ],
    [
        Input('new-playlist-btn', 'n_clicks'),
        Input('initialize-btn', 'n_clicks')
    ],
    [
        State('playlist-maker', 'className'),
        State('seed-header', 'className'),
        State('seed-list', 'children')
    ])
def toggle_playlist(playlist_clicks, initialize_clicks, playlist_maker_class, seed_header_class, seed_list):
    print("new playlist clicks: {}".format(playlist_clicks))
    if playlist_clicks == 0 or playlist_clicks == None:
        return 'display-none', ''
    elif playlist_clicks == 1:
        return '', 'choosing-seeds'
    else:
        return playlist_maker_class, seed_header_class



    
    

# @app.callback(
#     [
#         Output('playlist', 'children'),
#         Output('save-playlist-btn', 'style'),
#     ],
#     [
#         Input('initialize-btn', 'n_clicks'),
#     ])
# def init_playlist(n_clicks):
#     assert(choosing_seeds)
#     if n_clicks == 1:
#         choosing_seeds = False


# @app.callback(
#     Output('artist-graph', 'elements'),
#     [Input('artist-graph', 'tapNodeData')],
#     [State('artist-graph', 'elements')])
# def node_select(data, elements):
#     return elements


# @app.callback(
#     Output('artist-graph', 'elements'),
#     [Input('artist-graph', 'mouseoverNodeData')],
#     [State('artist-graph', 'elements')])
# def on_hover_node(data, elements):
#     if data:
#         for edge in elements:
#             if 'source' in edge['data']:
#                 if edge['data']['source'] == data['id'] or edge['data']['target'] == data['id']:
#                     edge['classes'] = 'selected-edge'
#                 elif 'classes' in edge and edge['classes'] == 'selected-edge':
#                     edge['classes'] = ''
#         for node in elements:
#             if node['data']['id'] == data['id']:
#                 node['classes'] = 'selected-node'
#             elif 'classes' in node and node['classes'] == 'selected-node':
#                 if node['data']['id'] in top_names:
#                     node['classes'] = 'top'
#                 else:
#                     node['classes'] = 'new'

#     return elements





if __name__ == '__main__':
    app.run_server(debug=True)
