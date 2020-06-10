import math
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import networkx as nx
import numpy as np
import pandas as pd
import spotipy
import spotipy.util as util
from dash.dependencies import Input, Output, State


ANDREW = '1258447710'
ARNAV  = '67th4pl1pr8noy9kg5p19llnf'

SPOTIPY_CLIENT_ID=''
SPOTIPY_CLIENT_SECRET=''

username = ANDREW
scope = 'user-top-read playlist-modify-private'


app = dash.Dash(
    __name__, meta_tags=[{'name': 'viewport', 'content': 'width=device-width'}]
)


def get_spotipy(token):
    return spotipy.Spotify(auth=token)

      
def flatten(deep_list):
    return [x for sub in deep_list for x in sub]


def get_top_artists(time_range, sp):
    top_artists = sp.current_user_top_artists(limit=50, time_range=time_range)
    remove_artists = ['Lata Mangeshkar', 'Alka Yagnik', 'Amitabh Bachchan', 'Josh A', 'Prince', 'KILLY']
    top_table = []
    for artist in top_artists['items']:
        top_table.append([artist['name'], artist['popularity'], artist['id']])
    top_df = pd.DataFrame(top_table, columns=['artist', 'pop', 'id'])
    top_df = top_df.loc[~top_df['artist'].isin(remove_artists)].reset_index(drop=True)
    return top_df


# Parameters
default_weight_threshold = 200
weight_baseline = 10
size_baseline = 30
size_multiplier = 1
min_rank = 50
default_time_range = 'medium_term'


def get_weight(rank):
    return -rank + min_rank + weight_baseline

def get_size(rank):
    return ((-rank + min_rank) * size_multiplier) + size_baseline


def get_graph_elements(time_range, threshold, sp):
    nodes = []
    edges = []

    top_df = get_top_artists(time_range, sp)
    top_ids = [id for id in top_df['id'].values]
    new_artist_edges = defaultdict(lambda: [])
    
    weights = defaultdict(lambda: 0)
    ranks = {}
    artist_names = {}

    for (rank, row) in top_df.iterrows():
        current_name = row['artist']
        current_id = row['id']
        ranks[current_id] = rank
        artist_names[current_id] = current_name
        for related_artist in sp.artist_related_artists(row['id'])['artists']:
            related_name = related_artist['name']
            related_id = related_artist['id']
            related_edge = {'data': {'source': current_id, 'target': related_id}, 'classes': ''}
            if related_id in top_ids:
                if len([edge for edge in edges if edge['data']['source'] == related_id and edge['data']['target'] == current_id]) == 0:
                    edges.append(related_edge)                    
            else:
                new_artist_edges[related_id].append(related_edge)
                if related_id not in artist_names:
                    artist_names[related_id] = related_name

    for edge in edges:
        (u, v) = (edge['data']['source'], edge['data']['target'])
        weights[u] += get_weight(ranks[v])
        weights[v] += get_weight(ranks[u])

    for (new_id, new_edges) in new_artist_edges.items():
        for edge in new_edges:
            (u, v) = (edge['data']['source'], edge['data']['target'])
            weights[v] += get_weight(ranks[u])

        if weights[new_id] >= threshold:
            nodes.append({'data': {'id': new_id, 'label': artist_names[new_id],
                'size': get_size(min_rank), 'weight': weights[new_id], 'activation': 0, 'selected': False, 'top': False}, 'classes': 'new'})
            edges += new_edges

    for (rank, id) in enumerate(top_ids):
        nodes.append({'data': {'id': id, 'label': artist_names[id], 'size': get_size(rank),
            'weight': weights[id], 'activation': 0, 'selected': False, 'top': True}, 'classes': 'top'})

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
    'edgeElasticity': 100, 
    'nodeRepulsion': 1000000,
    'nodeOverlap': 1000000,
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
            id='token',
            className='display-none'
        ),
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
                        html.Ul(id='artist-tracks-list', className='track-list', children=[]),
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
                html.Button('New Playlist', id='new-playlist-btn', className='button-primary'),
                html.Div(
                    id='playlist-maker',
                    children=[
                        html.Div(
                            id='playlist-form',
                            children=[
                                html.Label(id='playlist-size-label', htmlFor='playlist-size-slider', children='Playlist Size'),
                                dcc.Slider(
                                    id='playlist-size-slider',
                                    min=10,
                                    max=300,
                                    step=10,
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
                                html.Br()
                            ]
                        ),
                        html.Div(
                            id='playlist-view',
                            className='display-none',
                            children=[
                                dcc.Input(
                                    id='playlist-name',
                                    type='text',
                                    placeholder='Name your playlist...'
                                ),
                                html.Ul(id='playlist', className='track-list', children=[]),
                                html.H6(id='playlist-size'),
                                html.Button('Save to Spotify', id='save-playlist-btn', className='button-primary'),
                                html.H6(id='confirm-save')
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)





def get_search_suggestions(text, sp):
    query = "\"" + text.replace(" ", "+") + "\""
    results = sp.search(query, type='artist')
    return [{'label': artist['name'], 'value': artist['id']} for artist in results['artists']['items']]


def get_top_tracks(artist_id, sp):
    results = sp.artist_top_tracks(artist_id)
    return [html.Li(
        children=[
            html.P(className='track-name', children=track['name']),
            html.P(className='track-artists', children=', '.join([artist['name'] for artist in track['artists']])),
            html.P(className='track-id display-none', children=track['id']),
            html.Audio(className='track-audio', src=track['preview_url'], controls='controls')
        ]) for track in results['tracks']]


def old_class(elem):
    if 'label' in elem['data']:
        if elem['data']['top']:
            return 'top'
        else:
            return 'new'
    else:
        return ''


def is_node(elem):
    return 'label' in elem['data']
    


def initialize_playlist(elements, playlist_size, expansion_depth, exclude_tracks, sp):
    adjacency_list = {}    
    for elem in elements:
        if is_node(elem):
            adjacency_list[elem['data']['id']] = (elem, [])    
    for elem in elements:
        if not is_node(elem):
            (source, target) = (elem['data']['source'], elem['data']['target'])
            adjacency_list[source][1].append(target)
    
    seeds = [key for (key, value) in adjacency_list.items() if value[0]['data']['activation'] == 1]
    songs_per_artist = math.ceil(playlist_size / len(seeds))
    playlist = []
    for seed in seeds:
        playlist += get_top_tracks(seed, sp)[:songs_per_artist]

    return playlist


    # for seed in seeds:
    #     artist_queue = [seed]

    # while len(seeds) > 0:
    #     artist = seeds.pop()
    #     node = adjacency_list[artist]
    #     activation = node[0]['data']['activation']
    #     for other_artist in node[1]:
    #         adjacency_list[other_artist][0]['data']['activation'] += (activation / 2)
            




@app.callback(
    Output('artist-search-dropdown', 'options'),
    [
        Input('artist-search-dropdown', 'search_value'),
        Input('artist-search-dropdown', 'value')
    ],
    [
        State('artist-search-dropdown', 'options'),
        State('token', 'children')
     ])
def update_search_suggestions(search_value, value, options, token):
    sp = get_spotipy(token)
    if search_value:
        return get_search_suggestions(search_value, sp)
    elif value:
        return [option for option in options if option['value'] == value]
    else:
        return options
    


@app.callback(
    Output('artist-tracks-list', 'children'),
    [Input('artist-search-dropdown', 'value')],
    [State('token', 'children')])
def update_artist_tracks(artist_id, token):
    sp = get_spotipy(token)
    if artist_id:
        return get_top_tracks(artist_id, sp)



@app.callback(
    [
        Output('artist-graph', 'elements'),
        Output('seed-list', 'children'),
        Output('artist-search-dropdown', 'value'),
        Output('token', 'children')
    ],
    [
        Input('time-range-dropdown', 'value'),
        Input('new-artist-threshold-slider', 'value'),
        Input('artist-graph', 'tapNodeData')
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
    token = util.prompt_for_user_token(username,
                        scope,
                        client_id=SPOTIPY_CLIENT_ID,
                        client_secret=SPOTIPY_CLIENT_SECRET,
                        redirect_uri='http://localhost:8080')
    sp = get_spotipy(token)

    if trigger['value'] == None or trigger['prop_id'] in ['new-artist-threshold-slider.value', 'time-range-dropdown.value']:
        return get_graph_elements(time_range, threshold, sp), seed_list, artist_search_value, token
    elif trigger['prop_id'] == 'artist-graph.tapNodeData':
        seed = [node for node in elements if node['data']['id'] == node_data['id']][0]
        choosing_seeds = (seed_header_class == 'choosing-seeds')
        print("choosing_seeds: {}".format(choosing_seeds))
        if choosing_seeds:
            if len([elem for elem in seed_list if elem['props']['children'] == seed['data']['label']]) > 0:
                seed_list = [elem for elem in seed_list if elem['props']['children'] != seed['data']['label']]
                seed['data']['activation'] = 0
            else:
                new_elem = html.P(seed['data']['label'], id=seed['data']['label'])
                seed_list.append(new_elem)
                seed['data']['activation'] = 1
            return elements, seed_list, artist_search_value, token
        else:
            if seed['data']['selected'] == False:
                seed['data']['selected'] = True
                seed['classes'] = 'selected-node'
                for elem in elements:
                    if not is_node(elem):
                        if elem['data']['source'] == node_data['id'] or elem['data']['target'] == node_data['id']:
                            elem['classes'] = 'selected-edge'
                        elif elem['classes'] == 'selected-edge':
                            elem['classes'] = old_class(elem)
                    elif elem['data']['id'] != seed['data']['id'] and elem['data']['selected'] == True:
                        elem['data']['selected'] = False
                        elem['classes'] = old_class(elem)
                return elements, seed_list, seed['data']['id'], token
            else:
                seed['data']['selected'] = False
                seed['classes'] = old_class(seed)
                for elem in elements:
                    if not is_node(elem) and elem['classes'] == 'selected-edge':
                        elem['classes'] = old_class(elem)
                return elements, seed_list, '', token
    else:
        return elements, seed_list, artist_search_value, token




@app.callback(
    [
        Output('playlist-form', 'className'),
        Output('seed-header', 'className'),
        Output('playlist-view', 'className'),
        Output('new-playlist-btn', 'className'),
        Output('playlist', 'children'),
        Output('playlist-size', 'children')
    ],
    [
        Input('new-playlist-btn', 'n_clicks'),
        Input('initialize-btn', 'n_clicks'),
        Input('playlist-size-slider', 'value'),
        Input('exclude-playlist-tracks', 'value'),
        Input('expansion-slider', 'value')
    ],
    [
        State('playlist-maker', 'className'),
        State('seed-header', 'className'),
        State('playlist-view', 'className'),
        State('new-playlist-btn', 'className'),
        State('playlist', 'children'),
        State('artist-graph', 'elements'),
        State('token', 'children')
    ])
def toggle_playlist(playlist_clicks, initialize_clicks, playlist_size, exclude_tracks, expansion_depth, playlist_form_class, 
    seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, elements, token):

    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    print("trigger: {}".format(ctx.triggered))
    sp = get_spotipy(token)
    if trigger['prop_id'] == '.':
        return 'display-none', '', 'display-none', 'button-primary', [], ''
    elif trigger['prop_id'] == 'new-playlist-btn.n_clicks':
        if playlist_clicks == 1:
            return '', 'choosing-seeds', playlist_view_class, new_playlist_btn_class, [], ''
        else:
            return playlist_form_class, seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, "{} songs".format(len(playlist_children))
    elif trigger['prop_id'] == 'initialize-btn.n_clicks':
        if initialize_clicks == 1:
            # initialize playlist
            playlist_children = initialize_playlist(elements, playlist_size, expansion_depth, exclude_tracks, sp)
            return 'display-none', '', '', 'display-none', playlist_children, "{} songs".format(len(playlist_children))

    return playlist_form_class, seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, "{} songs".format(len(playlist_children))



@app.callback(
    Output('confirm-save', 'children'),
    [
        Input('save-playlist-btn', 'n_clicks'),
        Input('playlist-name', 'value')
    ],
    [
        State('playlist', 'children'),
        State('token', 'children')
    ])
def save_playlist(save_playlist_btn_clicks, playlist_name, playlist_tracks, token):
    if save_playlist_btn_clicks == 1:
        sp = get_spotipy(token)
        track_ids = [track['props']['children'][2]['props']['children'] for track in playlist_tracks]
        if len(playlist_name) == 0:
            playlist_name = "Artist Graph"
        playlist = sp.user_playlist_create(username, playlist_name, public=False)
        if len(track_ids) <= 100:
            sp.user_playlist_add_tracks(username, playlist['id'], track_ids)
        else:
            chunks = [track_ids[start:start+100] for start in range (0, len(track_ids), 100)]
            for chunk in chunks:
                sp.user_playlist_add_tracks(username, playlist['id'], chunk)
        return 'Playlist saved!'
    else:
        return ''


        


if __name__ == '__main__':
    app.run_server(debug=True)
