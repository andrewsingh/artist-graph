import math
import time
from collections import defaultdict
import pprint
import flask
from flask import Flask, redirect, request, session, make_response, render_template
import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import spotipy
import spotipy.util as util
import requests
import os



API_BASE = 'https://accounts.spotify.com'
SPOTIFY_CLIENT_ID = os.environ['SPOTIFY_CLIENT_ID']
SPOTIFY_CLIENT_SECRET = os.environ['SPOTIFY_CLIENT_SECRET']
REDIRECT_URI = 'https://spotify-artist-graph.herokuapp.com/callback'
REDIRECT_URI_LOCAL = 'http://127.0.0.1:8050/callback'
SCOPE = 'user-top-read,playlist-modify-public'
SHOW_DIALOG = True


if int(os.environ['DEBUG']) == 1:
    redirect_uri = REDIRECT_URI_LOCAL
else:
    redirect_uri = REDIRECT_URI


server = flask.Flask(__name__)
server.secret_key = os.environ['SECRET_KEY']


@server.route("/")
def index():
    return render_template('login.html')



# authorization-code-flow Step 1. Have your application request authorization; 
# the user logs in and authorizes access
@server.route("/login")
def verify():
    auth_url = f'{API_BASE}/authorize?client_id={SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}&scope={SCOPE}&show_dialog={SHOW_DIALOG}'
    print("auth_url: {}".format(auth_url))
    return redirect(auth_url)


# authorization-code-flow Step 2.
# Have your application request refresh and access tokens;
# Spotify returns access and refresh tokens
@server.route("/callback")
def api_callback():
    session.clear()
    code = request.args.get('code')

    auth_token_url = f"{API_BASE}/api/token"
    res = requests.post(auth_token_url, data={
        "grant_type":"authorization_code",
        "code":code,
        "redirect_uri": redirect_uri,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
        })

    res_body = res.json()
    print("auth response: {}".format(res.json()))
    session["token"] = res_body.get("access_token")
    return redirect("graph")


app = dash.Dash(
    __name__, 
    server=server,
    routes_pathname_prefix='/graph/',
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width'}]
)


#pp = pprint.PrettyPrinter(indent=4)



def get_spotipy():
    print("getting spotipy, token: {}".format(session['token']))
    return spotipy.Spotify(auth=session['token'])


def get_top_artists(time_range, sp):
    top_artists = sp.current_user_top_artists(limit=50, time_range=time_range)
    top_table = []
    for artist in top_artists['items']:
        top_table.append([artist['name'], artist['popularity'], artist['id']])
    top_df = pd.DataFrame(top_table, columns=['artist', 'pop', 'id'])
    return top_df


# Graph parameters
weight_baseline = 10
size_baseline = 30
size_multiplier = 1
min_rank = 50
default_time_range = 'medium_term'


def get_weight(rank):
    return -rank + min_rank + weight_baseline

def get_size(rank):
    return ((-rank + min_rank) * size_multiplier) + size_baseline


def get_graph_elements(time_range, new_artist_val, sp):
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

    weight_list = list(weights.values())
    weight_pctile = 100 - new_artist_val
    weight_threshold = np.percentile(weight_list, weight_pctile)
    if weight_pctile == 100:
        weight_threshold += 1e-6

    new_artist_count = 0

    for (new_id, new_edges) in new_artist_edges.items():
        if weights[new_id] >= weight_threshold:
            new_artist_count += 1
            nodes.append({'data': {'id': new_id, 'label': artist_names[new_id],
                'size': get_size(min_rank), 'weight': weights[new_id], 'activation': 0, 'selected': False, 'top': False}, 'classes': 'new'})
            edges += new_edges

    for (rank, id) in enumerate(top_ids):
        nodes.append({'data': {'id': id, 'label': artist_names[id], 'size': get_size(rank),
            'weight': weights[id], 'activation': 0, 'selected': False, 'top': True}, 'classes': 'top'})

    # print('new_artist_count: {}'.format(new_artist_count))
    return nodes + edges




def get_search_suggestions(text, sp):
    query = '\'' + text.replace(' ', '+') + '\''
    results = sp.search(query, type='artist')
    return [{'label': artist['name'], 'value': artist['id']} for artist in results['artists']['items']]


def get_html_tracks(tracks):
    return [html.Li(
        children=[
            html.P(className='track-name', children=track['name']),
            html.P(className='track-artists', children=', '.join([artist['name'] for artist in track['artists']])),
            html.P(className='track-id display-none', children=track['id']),
            html.Audio(className='track-audio', src=track['preview_url'], controls='controls')
        ]) for track in tracks]


def get_top_tracks(artist_id, sp):
    return sp.artist_top_tracks(artist_id)['tracks']


def get_all_tracks(artist_id, sp):
    all_tracks = {}
    for album in sp.artist_albums(artist_id)['items']:
        results = sp.tracks([track['id'] for track in sp.album_tracks(album['id'])['items']])
        for track in results['tracks']:
            if not (track['name'] in all_tracks and track['popularity'] <= all_tracks[track['name']]['popularity']):
                all_tracks[track['name']] = track
    track_list = list(all_tracks.values())
    list.sort(track_list, key=lambda track: track['popularity'], reverse=True)
    return track_list


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
    

def filter_tracks(tracks, exclude_set, exclude_remixes):
    if len(exclude_set) >= 1:
        tracks = [track for track in tracks if get_track_tuple(track) not in exclude_set]
    if exclude_remixes:
        tracks = [track for track in tracks if 'remix' not in track['name'].lower()]
    return tracks


def get_track_tuple(track):
    artists_str = ', '.join([artist['name'] for artist in track['artists']])
    return (track['name'], artists_str)


def initialize_playlist(elements, playlist_size, exclude_current_tracks, exclude_remixes, expand_seeds, sp):
    # print('Initializing playlist')
    adjacency_list = {}    
    for elem in elements:
        if is_node(elem):
            adjacency_list[elem['data']['id']] = (elem, [])    
    for elem in elements:
        if not is_node(elem):
            (source, target) = (elem['data']['source'], elem['data']['target'])
            adjacency_list[source][1].append(target)
            adjacency_list[target][1].append(source)
    # print('Built adjacency list')
    # pp.pprint(adjacency_list)
    seeds = [key for (key, value) in adjacency_list.items() if value[0]['data']['activation'] == 1]
    if len(seeds) == 0:
        return []
    neighbors = set()
    current_tracks = set()
    if exclude_current_tracks:
        for playlist in sp.current_user_playlists()['items']:
            if playlist['tracks']['total'] <= 100:
                current_tracks = current_tracks.union(set([get_track_tuple(item['track']) for item in sp.playlist_tracks(playlist['id'])['items']]))
            else:
                for start in range (0, playlist['tracks']['total'], 100):
                    current_tracks = current_tracks.union(set([get_track_tuple(item['track']) for item in sp.playlist_tracks(playlist['id'], offset=start)['items']]))
    if expand_seeds:
        for seed in seeds:
            for neighbor in adjacency_list[seed][1]:
                neighbors.add(neighbor)
    neighbors = list(neighbors)
    songs_per_seed = math.ceil(playlist_size / (len(seeds) + (len(neighbors) / 2)))
    songs_per_neighbor = math.ceil(songs_per_seed / 2)
    playlist = []
    # print('Got seeds and neighbors')
    for seed in seeds:
        # print('Getting tracks for seed {}'.format(seed))
        seed_tracks = []
        if songs_per_seed <= 10:
            seed_tracks = filter_tracks(get_top_tracks(seed, sp), current_tracks, exclude_remixes)
        if len(seed_tracks) < songs_per_seed:
            seed_tracks = filter_tracks(get_all_tracks(seed, sp), current_tracks, exclude_remixes)
        playlist += seed_tracks[:songs_per_seed]
    for neighbor in neighbors:
        # print('Getting tracks for neighbor {}'.format(neighbor))
        neighbor_tracks = []
        if songs_per_neighbor <= 10:
            neighbor_tracks = filter_tracks(get_top_tracks(neighbor, sp), current_tracks, exclude_remixes)
        if len(neighbor_tracks) < songs_per_neighbor:
            neighbor_tracks = filter_tracks(get_all_tracks(neighbor, sp), current_tracks, exclude_remixes)
        playlist += neighbor_tracks[:songs_per_neighbor]
    # print('Generated playlist')
    return get_html_tracks(playlist)



@app.callback(
    Output('artist-search-dropdown', 'options'),
    [
        Input('artist-search-dropdown', 'search_value'),
        Input('artist-search-dropdown', 'value')
    ],
    [State('artist-search-dropdown', 'options')])
def update_search_suggestions(search_value, value, options):
    sp = get_spotipy()
    if search_value:
        return get_search_suggestions(search_value, sp)
    elif value:
        return [option for option in options if option['value'] == value]
    else:
        return options
    


@app.callback(
    Output('artist-tracks-list', 'children'),
    [Input('artist-search-dropdown', 'value')])
def update_artist_tracks(artist_id):
    sp = get_spotipy()
    if artist_id:
        return get_html_tracks(get_top_tracks(artist_id, sp))



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
    ],
    [
        State('artist-graph', 'elements'),
        State('seed-list', 'children'),
        State('artist-search-dropdown', 'value'),
        State('seed-header', 'className'),
    ])
def update_artist_graph(time_range, new_artist_val, node_data, elements, seed_list, artist_search_value, seed_header_class):
    ctx = dash.callback_context
    # print('context: {}'.format(ctx.triggered))
    trigger = ctx.triggered[0]   
    sp = get_spotipy()
    if trigger['value'] == None or trigger['prop_id'] in ['new-artist-threshold-slider.value', 'time-range-dropdown.value']:
        return get_graph_elements(time_range, new_artist_val, sp), seed_list, artist_search_value
    elif trigger['prop_id'] == 'artist-graph.tapNodeData':
        seed = [node for node in elements if node['data']['id'] == node_data['id']][0]
        choosing_seeds = (seed_header_class == 'choosing-seeds')
        # print('choosing_seeds: {}'.format(choosing_seeds))
        if choosing_seeds:
            if len(seed_list) == 1 and isinstance(seed_list[0], str):
                new_elem = html.P(seed['data']['label'], id=seed['data']['label'])
                seed_list = [new_elem]
                seed['data']['activation'] = 1
            elif len([elem for elem in seed_list if elem['props']['children'] == seed['data']['label']]) > 0:
                seed_list = [elem for elem in seed_list if elem['props']['children'] != seed['data']['label']]
                seed['data']['activation'] = 0
            else:
                new_elem = html.P(seed['data']['label'], id=seed['data']['label'])
                seed_list.append(new_elem)
                seed['data']['activation'] = 1
            return elements, seed_list, artist_search_value
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
                return elements, seed_list, seed['data']['id']
            else:
                seed['data']['selected'] = False
                seed['classes'] = old_class(seed)
                for elem in elements:
                    if not is_node(elem) and elem['classes'] == 'selected-edge':
                        elem['classes'] = old_class(elem)
                return elements, seed_list, ''
    else:
        return elements, seed_list, artist_search_value




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
        Input('exclude-remixes', 'value'),
        Input('expand-seeds', 'value')
    ],
    [
        State('playlist-maker', 'className'),
        State('seed-header', 'className'),
        State('playlist-view', 'className'),
        State('new-playlist-btn', 'className'),
        State('playlist', 'children'),
        State('artist-graph', 'elements')
    ])
def toggle_playlist(playlist_clicks, initialize_clicks, playlist_size, exclude_current_tracks, exclude_remixes, expand_seeds, playlist_form_class, seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, elements):

    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    # print('trigger: {}'.format(ctx.triggered))
    sp = get_spotipy()
    if trigger['prop_id'] == '.':
        return 'display-none', '', 'display-none', 'button-primary', [], ''
    elif trigger['prop_id'] == 'new-playlist-btn.n_clicks':
        if playlist_clicks == 1:
            return '', 'choosing-seeds', playlist_view_class, new_playlist_btn_class, [], ''
        else:
            return playlist_form_class, seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, '{} songs'.format(len(playlist_children))
    elif trigger['prop_id'] == 'initialize-btn.n_clicks':
        if initialize_clicks == 1:
            # initialize playlist
            playlist_children = initialize_playlist(elements, playlist_size, exclude_current_tracks, exclude_remixes, expand_seeds, sp)
            return 'display-none', '', '', 'display-none', playlist_children, '{} songs'.format(len(playlist_children))

    return playlist_form_class, seed_header_class, playlist_view_class, new_playlist_btn_class, playlist_children, '{} songs'.format(len(playlist_children))



@app.callback(
    Output('confirm-save', 'children'),
    [
        Input('save-playlist-btn', 'n_clicks'),
        Input('playlist-name', 'value')
    ],
    [State('playlist', 'children')])
def save_playlist(save_playlist_btn_clicks, playlist_name, playlist_tracks):
    if save_playlist_btn_clicks == 1:
        sp = get_spotipy()
        track_ids = [track['props']['children'][2]['props']['children'] for track in playlist_tracks]
        if len(playlist_name) == 0:
            playlist_name = 'Artist Graph'
        username = sp.current_user()['id']
        playlist = sp.user_playlist_create(username, playlist_name)
        if len(track_ids) <= 100:
            sp.user_playlist_add_tracks(username, playlist['id'], track_ids)
        else:
            chunks = [track_ids[start:start+100] for start in range (0, len(track_ids), 100)]
            for chunk in chunks:
                sp.user_playlist_add_tracks(username, playlist['id'], chunk)
        return 'Playlist saved!'
    else:
        return ''







prototype1 = {
    'name': 'cose',
    'animate': False,
    'randomize': False, 
    'edgeElasticity': 30, 
    'nodeRepulsion': 20000,
    'nodeOverlap': 500,
    'gravity': 40,
    'componentSpacing': 200,
    'nodeDimensionsIncludeLabels': True,
    'numIter': 1000,
    'initialTemp': 1000,
    'coolingFactor': 0.99,
    'minTemp': 1.0
}



graph = cyto.Cytoscape(
    id='artist-graph',
    layout=prototype1,
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
                # 'text-outline-width': '2px',
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
        {
            'selector': '.selected-node',
            'style': {
                'background-color': '#f00000',
                # 'text-outline-color': '#f00000'
            }
        },
        {
            'selector': '.top',
            'style': {
                'background-color': '#555',
                # 'text-outline-color': '#555'
            }
        },
        {
            'selector': '.new',
            'style': {
                'background-color': 'green',
                # 'text-outline-color': 'green'
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
                    id='div-intro',
                    children=[
                        # html.P(id='subtitle', children='Your top artists'),
                        html.Ul(id='intro-list', 
                            children=[
                                html.Li('Each node is an artist, each edge joins two similar artists'),
                                html.Li('The larger a node, the more you listen to that artist'),
                                html.Li(children=[
                                    'Artists outside your top 50 are ',
                                    html.Span(className='green', children='green')
                                ])
                            ]
                        )
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
                        html.Label(id='new-artist-threshold-label', htmlFor='new-artist-threshold-slider', children='Percent of new artists in graph'),
                        dcc.Slider(
                            id='new-artist-threshold-slider',
                            min=0,
                            max=100,
                            step=1,
                            value=5,
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
                                dcc.Checklist(
                                    id='exclude-playlist-tracks',
                                    options=[
                                        {'label': 'Exclude tracks in my current playlists', 'value': 1}
                                    ]
                                ),
                                dcc.Checklist(
                                    id='exclude-remixes',
                                    options=[
                                        {'label': 'Exclude remixes', 'value': 1}
                                    ]
                                ),
                                dcc.Checklist(
                                    id='expand-seeds',
                                    options=[
                                        {'label': 'Include seed neighbors', 'value': 1}
                                    ],
                                    value=[1]
                                ),
                                html.H6('Seeds', id='seed-header', className=''),
                                html.Div(id='seed-list', children=[
                                    'Select nodes in the graph to seed the playlist. When you have selected your seeds, click the initialize button below.'
                                ]),
                                html.Button('Initialize', id='initialize-btn', className='button-primary'),
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





        

if __name__ == '__main__':
    app.run_server(debug=True)
