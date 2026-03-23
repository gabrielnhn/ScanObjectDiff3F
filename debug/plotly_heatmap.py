import torch
import numpy as np
import trimesh
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update

# --- CONFIG ---
source_ply = "pointcloud1_with_features.ply"
source_pt = "pointcloud1_with_features.pt"
target_ply = "pointcloud2_with_features.ply"
target_pt = "pointcloud2_with_features.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading Data...")
# Load Geometry
verts_source = trimesh.load(source_ply).vertices
verts_target = trimesh.load(target_ply).vertices

# Load Features (Applying the float32 and eps=1e-6 fixes from earlier!)
f_source = torch.load(source_pt, map_location=device).squeeze().float()
f_target = torch.load(target_pt, map_location=device).squeeze().float()
f_source = torch.nn.functional.normalize(f_source, dim=-1, eps=1e-6)
f_target = torch.nn.functional.normalize(f_target, dim=-1, eps=1e-6)

# --- PLOTLY SETUP ---
def create_base_figure(verts, title, color='lightgrey', colorscale=None):
    # Plotly handles colorscales natively, which is super convenient
    marker_dict = dict(size=3, color=color, opacity=0.8)
    if colorscale:
        marker_dict['colorscale'] = colorscale
        marker_dict['cmin'] = 0.0
        marker_dict['cmax'] = 1.0

    fig = go.Figure(data=[go.Scatter3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        mode='markers',
        marker=marker_dict
    )])
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode='data') # Keeps the 3D aspect ratio realistic
    )
    return fig

# Initialize the figures
fig_source = create_base_figure(verts_source, "Source Object (Click a point here!)")
fig_target = create_base_figure(verts_target, "Target Object (Heatmap appears here)")

# --- DASH APP ---
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive DINO Feature Heatmap", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    html.Div([
        # Source Graph (Listens for clicks)
        dcc.Graph(id='source-graph', figure=fig_source, style={'width': '50vw', 'height': '80vh'}),
        # Target Graph (Updates based on clicks)
        dcc.Graph(id='target-graph', figure=fig_target, style={'width': '50vw', 'height': '80vh'})
    ], style={'display': 'flex'})
])

@app.callback(
    Output('target-graph', 'figure'),
    Input('source-graph', 'clickData')
)
def update_heatmap(clickData):
    if clickData is None:
        return no_update # Do nothing if no click yet

    # 1. Get the exact vertex index the user clicked
    query_idx = clickData['points'][0]['pointNumber']
    print(f"Clicked Vertex {query_idx}")

    # 2. Extract feature and calculate similarity
    query_feat = f_source[query_idx].unsqueeze(0)
    sim = torch.mm(f_target, query_feat.T).squeeze().cpu().numpy()

    # 3. Min-Max Normalize the similarity for the heatmap
    sim_range = sim.max() - sim.min()
    if sim_range > 0:
        sim = (sim - sim.min()) / sim_range
    else:
        sim = np.zeros_like(sim)

    # 4. Update the Target figure with the new colors
    # We use Plotly's built-in 'jet' colorscale
    updated_fig = create_base_figure(verts_target, f"Target Heatmap (Similarity to Pt {query_idx})", color=sim, colorscale='jet')
    
    return updated_fig

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Server starting! Click the http://127.0.0.1:8050 link below to open.")
    print("="*50 + "\n")
    app.run(debug=True, use_reloader=False)