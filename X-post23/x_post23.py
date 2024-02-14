import osmnx as ox

def create_artistic_map(province_name='Istanbul', country_name='Turkey', background_color='#061529'):
    location = f'{province_name}, {country_name}'
    graph = ox.graph_from_place(
        location,
        retain_all=True,
        simplify=True,
        network_type='all'
    )

    edge_data = list(graph.edges(keys=True, data=True))

    road_colors = []
    road_widths = []

    for _, _, _, data in edge_data:
        length = data.get("length", 0)
        if length <= 100:
            linewidth = 0.10
            color = "#a6a6a6"
        elif 100 < length <= 200:
            linewidth = 0.15
            color = "#676767"
        elif 200 < length <= 400:
            linewidth = 0.25
            color = "#454545"
        elif 400 < length <= 800:
            linewidth = 0.35
            color = "#d5d5d5"
        else:
            linewidth = 0.45
            color = "#ededed"

        road_colors.append(color)
        road_widths.append(linewidth)

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        figsize=(27, 40),
        dpi=300,
        bgcolor=background_color,
        save=False,
        edge_color=road_colors,
        edge_linewidth=road_widths,
        edge_alpha=1
    )

    fig.tight_layout(pad=0)
    fig.savefig(
        f'{province_name}_{country_name}_map.png',
        dpi=300,
        bbox_inches='tight',
        format="png",
        facecolor=fig.get_facecolor(),
        transparent=False
    )

create_artistic_map(province_name='Mugla', country_name='Turkey')

# I made some modifications to the original code, which you can find at the link provided below.
# https://towardsdatascience.com/creating-beautiful-maps-with-python-6e1aae54c55c