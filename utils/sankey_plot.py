# -*- coding: utf-8 -*-

import plotly.graph_objects as go

def create_sankey_plot():
    """
    Generates a Sankey plot using Plotly in Python.

    The plot visualizes the flow of tasks through an annotation process,
    including different question types and annotators. Node positioning
    is now handled automatically by Plotly's layout algorithm.
    QTA is for open-ended, and QTB is for scoring (original mapping).
    """

    open_questions_1 = 37 * 2 # Represents tasks in E1 for QTB
    open_questions_2 = 15 * 2 # Represents tasks in E2 for QTB
    annotators = 9
    pairwise = 10 # Represents tasks for pairwise in E1 & E2
    scoring_1 = 37 * 5 # Represents tasks in E1 for QTA
    scoring_2 = 15 * 5 # Represents tasks in E2 for QTA
    pairwise_completed = 86
    scoring_completed = 330 - 4*5
    open_questions_completed = 123 - 4*2

    # Calculate total questions
    total_questions = (open_questions_1 * annotators) + \
                      (pairwise * annotators) + \
                      (scoring_1 * annotators)

    nodes_labels = [
        f"Questions (N={total_questions})",
        f"E1 - QTB open-ended (N={open_questions_1 * annotators})",
        f"E1 - QTA scoring (N={scoring_1 * annotators})",
        f"E1 - QTC pairwise (N={pairwise * annotators})",
        f"E2 - QTB open-ended (N={open_questions_2 * annotators})",
        f"E2 - QTA scoring (N={scoring_2 * annotators})",
        f"E2 - QTC pairwise (N={pairwise * annotators})",
        "EA 1", "EA 2", "EA 3", "EA 4", "EA 5", "EA 6", "EA 7", "EA 8", "EA 9",
        f"C QTB open-ended (N={open_questions_completed})",
        f"C QTA scoring (N={scoring_completed})",
        f"C QTC pairwise (N={pairwise_completed})"
    ]

    # Node colors from the "nicer colors" version, mapped to original QTA/QTB definitions.
    node_colors = [
        '#FFCC33',  # Goldenrod - Questions

        '#33CCFF',  # Sky Blue - E1 QTB
        '#FF338D',  # Magenta - E1 QTA
        '#33FF8D',  # Seafoam Green - E1 QTC

        '#3357FF',  # Blue-Violet - E2 QTB
        '#8D33FF',  # Purple - E2 QTA
        '#00c04b',  # Green - E2 QTC

        # Annotator Colors (can be diversified if needed)
        '#D21404', '#FF5733', '#fee12b', '#32cd32',
        '#33CCFF', '#338DFF', '#3357FF', '#AF69EF', '#821544',

        # Completed Task Colors
        '#338DFF',  # Royal Blue
        '#FF3357',  # Red-Pink
        '#00ab41',  # Forest Green
    ]

    # Map node names to their indices for the 'source' and 'target' arrays.
    node_to_index = {name: i for i, name in enumerate(nodes_labels)}

    links_data = [
        # Initial task distribution
        {"source": f"Questions (N={total_questions})", "target": f"E1 - QTB open-ended (N={open_questions_1 * annotators})", "value": open_questions_1 * annotators},
        {"source": f"Questions (N={total_questions})", "target": f"E1 - QTA scoring (N={scoring_1 * annotators})", "value": scoring_1 * annotators},
        {"source": f"Questions (N={total_questions})", "target": f"E1 - QTC pairwise (N={pairwise * annotators})", "value": pairwise * annotators},

        # Flow from E1 to E2 (using the _2 variables for E2 totals)
        {"source": f"E1 - QTB open-ended (N={open_questions_1 * annotators})", "target": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "value": open_questions_2 * annotators},
        {"source": f"E1 - QTA scoring (N={scoring_1 * annotators})", "target": f"E2 - QTA scoring (N={scoring_2 * annotators})", "value": scoring_2 * annotators},
        {"source": f"E1 - QTC pairwise (N={pairwise * annotators})", "target": f"E2 - QTC pairwise (N={pairwise * annotators})", "value": pairwise * annotators},

        # E2 Question types to Annotators (Distribute E2 tasks among annotators)
        # Assuming equal distribution for simplicity for now, adjust values if a specific distribution is known
        # For demonstration, distributing E2 tasks to a subset of annotators as in original.
        # Ensure the sum of values from E2 to EAs equals the E2 total for each type.
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 1", "value": open_questions_2}, # open_questions_2 is per annotator
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 2", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 3", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 4", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 5", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 6", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 7", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 8", "value": open_questions_2},
        {"source": f"E2 - QTB open-ended (N={open_questions_2 * annotators})", "target": "EA 9", "value": open_questions_2},

        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 1", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 2", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 3", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 4", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 5", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 6", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 7", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 8", "value": scoring_2},
        {"source": f"E2 - QTA scoring (N={scoring_2 * annotators})", "target": "EA 9", "value": scoring_2},

        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 1", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 2", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 3", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 4", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 5", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 6", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 7", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 8", "value": pairwise},
        {"source": f"E2 - QTC pairwise (N={pairwise * annotators})", "target": "EA 9", "value": pairwise},

        # Annotators to Completed Question Types (using the 'completed' variables)
        # The sum of these values from all EAs should equal the total completed for each type.
        # These values need to be carefully set to match the total completed.
        # For simplicity, assuming distribution based on the total completed and annotators.
        # You'll need to refine these values based on actual distribution per annotator.
        {"source": "EA 1", "target": f"C QTB open-ended (N={open_questions_completed})", "value": open_questions_2},
        {"source": "EA 2", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 11*2},
        {"source": "EA 3", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 11*2},
        {"source": "EA 4", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 4*2},
        {"source": "EA 5", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 4*2},
        {"source": "EA 6", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 4*2},
        {"source": "EA 7", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 4*2},
        {"source": "EA 8", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 4*2},
        {"source": "EA 9", "target": f"C QTB open-ended (N={open_questions_completed})", "value": 7},

        {"source": "EA 1", "target": f"C QTA scoring (N={scoring_completed})", "value": scoring_2},
        {"source": "EA 2", "target": f"C QTA scoring (N={scoring_completed})", "value": 11*5},
        {"source": "EA 3", "target": f"C QTA scoring (N={scoring_completed})", "value": 11*5},
        {"source": "EA 4", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},
        {"source": "EA 5", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},
        {"source": "EA 6", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},
        {"source": "EA 7", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},
        {"source": "EA 8", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},
        {"source": "EA 9", "target": f"C QTA scoring (N={scoring_completed})", "value": 4*5},

        {"source": "EA 1", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 2", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 3", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 4", "target": f"C QTC pairwise (N={pairwise_completed})", "value": 6},
        {"source": "EA 5", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 6", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 7", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 8", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
        {"source": "EA 9", "target": f"C QTC pairwise (N={pairwise_completed})", "value": pairwise},
    ]

    # Filter out links with zero value to avoid errors in Sankey diagram
    links_data = [link for link in links_data if link["value"] > 0]

    sources = [node_to_index[link["source"]] for link in links_data]
    targets = [node_to_index[link["target"]] for link in links_data]
    values = [link["value"] for link in links_data]

    link_colors = []
    for link in links_data:
        source_node_name = link["source"]
        source_node_index = node_to_index[source_node_name]
        base_color = node_colors[source_node_index]
        # Convert hex color to RGBA for transparency
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        link_colors.append(f'rgba({r},{g},{b},0.4)') # 0.4 opacity for links

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        #title_text="Annotation Process Flow",
        font_size=12,
        height=700,
        width=1200,
    )

    fig.show()
    return fig

if __name__ == "__main__":
    fig = create_sankey_plot() # Ensure your function returns the figure object
    fig.write_html("sankey_plot.html")
    print("Plot saved as interactive HTML: sankey_plot.html")

