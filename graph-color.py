import dimod
import itertools
import minorminer
import json
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from dwave.system.samplers import DWaveSampler
from dwave.system import LeapHybridSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
import networkx as nx
import matplotlib.pyplot as plt
import pulp

# colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "magenta", "brown", "teal", "lime", "indigo", "violet", "maroon", "turquoise", "olive", "navy", "beige", "gray"]
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#8c564b", "#c49c94", "#ff9896", "#9467bd", "#f7b6d2",
    "#dbdb8d", "#17becf", "#ffbb78", "#1f77b4", "#aec7e8",
    "#98df8a", "#2ca02c", "#9edae5", "#ff7f0e", "#7f7f7f",
    "#c5b0d5", "#d62728", "#c7c7c7", "#bcbd22", "#8c564b",
    "#e377c2", "#ff9896", "#9467bd", "#f7b6d2", "#dbdb8d",
    "#17becf", "#ffbb78", "#1f77b4", "#aec7e8", "#98df8a"
]

def build_bqm(graph):
    bqm = BinaryQuadraticModel({}, {}, 0, 'BINARY')

    for node in graph.nodes():
        bqm.add_variable(node, -1)

    for edge in graph.edges():
        bqm.add_interaction(edge[0], edge[1], 2)

    return bqm

embedding_file = "embedding.json"

def get_embed_size(bqm, outputp):
    best_emb = None
    TT = 0
    token = 'DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'
    child = DWaveSampler(token=token)
    source_edgelist = list(itertools.chain(bqm.quadratic, ((int(v), int(v)) for v in bqm.linear)))

    stderr = open(outputp.replace('.json', '_log.txt'), 'w')
    target_edgelist = dimod.child_structure_dfs(child).edgelist
    min_val = 1000000000
    min_len = 100000

    while TT < 50:
        TT += 1
        embedding = minorminer.find_embedding(source_edgelist, target_edgelist, verbose=2, max_no_improvement=20,
                                              random_seed=TT,
                                              chainlength_patience=20)
        var1 = len(embedding.keys())
        if var1 == 0:
            print("Failed ", TT, file=stderr)
            continue
        len_emb = max(map(len, embedding.values()))
        var = sum(len(embedding[node]) for node in embedding)
        print(len_emb, var, file=stderr)
        if len_emb < min_len or (len_emb == min_len and min_val > var):
            min_len = len_emb
            min_val = var
            best_emb = embedding
    print(min_len, min_val, file=stderr)
    # with open(outputp, "w") as f:
    #     json.dump(best_emb, f)
    
    # # Close stderr
    # stderr.close()
    
    # return best_emb
    from dwave.embedding import EmbeddedStructure
    pp = EmbeddedStructure(target_edgelist, best_emb)
    import json
    with open(outputp, "w") as f:
        json.dump(pp, f)
    return pp

def maximum_independent_set(graph):
    bqm = build_bqm(graph)
    
    sampler = SimulatedAnnealingSampler()
    # sampler = LeapHybridSampler()
    # sampler = EmbeddingComposite(DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'))

    # if not os.path.exists(embedding_file) or os.path.getsize(embedding_file) == 0:
    #     embedding = get_embed_size(bqm, embedding_file)
    # else:
    #     with open(embedding_file, "r") as fff:
    #         embbed = json.load(fff)
    #     embbed = {int(k): v for k, v in embbed.items()}


    # sampler = FixedEmbeddingComposite(DWaveSampler(token='DEV-e0ac368d04813d5d0a2019a61e39c30c446c6397'), embedding=embedding)

    response = sampler.sample(bqm, num_reads=1000)
    
    best_solution = next(response.samples())
    independent_set = [node for node in best_solution if best_solution[node] == 1]
    
    return independent_set

def ilp_graph_coloring(graph):
    problem = pulp.LpProblem("GraphColoring", pulp.LpMinimize)

    # variables
    colors = range(1, len(graph) + 1)  
    vertices = list(graph.nodes())  

    x = pulp.LpVariable.dicts("x", (colors, vertices), 0, 1, pulp.LpInteger)
    w = pulp.LpVariable.dicts("w", (colors), 0, 1, pulp.LpInteger)

    # objective function
    problem += pulp.lpSum(w[i] for i in colors)

    # constraints
    for v in vertices:
        problem += pulp.lpSum(x[i][v] for i in colors) == 1  # Each vertex must have exactly one color
        for i in colors:
            problem += x[i][v] <= w[i]  # If a color is used for a vertex, set w[i] = 1

    for u in vertices:
        for v in graph[u]:
            for i in colors:
                problem += x[i][u] + x[i][v] <= 1  # Adjacent vertices cannot have the same color

    problem.solve()

    # solution
    coloring = {}
    for v in vertices:
        for i in colors:
            if pulp.value(x[i][v]) == 1:
                coloring[v] = i
                break

    return coloring

def count_colors(coloring):
    unique_colors = set(coloring.values())
    return len(unique_colors)

def is_valid_coloring(graph, coloring):
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            return False
    return True

def new_graph(graph, independent_set):
    modified_graph = graph.copy()
    modified_graph.remove_nodes_from(independent_set) 
    return modified_graph

# edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6,1), (6,2), (6,3), (6,4), (6, 5)]
# graph = nx.Graph(edges)
graph = nx.gnm_random_graph(10, 90)
input_graph = graph.copy()

independent_sets = []
while graph.number_of_nodes() != 0:
    current_independent_set = maximum_independent_set(graph)
    independent_sets.append(current_independent_set)
    graph = new_graph(graph, current_independent_set)

coloring = ilp_graph_coloring(input_graph)
num_independent_sets = len(independent_sets)
print("Min number of colors with qubo: ", len(independent_sets))

print("ILP min number of colors: ", coloring)
print("num of ILP min color", count_colors(coloring))

if (num_independent_sets == count_colors(coloring)):
    print ("Valid result")

plt.figure(figsize=(12, 6))

# Qubo graph
plt.subplot(1, 2, 1)
pos = nx.circular_layout(input_graph)
for i, independent_set in enumerate(independent_sets):
    nx.draw(input_graph.subgraph(independent_set), pos=pos, node_color=colors[i], label=f"Independent Set {i + 1}")
nx.draw(input_graph, pos=pos, node_color="k", node_size=100)
plt.title('Maximum Independent Sets')
# plt.legend()
plt.text(0, -1.2, f'Chromatic number: {len(independent_sets)}', horizontalalignment='center')

# ILP graph
plt.subplot(1, 2, 2)
for node, color in coloring.items():
    nx.draw_networkx_nodes(input_graph, pos, nodelist=[node], node_color=colors[color - 1], node_size=100)
node_colors = [colors[coloring[node] - 1] for node in input_graph.nodes()]
nx.draw(input_graph, pos=pos, with_labels=False, node_color=node_colors, node_size=300)
nx.draw(input_graph, pos=pos, with_labels=False, node_color="k", node_size=100)
plt.title('ILP Graph Coloring')
plt.text(0, -1.2, f'Chromatic number: {count_colors(coloring)}', horizontalalignment='center')
if is_valid_coloring(input_graph, coloring):
    plt.text(0, -1.3, "Valid result", horizontalalignment='center')
else:
    plt.text(0, -1.3, "Invalid result", horizontalalignment='center')

plt.tight_layout()
plt.show()