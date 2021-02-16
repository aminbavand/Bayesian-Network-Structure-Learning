
import argparse
import logging
import networkx
import json
import names
import random


def generate_random_dag(n, p):
    logging.debug(
        'Generate a random graph of %s nodes and an edge probability of %s',
        args.nodes,
        args.probability
    )
    random_graph = networkx.fast_gnp_random_graph(n, p, directed=True)
    random_dag = networkx.DiGraph(
        [
            (u, v) for (u, v) in random_graph.edges() if u < v
        ]
    )
    return random_dag


def create_random_node(id, female):
    return {
        'children': [],
        'data': {
            'bars': {
                'precision': '{:.2f}'.format(random.random()),
                'recall': '{:.2f}'.format(random.random())
            },
            'name': '{id}. {name}'.format(
                id=id,
                name=names.get_first_name(
                    gender='female' if female else 'male'
                )
            )
        },
        'id': str(id)
    }


def graph_to_json(graph, args):
    logging.debug(
        'Generate a random graph of %s nodes and an edge probability of %s',
        args.nodes,
        args.probability
    )

    nodes = {}

    # Root node
    nodes['-1'] = create_random_node('-1', args.female)

    # Create node objects with random bar data
    for node_id in graph.nodes_iter():
        nodes[str(node_id)] = create_random_node(node_id, args.female)

    # Connect the root node with a random number of other nodes.
    root_num_children = int(round(random.random() * args.nodes / 10))
    root_num_children = 1 if root_num_children == 0 else root_num_children
    for node_id in range(root_num_children):
        nodes['-1']['children'].append(str(node_id))

    # Add parent-child relationships
    for edge in graph.edges_iter():
        nodes[str(edge[0])]['children'].append(str(edge[1]))

    graph_json = json.dumps(nodes)

    if args.output:
        # Write graph to file and overrite existing content
        with open(args.output, 'w+') as f:
            # Write graph
            f.write(graph_json)
    else:
        print(graph_json)


def main(args, loglevel):
    graph_to_json(
        generate_random_dag(args.nodes, args.probability),
        args
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a random graph and save as a JSON file.'
    )

    parser.add_argument(
        '-n',
        '--nodes',
        default=30,
        dest='nodes',
        type=int,
        help='Number of nodes to be generated.'
    )

    parser.add_argument(
        '-p',
        '--probability',
        default=0.5,
        dest='probability',
        type=float,
        help='Probability of edge creation.'
    )

    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        type=str,
        help='Specify output. If `None` print to console.'
    )

    parser.add_argument(
        '-f',
        '--female',
        dest='female',
        action='store_true',
        help='Use female random first names.'
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help='Increase output verbosity.',
        action='store_true'
    )
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

main(args, loglevel)