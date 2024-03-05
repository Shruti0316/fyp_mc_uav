import os
import torch
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import load_model
from utils.functions import load_problem
from utils.data_utils import set_seed, str2bool
from nets.attention_model import AttentionModel


def arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=5, help='Random seed to use')

    # Model
    parser.add_argument('--num_agents', type=int, default=2, help="Number of agents")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")  # 2 depots is only supported for Attention on OP
    parser.add_argument('--load_path', help='Path to load model. Just indicate the directory where epochs are saved or'
                                            'the directory + the specific epoch you want to load')
    parser.add_argument('--baseline', default=None, help="If not None, it will execute the given baseline for the"
                                                         "specified problem instead of the loaded model")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # Problem
    parser.add_argument('--problem', default='op', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--data_dist', type=str, default='coop',
                        help='Data distribution to use during training. Options: coop, nocoop, const, dist, unif')
    parser.add_argument('--test_coop', type=str2bool, default=True,
                        help="For the OP with coop/nocoop distribution, set test_coop=True to see the multi-agent plot")

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    # Check problem is correct
    assert opts.problem in ('op'), 'Supported problems is OP'
    # TODO: add VRP and PCTSP
    assert opts.num_agents > 0, 'num_agents must be greater than 0'

    # Check baseline is correct for the given problem
    if opts.baseline is not None:
        if opts.problem == 'op':  # TODO: add TSP, VRP and PCTSP baselines
            assert opts.baseline in ('tsili', 'tsiligreedy', 'gurobi', 'gurobigap', 'gurobit', 'opga', 'ortools',
                                     'compass'),\
                'Supported baselines for OP are tsili, gurobi, opga, ortools and compass'
    return opts


def assign_colors(n):
    color = {k: [] for k in 'rgb'}
    for i in range(n):
        temp = {k: random.randint(0, 230) for k in 'rgb'}
        for k in temp:
            while 1:
                c = temp[k]
                t = set(j for j in range(c - 25, c + 25) if 0 <= j <= 230)
                if t.intersection(color[k]):
                    temp[k] = random.randint(0, 230)
                else:
                    break
            color[k].append(temp[k])
    return [(color['r'][i] / 256, color['g'][i] / 256, color['b'][i] / 256) for i in range(n)]


def baselines(baseline, problem, dataset, device):

    # Prepare inputs
    inputs = dataset.data[0]
    if not (baseline == 'tsili' or baseline == 'tsiligreedy'):
        for k, v in inputs.items():
            inputs[k] = v.detach().numpy().tolist()

    for k, v in inputs.items():
        inputs[k] = np.array(v)

    return np.array(tour).squeeze(), inputs, model_name


def plot_tour(tour, inputs, problem, model_name, data_dist='', num_depots=1):
    """
    Plot a given tour.
    # Arguments
        tour (numpy array): ordered list of nodes.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
        num_depots: number of depots. Options are 1 or 2.
    """

    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Data
    depot = inputs['depot']
    if num_depots > 1:
        depot2 = inputs['depot2']
        ax.scatter(depot2[0], depot2[1], depot2[2], c='r')  # z-dimension added
    loc = inputs['loc']

    # Plot nodes (black circles) and depot (red circle)
    ax.scatter(depot[0], depot[1], depot[2], c='b')  # z-dimension added
    ax.scatter(loc[..., 0], loc[..., 1], loc[..., 2], c='k')  # z-dimension added

    # If tour starts and ends in depot
    if len(tour.shape) == 0:
        # Set title
        title = problem.upper()
        title += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else title
        title += ': Length = 0'
        if problem == 'op':
            # Add OP rewards to the title (if problem is OP)
            prize = inputs['prize']
            title += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(inputs['max_length'], 0, np.sum(prize[prize>0]))
        ax.set_title(title)
        plt.show()
        return

    # Calculate the length of the tour
    loc = np.concatenate(([depot], loc), axis=0)
    if num_depots > 1:
        loc = np.concatenate((loc, [depot2]), axis=0)
    nodes = np.take(loc, tour, axis=0)
    d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)) + np.linalg.norm(nodes[0] - depot)

    # Set title
    title = problem.upper()
    title += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    title += ' - {:s}: Length = {:.4g}'.format(model_name, d)
    if problem == 'op':
        # Add OP prize to the title (if problem is OP)
        prize = inputs['prize']
        reward = np.sum(np.take(prize, tour[:-1] - 1))
        title += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(inputs['max_length'], reward, np.sum(prize[prize>0]))
    ax.set_title(title)

    # Add the start depot at the start of the tour
    tour = np.insert(tour, 0, 0, axis=0)

    # Draw arrows
    for i in range(1, tour.shape[0]):
        dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
        dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
        dz = loc[tour[i], 2] - loc[tour[i - 1], 2]  # z-dimension added
        ax.quiver(loc[tour[i - 1], 0], loc[tour[i - 1], 1], loc[tour[i - 1], 2], dx, dy, dz, color='g')
    plt.show()


def plot_multitour(num_agents, tours, inputs, problem, model_name, data_dist='', num_depots=1):
    """
    Plot the tours of all the agents.
    # Arguments
        num_agents: number of agents.
        tours (numpy array): ordered list of nodes.
        inputs (dict or numpy array): if TSP, inputs is an array containing the coordinates of the nodes. Otherwise, it
        is a dict with the coordinates of the nodes (loc) and the depot (depot), and other possible features.
        problem (str): name of the problem.
        model_name (str): name of the model.
        data_dist (str): type of prizes for the OP. For any other problem, just set this to ''.
    """

    data_dist_dir = '' if data_dist == '' else '_{}'.format(data_dist)
    image_dir = 'images/{}_{}'.format(model_name.lower(), problem.lower()) + data_dist_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Initialize global plots
    fig1 = plt.figure(num_agents)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlim([-.05, 1.05])
    ax1.set_ylim([-.05, 1.05])
    ax1.set_zlim([-.05, 1.05])

    fig2 = plt.figure(num_agents + 1)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlim([-.05, 1.05])
    ax2.set_ylim([-.05, 1.05])
    ax2.set_zlim([-.05, 1.05])

    # Assign a color to each agent
    colors = assign_colors(num_agents + 2)
    color_shared = np.array(colors[-2]).reshape(1, -1)
    color_depot = np.array(colors[-1]).reshape(1, -1)

    length_sum, prize_sum, prize_max = 0, 0, 0
    for agent in range(num_agents):
        tour = tours[agent]
        tour = tour if tour.size > 1 else np.array([tour])
        color = np.array(colors[agent]).reshape(1, -1)

        # Initialize individual plots
        fig = plt.figure(agent)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-.05, 1.05])
        ax.set_ylim([-.05, 1.05])
        ax.set_zlim([-.05, 1.05])

        # Data
        loc = inputs[agent]['loc']
        prize = inputs[agent]['prize']
        depot = inputs[agent]['depot']
        if num_depots > 1:
            depot2 = inputs[agent]['depot2']
        max_length = inputs[agent]['max_length']

        # Plot region assignment
        ax2.scatter(loc[prize == 1][..., 0], loc[prize == 1][..., 1], loc[prize == 1][..., 2], c=color, label='Agent {}'.format(agent))
        ax2.scatter(depot[0], depot[1], depot[2], s=200, c=color_depot, marker='^', label='Depot' if agent == 0 else '')
        if num_depots > 1:
            ax2.scatter(depot2[0], depot2[1], depot2[2], s=200, c=color_depot, marker='v', label='Depot' if agent == 0 else '')

        # Plot regions
        ax1.scatter(loc[prize == 1][..., 0], loc[prize == 1][..., 1], loc[prize == 1][..., 2], c=color, label='Agent {}'.format(agent))
        ax1.scatter(depot[0], depot[1], depot[2], s=200, c=color_depot, marker='^', label='Depot' if agent == 0 else '')
        if num_depots > 1:
            ax1.scatter(depot2[0], depot2[1], depot2[2], s=200, c=color_depot, marker='v', label='Depot' if agent == 0 else '')
        if agent == 0:
            for l in range(len(loc)):
                ax1.text(loc[l, 0] + .005, loc[l, 1] + .005, loc[l, 2] + .005, str(l + 1))
        
        ax.scatter(depot[0], depot[1], s=200, c=color_depot, marker='^', label='Depot')
        if num_depots > 1:
            ax.scatter(depot2[0], depot2[1], s=200, c=color_depot, marker='v', label='Depot')
        
        non_unit_non_obstacle_prize_indices = (prize !=1) & (prize>0)
        unit_non_obstacle_prize_indices = (prize == 1) & (prize>0)
        ax.scatter(loc[unit_non_obstacle_prize_indices][..., 0], loc[unit_non_obstacle_prize_indices][..., 1], loc[unit_non_obstacle_prize_indices][..., 2], c=color, label='Initial')
        ax.scatter(loc[non_unit_non_obstacle_prize_indices][..., 0], loc[non_unit_non_obstacle_prize_indices][..., 1], loc[non_unit_non_obstacle_prize_indices][..., 2], c=color_shared, label='Shared')
        ax.scatter(loc[prize < 0][..., 0], loc[prize<0][..., 1], loc[prize<0][..., 2], c='red', label='Obstacle')

        for l in range(len(loc)):
            ax.text(loc[l, 0] + .005, loc[l, 1] + .005, loc[l, 2] + .005, str(l + 1))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.9))
        # ax.tight_layout(rect=[0, 0, 1, 0.95])

        # Calculate the length of the tour
        loc = np.concatenate(([depot], loc), axis=0)
        if num_depots > 1:
            loc = np.concatenate((loc, [depot2]), axis=0)
        nodes = np.take(loc, tour, axis=0)
        d = np.sum(np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)) + np.linalg.norm(nodes[-1] - nodes[0])
        length_sum += d

        # Agent information
        info = problem.upper()
        info += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else info
        info += ' - {:s} {}: Length = {:.4g}'.format(model_name, agent, d)

        # Add OP prize to the title
        reward = np.sum(np.take(prize, tour[:-1] - 1))
        prize_sum += reward
        prize_max += np.sum(prize[prize>0])
        info += ' / {:.4g} | Prize = {:.4g} / {:.4g}'.format(max_length, reward, np.sum(prize[prize>0]))
        plt.title(info)

        # Add the start depot and the end depot to the tour
        if tour[0] != 0:
            tour = np.insert(tour, 0, 0, axis=0)
        elif tour[-1] != loc.shape[0] - 1:
            tour = np.insert(tour, len(tour), 0, axis=0)
        print('Agent {}: '.format(agent), tour)

        # Draw arrows
        for i in range(1, tour.shape[0]):
            dx = loc[tour[i], 0] - loc[tour[i - 1], 0]
            dy = loc[tour[i], 1] - loc[tour[i - 1], 1]
            dz = loc[tour[i], 2] - loc[tour[i - 1], 2]
            # plt.figure(agent)
            ax.quiver(loc[tour[i - 1], 0], loc[tour[i - 1], 1], loc[tour[i - 1], 2], dx, dy, dz, color=color, length=0.1)
            ax1.quiver(loc[tour[i - 1], 0], loc[tour[i - 1], 1], loc[tour[i - 1], 2], dx, dy, dz, color=color, length=0.1)
            # ax.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], loc[tour[i - 1], 2], dx, dy, dz, head_width=.025, fc=color, ec=color,
                    #   length_includes_head=True, alpha=0.5)
            # ax.figure(num_agents)
            # ax1.arrow(loc[tour[i - 1], 0], loc[tour[i - 1], 1], loc[tour[i - 1], 2], dx, dy, dz, head_width=.025, fc=color, ec=color,
                    #   length_includes_head=True, alpha=0.5)
        fig.savefig(image_dir + '/agent_{}.png'.format(agent), dpi=150)

    # Plot region assignment
    # plt.figure(num_agents + 1)
    ax2.set_title('Region assignment')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    fig2.tight_layout(rect=[0.05, 0, 1, 1])
    fig2.savefig(image_dir + '/assignment.png', dpi=150)

    # Agents information
    info = problem.upper()
    info += ' (' + data_dist.lower() + ')' if len(data_dist) > 0 else ''
    info += ' - {:s}: Av. Length = {:.3g}'.format(model_name, length_sum / num_agents)
    info += ' / {:.3g} | Total Prize = {:.3g} / {:.3g}'.format(inputs[0]['max_length'], prize_sum, np.sum(prize_max))
    # plt.figure(num_agents)
    ax1.set_title(info)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    fig1.tight_layout(rect=[0.05, 0, 1, 1])
    fig1.savefig(image_dir + '/solution.png', dpi=150)
    plt.show()


def main(opts):

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Set the device
    device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=1, distribution=opts.data_dist,
                                   test_coop=opts.test_coop, num_agents=opts.num_agents, num_depots=opts.num_depots)
    inputs = dataset.data[0]

    # Apply baseline (OR-Tools, Compass, GA, Tsiligirides, Gurobi) instead of a trained model (Transformer, PN, GPN)
    if opts.baseline is not None:
        if problem.NAME == 'op' and opts.data_dist == 'coop' and opts.test_coop:
            tours, inputs_dict = [], {}
            for agent in range(opts.num_agents):
                ds = dataset
                ds.data[0] = inputs[agent]
                tour, inp, model_name = baselines(opts.baseline, problem, ds, device)
                tours.append(tour)
                inputs_dict[agent] = inp
            plot_multitour(opts.num_agents, tours, inputs_dict, problem.NAME, model_name,
                           data_dist=opts.data_dist)
        return

    # Load model (Transformer, PN, GPN) for evaluation on the chosen device
    model, _ = load_model(opts.load_path)
    model.set_decode_type('greedy')
    model.num_depots = opts.num_depots
    model.eval()  # Put in evaluation mode to not track gradients
    model.to(device)
    if isinstance(model, AttentionModel):
        model_name = 'Transformer'

    # OP (coop)
    if problem.NAME == 'op' and (opts.data_dist == 'coop' or opts.data_dist == 'nocoop') and opts.test_coop:
        tours = []
        for i in range(opts.num_agents):
            for k, v in inputs[i].items():
                inputs[i][k] = v.unsqueeze(0).to(device)
            _, _, tour = model(inputs[i], return_pi=True)
            tours.append(tour.cpu().detach().numpy().squeeze())
            for k, v in inputs[i].items():
                inputs[i][k] = v.cpu().detach().numpy().squeeze()
        plot_multitour(opts.num_agents, tours, inputs, problem.NAME, model_name, data_dist=opts.data_dist,
                       num_depots=opts.num_depots)
        return

    # OP (const, dist, unif)
    elif problem.NAME == 'op' and (opts.data_dist == 'const' or opts.data_dist == 'dist' or opts.data_dist == 'unif'):
        for k, v in inputs.items():
            inputs[k] = v.unsqueeze(0).to(device)

    # Calculate tour
    _, _, tour = model(inputs, return_pi=True)

    # Torch tensors to numpy
    tour = tour.cpu().detach().numpy().squeeze()
    for k, v in inputs.items():
        inputs[k] = v.cpu().detach().numpy().squeeze()

    # Print/Plot results
    print(np.insert(tour, 0, 0, axis=0))
    plot_tour(tour, inputs, problem.NAME, model_name, data_dist=opts.data_dist, num_depots=opts.num_depots)


if __name__ == "__main__":
    main(arguments())
