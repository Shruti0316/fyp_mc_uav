# Routing Problems for Multiple Cooperative UAVs using Transformers

## Paper
Solving a variant of the Orieentering Problem (OP) called the Orienteering Problem with Multiple Prizes and Types of
Node (OP-MP-TN) with a cooperative multi-agent system based on Transformer Networks. For more details, please see our
[paper](https://doi.org/10.1016/j.engappai.2023.106085). If this repository is useful for your work, please cite our paper:

```
@article{FUERTES2023106085,
    title = {Solving routing problems for multiple cooperative Unmanned Aerial Vehicles using Transformer networks},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {122},
    pages = {106085},
    year = {2023},
    issn = {0952-1976},
    doi = {https://doi.org/10.1016/j.engappai.2023.106085},
    url = {https://www.sciencedirect.com/science/article/pii/S0952197623002695},
    author = {Daniel Fuertes and Carlos R. del-Blanco and Fernando Jaureguizar and Juan José Navarro and Narciso García},
    keywords = {Unmanned Aerial Vehicle, Orienteering problem, Deep reinforcement learning, Attention model, Shared regions},
    abstract = {Missions involving Unmanned Aerial Vehicle usually consist of reaching a set of regions, performing some actions in each region, and returning to a determined depot after all the regions have been successfully visited or before the fuel/battery is totally consumed. Hence, planning a route becomes an important task for many applications, especially if a team of Unmanned Aerial Vehicles is considered. From this team, coordination and cooperation are expected to optimize results of the mission. In this paper, a system for managing multiple cooperative Unmanned Aerial Vehicles is presented. This system divides the routing problem into two stages: initial planning and routing solving. Initial planning is a first step where the regions to be visited are grouped in multiple clusters according to a distance criterion, with each cluster being assigned to each of the Unmanned Aerial Vehicles. Routing solving computes the best route for every agent considering the clusters of the initial planning and a variant of the Orienteering Problem. This variant introduces the concept of shared regions, allowing an Unmanned Aerial Vehicle to visit regions from other clusters and compensating for the suboptimal region clustering of the previous stage. The Orienteering Problem with shared regions is solved using the deep learning architecture Transformer along with a deep reinforcement learning framework. This architecture is able to obtain high-quality solutions much faster than conventional optimization approaches. Extensive results and comparison with other Combinatorial Optimization algorithms, including cooperative and non-cooperative scenarios, have been performed to show the benefits of the proposed solution.}
}
``` 

## Dependencies

* python3 >= 3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/) >= 1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib
* [k-means-constrained](https://joshlk.github.io/k-means-constrained/)
* fuzzy-c-means
* scikit-learn

## Usage

First, it is necessary to create training, testing, and validation sets:
```bash
python3 create_dataset.py --name train --seed 1111 --graph_sizes 20 --dataset_size 1280000 --cluster km --num_agents 2 --max_length 2
python3 create_dataset.py --name test --seed 1234 --graph_sizes 20 --dataset_size 10000 --cluster km --num_agents 2 --max_length 2
python3 create_dataset.py --name val --seed 4321 --graph_sizes 20 --dataset_size 10000 --cluster km --num_agents 2 --max_length 2
```
Note that the option `--cluster` defines the type of clustering for the initial planning: K-Means(`km`), K-Means
constrained(`kmc`), or Fuzzy C-Means(`fcm`). The option `--num_agents` defines the number of agents/clusters. The option
`max_length` indicates the normalized time limit to solve the problem.

To train a Transformer model (`attention`) use:
```bash
python3 run.py --model attention --graph_size 20 --max_length 2 --num_agents 2 --cluster km --data_dist coop --baseline rollout --train_dataset data/op/1depots/2agents/coop/km/20/train_seed1111_L2.pkl --val_dataset data/op/1depots/2agents/coop/km/20/val_seed4321_L2.pkl
```

Pointer Network (`pointer`) and Graph Pointer Network (`gpn`) can also be trained with the `--model` option. To resume
training, load your last saved model with the `--resume` option.

Evaluate your trained models with:
```bash
python3 eval.py data/op/1depots/2agents/coop/km/20/test_seed1234_L2.pkl --model outputs/op_coop20/attention_run... --num_agents 2
```
If the epoch is not specified, by default the last one in the folder will be used.

Baselines like [OR-Tools](https://developers.google.com/optimization), [Gurobi](https://www.gurobi.com),
[Tsiligirides](https://www.tandfonline.com/doi/abs/10.1057/jors.1984.162),
[Compass](https://github.com/bcamath-ds/compass) or a [Genetic Algorithm](https://github.com/mc-ride/orienteering) can
be executed as follows:
```bash
python3 -m problems.op.op_baseline --method ortools --multiprocessing True --datasets data/op/1depots/2agents/coop/km/20/test_seed1234_L2.pkl
```
To run Compass, you need to install it by running the `install_compass.sh` script from within the `problems/op`
directory. To use Gurobi, obtain a ([free academic](http://www.gurobi.com/registration/academic-license-reg)) license
and follow the
[installation instructions](https://www.gurobi.com/documentation/8.1/quickstart_windows/installing_the_anaconda_py.html)
. OR-Tools has to be installed too (`pip install ortools`).

Finally, you can visualize an example of executions using:
```bash
python3 test_plot.py --graph_size 20 --num_agents 2 --data_dist coop --load_path outputs/op_coop20/km/attention_rollout_2agents_20240305T122127 --test_coop True
```

Use the `--baseline` option to visualize the prediction of one of the baselines mentioned before:
```bash
python3 test_plot.py --graph_size 20 --num_agents 2 --data_dist coop --baseline ortools --test_coop True
```

### Other options and help
```bash
python3 run.py -h
python3 eval.py -h
python3 -m problems.op.op_baseline -h
python3 test_plot.py -h
```

## Acknowledgements
This repository is an adaptation of
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for the case of multiple
cooperative UAVs.
