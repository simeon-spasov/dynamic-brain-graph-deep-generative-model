import uuid

from .dataset import load_hcp_datasets
from .model import Model
from .train import get_optimizer, get_scheduler, train
from .utils import set_device, set_random_seed


def run(args): 
    # seed for random number generators
    set_random_seed(seed=args.seed, is_gpu=args.is_gpu)
    # device to run model on
    device = set_device(is_gpu=args.is_gpu, gpu_number=args.gpu_number)
    # experiment id
    args.exp_id = uuid.uuid4().hex[:8]

    # data
    datasets = load_hcp_datasets(args.data.valid_split, args.data.test_split, args.data.num_nodes, 
                                 args.data.num_timesteps, args.train.window_size, args.seed) 
    train_data, valid_data, test_data = datasets.values()

    # model
    model = Model(len(train_data), args.data.num_nodes, args.model.num_communities, args.model.alpha_dim, 
                  args.model.beta_dim, args.model.phi_dim, args.model.alpha_std, args.model.temp, 
                  args.model.window_size, args.model.window_stride, args.model.measure, args.model.percentile, device)

    # optimizer and scheduler
    opt, sch = args.optimizer.pop("name"), args.scheduler.pop("name")
    optimizer = get_optimizer(opt)(model.parameters(), **args.optimizer)
    scheduler = get_scheduler(sch)(optimizer, **args.scheduler) if sch is not None else None

    # print summary
    print("\nexperiment:", args.exp_id, "\nmodel:", args.model.name, "\ndataset:", args.data.name)
    print("train size:", len(train_data), "valid size:", len(valid_data), "test size:", len(test_data), "\ndevice:", device.type)

    # train
    print("\ntraining")
    train(model, train_data, valid_data, optimizer, scheduler, device, args)
    # test()

