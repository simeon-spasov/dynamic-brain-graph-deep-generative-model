import uuid

from .dataset import load_hcp_datasets
from .model import Model
from .train import get_optimizer, get_scheduler, train
from .utils import get_device, set_random_seed


def run(args): 
    # seed for random number generators
    set_random_seed(seed=args.seed, is_gpu=args.is_gpu)
    # device to train / test model on
    device = get_device(is_gpu=args.is_gpu, gpu_number=args.gpu_number)
    # experiment id
    args.exp_id = uuid.uuid4().hex[:8]

    # data
    datasets = load_hcp_datasets(args.data.valid_split, args.data.test_split, args.data.num_nodes, 
                                 args.data.num_timesteps, args.train.window_size, args.seed) 
    train_data, valid_data, test_data = datasets.values()

    # model
    # model = Model()

    # optimizer and scheduler
    opt, sch = args.optimizer.pop("name"), args.scheduler.pop("name")
    optimizer = get_optimizer(opt)(model.parameters(), **args.optimizer)
    scheduler = get_scheduler(sch)(optimizer, **args.scheduler) if sch is not None else None

    print("\nexperiment {}".format(args.exp_id))
    print("model", args.model.name "dataset", args.data.name, "device", device.type)
    print("train size", len(train_data), "valid size", len(valid_data), "test size", len(test_data))

    # train
    print("\ntraining")
    # train(model, train_data, valid_data, optimizer, scheduler, device, args)
    # test()

