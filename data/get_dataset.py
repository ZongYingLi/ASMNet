from copy import copy
def get_dataset(name="ntu13"):
    if name == "sxia":
        from data.xia import Xia
        return Xia
    elif name == "humanact12":
        from data.humanact12poses import HumanAct12Poses
        return HumanAct12Poses

def get_datasets(parameters):
    name = parameters["dataset"]

    DATA = get_dataset(name)
    if name == 'sxia':
        train = DATA(datapath="../xia/", split='train')
        test = DATA(datapath="../xia/", split='val')
        datasets = {"train": train, "test": test}

        parameters["num_classes"] = 6
        parameters["nfeats"] = 3
        parameters["njoints"] = 21
    elif name == 'humanact12':
        dataset = DATA(split="train", **parameters)
        train = dataset
        test = copy(train)
        test.split = test

        datasets = {"train": train,
                    "test": test}

        dataset.update_parameters(parameters)
    else:
        print("other dataset except xia")

    return datasets
