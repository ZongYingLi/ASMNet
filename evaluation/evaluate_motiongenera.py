from src_parser.evaluation import parser

def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["humanact12", "sxia"]:
        from evaluation.gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
