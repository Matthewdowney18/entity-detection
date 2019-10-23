import argparse

from src.model_operator import ModelOperator
#import src.

def main():
    parser = argparse.ArgumentParser()
    # good arguments
    parser.add_argument("-dataset_filename",
                        default="reformatted_data/data_11",
                        type=str,
                        required = False,
                        help = "The input data dir. Should contain the csv for the task.")
    parser.add_argument("-experiment_dir",
                        default="experiments/exp_5/",
                        type=str,
                        required=False,
                        help="The output data dir")
    parser.add_argument("-run_name",
                        default="run_1/",
                        type=str,
                        required=False,
                        help="The output data dir")
    parser.add_argument("-old_model_dir",
                        #default="experiments/exp_8/run_6/",
                        default=None,
                        type=str,
                        required=False,
                        help="filename of saved model. say None to train new model")
    parser.add_argument("-num_epoch",
                        default=100,
                        type=int,
                        required=False,
                        help="The number of training epochs")
    parser.add_argument("-a_nice_note",
                        default="using windows now",
                        type=str,
                        required=False,
                        help="leave a nice lil note for yourself in the future")
    parser.add_argument("-real_run",
                        default=False,
                        type=bool,
                        required=False,
                        help="true if a real run")

    # dataset info
    parser.add_argument("-do_eval",
                        default=True,
                        type=bool,
                        required=False,
                        help="True to train model")
    parser.add_argument("-do_train",
                        default=True,
                        type=bool,
                        required=False,
                        help="True to train model")
    parser.add_argument("-min_count",
                        default=1,
                        type=int,
                        required=False,
                        help="The minimum amount of instances to be in vocab")
    parser.add_argument("-train_batch_size",
                        default=600,
                        type=int,
                        required=False,
                        help="The batch size for training")
    parser.add_argument("-val_batch_size",
                        default=100,
                        type=int,
                        required=False,
                        help="The batch size for training")
    parser.add_argument("-sentence_len",
                        default=50,
                        type=int,
                        required=False,
                        help="The max length of the history")
    parser.add_argument("-window_size",
                        default=6,
                        type=int,
                        required=False,
                        help="The max length of the response")

    # transformer specs
    parser.add_argument("-pretrained_embeddings_dir",
                        default="/home/mattd/embeddings/conll_3/reddit2.bin",
                        #default=None,
                        type=str,
                        required=False,
                        help="use pretrained embeddings")
    parser.add_argument("-embedding_dim",
                        default=1,
                        type=int,
                        required=False,
                        help="The embeddings dim will be ignored if pretrained"
                        "embeddings dir is not none")
    parser.add_argument("-model_dim",
                        default=1,
                        type=int,
                        required=False,
                        help="The hidden layer dimension")
    parser.add_argument("-inner_dim",
                        default=1024,
                        type=int,
                        required=False,
                        help="The inner dim")
    parser.add_argument("-num_layers",
                        default=6,
                        type=int,
                        required=False,
                        help="The number of layers")
    parser.add_argument("-num_heads",
                        default=8,
                        type=int,
                        required=False,
                        help="The number of attention heads")
    parser.add_argument("-dim_k",
                        default=64,
                        type=int,
                        required=False,
                        help="not really sure what k is")
    parser.add_argument("-dim_v",
                        default=64,
                        type=int,
                        required=False,
                        help="not really sure what v is")
    parser.add_argument("-dropout",
                        default=.3,
                        type=float,
                        required=False,
                        help="dropout probability")

    # optimizer specs
    parser.add_argument("-weight",
                        default=[0.8, 1.0],
                        type=list,
                        required=False,
                        help="The warmup steps for optimizer")
    parser.add_argument("-warmup_steps",
                        default=4000,
                        type=int,
                        required=False,
                        help="The warmup steps for optimizer")
    parser.add_argument("-label_smoothing",
                        default=False,
                        type=bool,
                        required=False,
                        help="The batch size for training")

    args = parser.parse_args()

    model_operator = ModelOperator(args)

    model_operator.train(args.num_epoch)


if __name__ == '__main__':
    main()