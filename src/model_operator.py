import os
import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import json
import random
from tqdm import tqdm
import wandb

from src.utils import load_from_file, get_pretrained_embeddings, print_config, save_config, load_checkpoint, get_results, record_predictions, labels_2_mention_str

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import BertForTokenClassification, BertConfig

from src.dataset import DialogueDataset

class ModelOperator:
    def __init__(self, args):

        # set up output directory
        self.output_dir = os.path.join(args.experiment_dir, args.run_name)
        if not os.path.exists(args.experiment_dir):
            os.mkdir(args.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(os.path.join(args.experiment_dir,"runs/")):
            os.mkdir(os.path.join(args.experiment_dir,"runs/"))

        # initialize model config
        self.config = vars(args)

        if args.real_run:
            run_name = "{}-{}".format(args.experiment_dir, args.run_name)
        else:
            run_name = None

        # initialize weights and biases
        wandb.init(name=run_name,
                   notes=args.a_nice_note,
                   project="coreference-detection",
                   config=self.config,)

        # check if there is a model to load
        if args.old_model_dir is not None:
            self.use_old_model = True
            self.load_dir = args.old_model_dir
            load_from_file(
                os.path.join(self.load_dir, "config.json"), self.config)

        else:
            self.use_old_model = False

        # train
        self.train_dataset = DialogueDataset(
            os.path.join(self.config["dataset_filename"], "train_data.json"),
            self.config["sentence_len"])

        self.data_loader_train = DataLoader(
            self.train_dataset, self.config["train_batch_size"], shuffle=True)
        self.config["train_len"] = len(self.train_dataset)

        self.t_total = len(self.data_loader_train) // self.config["gradient_accumulation_steps"] * self.config["num_epoch"]

        # eval
        self.val_dataset = DialogueDataset(
            os.path.join(self.config["dataset_filename"], "val_data.json"),
            self.config["sentence_len"])

        self.data_loader_val = DataLoader(
            self.val_dataset, self.config["val_batch_size"], shuffle=False)
        self.config["val_len"] = len(self.val_dataset)

        self.config["vocab_size"] = self.train_dataset.tokenizer.vocab_size

        # print and save the config file
        print_config(self.config)
        save_config(os.path.join(self.output_dir, "config.json"),
                          self.config)

        # set device
        self.device = torch.device('cuda')

        self.b_config = BertConfig.from_pretrained('bert-base-uncased')
        self.b_config.num_labels = self.config["label_len"]

        # create model
        self.model = BertForTokenClassification(self.b_config).to(self.device)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        # create optimizer
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.config["weight_decay"]},
            {'params': [p for n, p in self.model.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"],
                                eps=self.config["adam_epsilon"])
        self.scheduler = WarmupLinearSchedule(self.optimizer,
                                       warmup_steps=self.config["warmup_steps"],
                                       t_total=self.t_total)

        # load old model, optimizer if there is one
        if self.use_old_model:
            self.model, self.optimizer = load_checkpoint(
                os.path.join(self.load_dir, "model.bin"),
                self.model, self.optimizer, self.device)

        #self.optimizer.optimizer.to(torch.device('cpu'))
        if self.config["weight"] is None:
            self.weight = None
        else:
            self.weight = torch.Tensor(self.config["weight"]).to(self.device)

        wandb.config.update(self.config)
        wandb.watch(self.model)


    def train(self, num_epochs):
        metrics = {"best_epoch":0, "highest_f1":0}

        # output an example
        self.output_example(0)

        for epoch in range(num_epochs):

            epoch_metrics = dict()

            # train
            epoch_metrics["train"] = self.execute_phase(epoch, "train")
            # save metrics
            metrics["epoch_{}".format(epoch)] = epoch_metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # validate
            epoch_metrics["val"] = self.execute_phase(epoch, "val")
            # save metrics
            metrics["epoch_{}".format(epoch)] = epoch_metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # save checkpoint
            #if epoch_metrics["val"]["avg_results"]["F1"] > metrics["highest_f1"]:
            #if epoch_metrics["train"]["loss"] < metrics["lowest_loss"]:
            if epoch % 5 == 0:
                self.save_checkpoint(os.path.join(self.output_dir, "model.bin"))
                metrics["lowest_f1"] = epoch_metrics["val"]["avg_results"]["F1"]
                metrics["best_epoch"] = epoch

                test_results = self.get_test_predictions(
                    os.path.join(self.config["dataset_filename"], "test_data.json"),
                    os.path.join(self.output_dir, "predictions{}.json".format(epoch)))

            save_dict = dict()
            for phase, data in epoch_metrics.items():
                save_dict.update({"{}_loss".format(phase): data["loss"]})
                save_dict.update({"{}_{}".format(phase, key): value for key, value in data["avg_results"].items()})

            wandb.log(save_dict)

            # output an example
            self.output_example(epoch+1)

    def execute_phase(self, epoch, phase):
        if phase == "train":
            self.model.train()
            dataloader = self.data_loader_train
            batch_size = self.config["train_batch_size"]
            train = True
        else:
            self.model.eval()
            dataloader = self.data_loader_val
            batch_size = self.config["val_batch_size"]
            train = False

        start = time.clock()
        phase_metrics = dict()
        epoch_loss = list()
        epoch_metrics = list()
        results = {"accuracy": list(), "precision": list(), "recall": list(), "F1": list()}

        average_epoch_loss = None
        for i, batch in enumerate(tqdm(dataloader,
                          mininterval=2, desc=phase, leave=False)):
            # prepare data
            input_ids, tgt= map(
                lambda x: x.to(self.device), batch[:2])

            ids = batch[2]
            start_end_idx = batch[3]

            # forward
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels=tgt)

            loss, scores = outputs[:2]

            average_loss = float(loss)
            epoch_loss.append(average_loss)
            average_epoch_loss = np.mean(epoch_loss)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config["max_grad_norm"])
                # update parameters
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                self.model.zero_grad()

            pred = torch.argmax(scores.view(-1, self.config["label_len"]), 1)
            get_results(tgt.view(-1).cpu(), pred.view(-1).cpu(), results)

        phase_metrics["avg_results"] = {key: np.mean(value) for key, value in results.items()}
        phase_metrics["loss"] = average_epoch_loss

        phase_metrics["time_taken"] = time.clock() - start
        string = ' {} loss: {:.3f} '.format(phase, average_epoch_loss)
        print(string, end='\n')
        return phase_metrics

    def get_test_predictions(self, test_filename, save_filename):
        test_dataset = DialogueDataset(
            test_filename,
            self.config["sentence_len"])

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, self.config["val_batch_size"], shuffle=True)

        with open(test_filename, 'r') as f:
            data = json.load(f)

        start = time.clock()
        phase_metrics = dict()
        epoch_loss = list()
        epoch_metrics = list()
        results = {"accuracy": list(), "precision": list(), "recall": list(),
                   "F1": list()}
        average_epoch_loss = None
        for i, batch in enumerate(tqdm(test_data_loader,
                                       mininterval=2, desc='test', leave=False)):
            # prepare data
            input_ids, tgt = map(
                lambda x: x.to(self.device), batch[:2])

            ids = batch[2]
            start_end_idx = batch[3]

            # forward
            output = self.model(input_ids, labels=tgt)


            loss, scores = output[:2]

            average_loss = float(loss)
            epoch_loss.append(average_loss)
            average_epoch_loss = np.mean(epoch_loss)

            pred = torch.argmax(scores, 2)
            record_predictions(pred, data, ids, start_end_idx)
            get_results(tgt.view(-1).cpu(), pred.view(-1).cpu(), results)

        phase_metrics["avg_results"] = {key: np.mean(value) for key, value in
                                        results.items()}
        phase_metrics["loss"] = average_epoch_loss

        phase_metrics["time_taken"] = time.clock() - start
        string = ' {} loss: {:.3f} '.format('test', average_epoch_loss)
        print(string, end='\n')

        data["results"] = phase_metrics

        with open(save_filename, 'w') as f:
            json.dump(data, f, indent=2)

        return phase_metrics



    def save_checkpoint(self, filename):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, filename)

    def output_example(self, epoch):
        random_index = random.randint(0, len(self.val_dataset))
        example = self.val_dataset[random_index]

        # prepare data
        input_ids, tgt_seq = map(
            lambda x: torch.from_numpy(x).to(self.device).unsqueeze(0), example[:2])


        # forward
        output = self.model(input_ids, labels=tgt_seq)

        scores = output[1].view(-1, self.config["label_len"])

        words = input_ids.tolist()[0]
        target_strings = labels_2_mention_str(tgt_seq.squeeze(0))
        output_strings = labels_2_mention_str(torch.argmax(scores, dim=1))

        # get history text
        string = "word: output - target\n"
        data = list()
        for word, t, o in zip(words, target_strings, output_strings):
            token = self.train_dataset.tokenizer.convert_ids_to_tokens(word)
            if token != "<blank>":
                string += "[{}: {} - {}], \n".format(token, o, t)
                data.append([token, o, t])

        # print
        print("\n------------------------\n")
        print(string)
        print("\n------------------------\n")

        # add result to tensorboard
        wandb.log({"example": wandb.Table(data=data, columns=["word", "output", "target"])})
