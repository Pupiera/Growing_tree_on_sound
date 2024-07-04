import sys

import wandb
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from transformers import CamembertModel, CamembertTokenizerFast

import parsebrain as pb
import torch


class Parser(sb.core.Brain):
    def compute_forward(self, batch, stage):
        tokens = batch.tokens.data
        tokens = tokens.to(self.device)
        tokens_conllu = batch.tokens_conllu.data.to(self.device)
        features = self.hparams.features_extractor.extract_features(
            tokens, tokens_conllu
        )
        batch_len = features.shape[0]
        seq_len = torch.tensor([len(w) for w in batch.words]).to(self.device)
        num_hidden = self.hparams.rnn_num_layer
        if self.hparams.rnn_bidirectional:
            num_hidden = num_hidden * 2
        hidden = torch.zeros(num_hidden, batch_len, 768, device=self.device)
        cell = torch.zeros(num_hidden, batch_len, 768, device=self.device)
        lstm_out, _ = self.modules.neural_network(features, (hidden, cell))
        logits_posLabel = self.modules.posDep2Label(lstm_out)
        p_posLabel = self.hparams.log_softmax(logits_posLabel)
        logits_depLabel = self.modules.depDep2Label(lstm_out)
        p_depLabel = self.hparams.log_softmax(logits_depLabel)
        logits_govLabel = self.modules.govDep2Label(lstm_out)
        p_govLabel = self.hparams.log_softmax(logits_govLabel)
        result = {
            "p_posLabel": p_posLabel,
            "p_depLabel": p_depLabel,
            "p_govLabel": p_govLabel,
            "seq_len": seq_len,
        }
        return result

    def compute_objectives(self, predictions, batch, stage):
        allowed = 0
        sent_id = batch.sent_id
        pos = batch.pos_tokens.data.to(self.device)
        head = batch.head.data.to(self.device)
        dep = batch.dep_tokens.data.to(self.device)
        loss_depLabel = self.hparams.depLabel_cost(
            predictions["p_depLabel"],
            dep,
            length=predictions["seq_len"],
            allowed_len_diff=allowed,
        )
        loss_govLabel = self.hparams.govLabel_cost(
            predictions["p_govLabel"],
            head,
            length=predictions["seq_len"],
            allowed_len_diff=allowed,
        )
        loss_POS = self.hparams.posLabel_cost(
            predictions["p_posLabel"],
            pos,
            length=predictions["seq_len"],
            allowed_len_diff=allowed,
        )
        if stage != sb.Stage.TRAIN:
            self.hparams.evaluator.decode(
                [
                    predictions["p_govLabel"],
                    predictions["p_depLabel"],
                    predictions["p_posLabel"],
                ],
                batch.words,
                sent_id,
            )
        return 0.3 * loss_POS + 0.4 * loss_govLabel + 0.3 * loss_depLabel

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if stage == sb.Stage.VALID:
                st = self.hparams.file_valid
                goldPath_CoNLLU = self.hparams.valid_conllu
                order = self.dev_order
            else:
                st = self.hparams.file_test
                goldPath_CoNLLU = self.hparams.test_conllu
                order = self.test_order
            self.hparams.evaluator.writeToCoNLLU(st, order)
            metrics_dict = self.hparams.evaluator.evaluateCoNLLU(goldPath_CoNLLU, st)
            stage_stats["LAS"] = metrics_dict["LAS"].f1 * 100
            stage_stats["UAS"] = metrics_dict["UAS"].f1 * 100
            stage_stats["UPOS"] = metrics_dict["UPOS"].f1 * 100
        if stage == sb.Stage.VALID:
            '''
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            '''
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    #"lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}  # fuse dict
            wandb.log(wandb_stats)
            self.checkpointer.save_and_keep_only(
                meta={
                    "LAS": stage_stats["LAS"],
                },
                max_keys=["LAS"],
            )
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    #"lr_model": old_lr_model,
                },
                test_stats=stage_stats,
            )
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}  # fuse dict
            wandb.log(wandb_stats)


    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.model_opt_class(self.hparams.model.parameters())
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.optimizer)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["train_conllu"],
        keys=hparams["conllu_keys"],
    )

    valid_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["valid_conllu"],
        keys=hparams["conllu_keys"],
    )

    test_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["test_conllu"],
        keys=hparams["conllu_keys"],
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("sent_id", "words")
    @sb.utils.data_pipeline.provides(
        "sent_id",
        "words",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "tokens_conllu",
    )
    def text_pipeline(sent_id, words):
        yield sent_id
        wrd = " ".join(words)
        yield words
        # tokens_list = tokenizer.encode_as_ids(wrd)
        tokens_list = tokenizer.encode_plus(wrd)["input_ids"]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        tokens_conllu = []
        for i, str in enumerate(words):
            # tokens_conllu.extend([i + 1] * len(tokenizer.encode_as_ids(str)))
            tokens_conllu.extend(
                [i + 1] * len(tokenizer.encode_plus(str)["input_ids"][1:-1])
            )
        x = []
        y = 0
        for t_c in reversed(tokens_conllu):
            if t_c != y:
                x.append(True)
                y = t_c
            else:
                x.append(False)
        x.reverse()
        tokens_conllu = torch.BoolTensor(x)
        yield tokens_conllu

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    @sb.utils.data_pipeline.takes("words", "POS", "HEAD", "DEP")
    @sb.utils.data_pipeline.provides("pos_tokens", "head", "dep_tokens")
    def syntax_pipeline(wrd, pos, head, dep):
        """
        compute gold configuration here
        """
        try:
            poss = []
            for p in pos:
                p = p.upper()
                if p =='VINF':
                    p = 'VNF'
                elif p =='N':
                    p = 'NOM'
                poss.append(p)
            pos_list = torch.LongTensor([label_alphabet[2].get(p) for p in poss])
        except TypeError as e:
            print(poss)
            print([label_alphabet[2].get(p) for p in poss])
            raise e
        yield pos_list
        fullLabel = encoding.encodeFromList(
            [w for w in wrd],
            [p for p in poss],
            [g for g in head],
            [d for d in dep],
        )
        # first task is gov pos and relative position
        try:
            govDep2label = torch.LongTensor(
                [
                    label_alphabet[0].get(fl.split("\t")[-1].split("{}")[0].upper())
                    for fl in fullLabel
                ]
            )
            yield govDep2label
        except TypeError as e:
            print(wrd)
            print([fl.split("\t")[-1].split("{}")[0] for fl in fullLabel])
            print(
                [
                    label_alphabet[0].get(fl.split("\t")[-1].split("{}")[0].upper())
                    for fl in fullLabel
                ]
            )
            raise TypeError() from e
        # second task is dependency type
        try:
            depDep2Label = torch.LongTensor(
                [label_alphabet[1].get(fl.split("{}")[1].upper()) for fl in fullLabel]
            )
        except TypeError as e:
                print([label_alphabet[1].get(fl.split("{}")[1].upper()) for fl in fullLabel])
                raise e
            
        yield depDep2Label

    sb.dataio.dataset.add_dynamic_item(datasets, syntax_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "sent_id",
            "words",
            "tokens_list",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "tokens_conllu",
            "pos_tokens",
            "head",
            "dep_tokens",
        ],
    )
    return train_data, valid_data, test_data


def build_label_alphabet(path_encoded_train):
    label_gov = dict()
    label_dep = dict()
    label_pos = dict()
    with open(path_encoded_train, "r", encoding="utf-8") as inputFile:
        for line in inputFile:
            field = line.split("\t")
            if len(field) > 1:
                fullLabel = field[-1]
                labelSplit = fullLabel.split("{}")
                govLabel = labelSplit[0].upper()
                if govLabel not in label_gov:
                    label_gov[govLabel] = len(label_gov)
                depLabel = labelSplit[-1].replace("\n", "").upper()
                if depLabel not in label_dep:
                    label_dep[depLabel] = len(label_dep)
                pos = field[1]
                if pos not in label_pos:
                    label_pos[pos] = len(label_pos)
    for pos_key in label_pos:
        for i in range(1, 20):
            key = f"{i}@{pos_key}"
            if "+" + key not in label_gov.keys():
                label_gov["+" + key] = len(label_gov)
            if "-" + key not in label_gov.keys():
                label_gov["-" + key] = len(label_gov)
    for i in range(1, 10):
        key = f"{i}@INSERTION"
        if "+" + key not in label_gov.keys():
            label_gov["+" + key] = len(label_gov)
        if "-" + key not in label_gov.keys():
            label_gov["-" + key] = len(label_gov)

    #label_gov["-1@INSERTION"] = len(label_gov)
    #label_gov["+1@INSERTION"] = len(label_gov)
    label_gov["-1@DELETION"] = len(label_gov)
    label_dep["INSERTION"] = len(label_dep)
    label_dep["DELETION"] = len(label_dep)
    label_pos["INSERTION"] = len(label_pos)
    print(len(label_gov))
    print(len(label_dep))
    print(len(label_pos))
    return [label_gov, label_dep, label_pos]


def get_id_from_CoNLLfile(path):
    """
    Get the sentence id from the conll file in the order
    Will be used to write in the same order for comparaison sakes.
    """
    sent_id = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("# sent_id"):
                field = line.split("=")
                sent_id.append(field[1].replace(" ", "").replace("\n", ""))
    return sent_id


def build_reverse_alphabet(alphabet):
    reverse = []
    for alpha in alphabet:
        reverse.append({item: key for (key, item) in alpha.items()})
    return reverse


path_encoded_train = "all.seq"  # For alphabet generation
label_alphabet = build_label_alphabet(path_encoded_train)
reverse_label_alphabet = build_reverse_alphabet(label_alphabet)


def main():
    wandb.init()
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    camembert = CamembertModel.from_pretrained("camembert-base").to(run_opts["device"])

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    brain = Parser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.hparams.features_extractor.set_model(camembert)
    brain.tokenizer = tokenizer
    # not super clean ...
    global encoding
    encoding = brain.hparams.encoding
    brain.hparams.evaluator.set_alphabet(label_alphabet)

    brain.dev_order = get_id_from_CoNLLfile(hparams["valid_conllu"])
    brain.test_order = get_id_from_CoNLLfile(hparams["test_conllu"])
    brain.data_valid = []
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
    brain.evaluate(
        test_data,
        max_key="LAS",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )


if __name__ == "__main__":
    main()
