import math
import sys
import types
from typing import NamedTuple

import speechbrain as sb
import parsebrain as pb  # extension of speechbrain
import torch
import wandb
from hyperpyyaml import load_hyperpyyaml
from transformers import CamembertModel, CamembertTokenizerFast
import numpy as np

from parsebrain.processing.dependency_parsing.graph_based.mst import score_to_tree
from parsebrain.dataio.pred_to_file.pred_to_conllu import write_token_dict_conllu


class Edge(NamedTuple):
    gov: int
    label: str
    dep: int



class GraphParser(sb.core.Brain):
    def compute_forward(self, batch, stage):
        tokens = batch.tokens.data
        tokens = tokens.to(self.device)
        tokens_conllu = batch.tokens_conllu.data.to(self.device)
        features = self.hparams.features_extractor.extract_features(
            tokens, tokens_conllu
        )
        seq_len = torch.tensor([len(w) for w in batch.words]).to(self.device)
        # bi directional rnn
        features, _ = self.modules.rnn(features)
        pos_scores, arc_scores, dep_scores = self.modules.graph_parser(features)
        arc_scores = self.hparams.softmax(arc_scores)
        dep_scores = self.hparams.softmax(dep_scores)
        return {
            "pos_scores": pos_scores,
            "arc_scores": arc_scores,
            "dep_scores": dep_scores,
            "seq_len": seq_len,
        }

    def compute_objectives(self, predictions, batch, stage):
        """
        batch :
            pos : [bacth_size, max_seq]
            head : [batch_size, max_seq]
            deps : [batch_size, max_seq]
        predictions :
            pos_scores : [batch_size, max_seq, num_POS]
            head : [batch_size, max_seq, max_seq]
            deps : [batch_size, max_seq, max_seq, num_dep]
        """
        batch_size = predictions["pos_scores"].shape[0]
        head = batch.head.data.to(self.device)
        num_padded_deps = head.shape[1]
        num_deprels = predictions["dep_scores"].shape[-1]

        # pos loss
        pos = batch.pos_tokens.data.to(self.device)
        loss_pos = self.hparams.pos_cost(
            predictions["pos_scores"], pos, length=predictions["seq_len"]
        )
        # arc loss, may need other format for target

        arc_loss = self.hparams.arc_cost(
            predictions["arc_scores"], head, length=predictions["seq_len"]
        )
        # label loss, must not take into account the non existing arc
        # we replace the value of padding with 0
        dep = batch.dep_tokens.data.to(self.device)
        # todo: replace positive_heads generation with seq_len instead of this ?
        # is it needed if the padding is already 0 ?
        positive_heads = head.masked_fill(head.eq(self.hparams.LABEL_PADDING), 0)
        heads_selection = positive_heads.view(batch_size, num_padded_deps, 1, 1)
        # [batch, n_dependents, 1, n_labels]
        heads_selection = heads_selection.expand(
            batch_size, num_padded_deps, 1, num_deprels
        )
        # [batch, n_dependents, 1, n_labels]
        # the dim=2 is to only squeeze this dimension
        # in case of batch size = 1 at end of epoch.
        predicted_labels_scores = torch.gather(
            predictions["dep_scores"], -2, heads_selection
        ).squeeze(dim=2)
        dep_loss = self.hparams.dep_cost(
            predicted_labels_scores, dep, length=predictions["seq_len"]
        )

        if sb.Stage.TRAIN != stage:
            for a_scores, d_scores, p_scores, seq_len, form, sent_id in zip(
                predictions["arc_scores"],
                predictions["dep_scores"],
                predictions["pos_scores"],
                predictions["seq_len"],
                batch.words,
                batch.sent_id,
            ):
                tree = score_to_tree(
                    a_scores[:seq_len, :seq_len],
                    d_scores[:seq_len, :seq_len, :],
                    p_scores[:seq_len, :],
                    sent_id=sent_id,
                    greedy=False,
                    form=form,
                    reverse_pos_dict=reverse_pos_dict,
                    reverse_dep_label_dict=reverse_dep_label_dict,
                )
                self.result_trees.append(tree)

        return 0.3 * loss_pos + 0.4 * arc_loss + 0.3 * dep_loss

    def _convert_tree_to_conllu(self):
        for tree in self.result_trees:
            s_id = tree["sent_id"]
            self.data_valid[s_id] = {"sent_id": s_id, "sentence": []}
            for i, (edge, pos, f) in enumerate(
                zip(tree["edges"], tree["pos_tags"], tree["form"]), start=1
            ):
                self.data_valid[s_id]["sentence"].append(
                    {
                        "ID": i,
                        "FORM": f,
                        "UPOS": pos,
                        "HEAD": edge[0],
                        "DEPREL": edge[1],
                    }
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.VALID:
                st = self.hparams.file_valid
                goldPath_CoNLLU = self.hparams.valid_conllu
                #order = self.dev_order
            else:
                st = self.hparams.file_test
                goldPath_CoNLLU = self.hparams.test_conllu
                #order = self.test_order

            self._convert_tree_to_conllu()
            with open(st, "w", encoding="utf-8") as f_v:
                write_token_dict_conllu(self.data_valid, f_v)
            self.result_trees = []
            self.data_valid = {}
            d = types.SimpleNamespace()
            d.system_file = st
            d.gold_file = goldPath_CoNLLU
            metrics = self.hparams.eval_conll.evaluate_wrapper(d)

            stage_stats["LAS"] = metrics["LAS"].f1 * 100
            stage_stats["UAS"] = metrics["UAS"].f1 * 100
            stage_stats["UPOS"] = metrics["UPOS"].f1 * 100
            print(
                f"UPOS : {stage_stats['UPOS']} , UAS : {stage_stats['UAS']} LAS : {stage_stats['LAS']}"
            )
        # Optimization of learning rate, logging, checkpointing
        if stage == sb.Stage.VALID:
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}
            wandb.log(wandb_stats)
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    #"lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )


            self.checkpointer.save_and_keep_only(
                meta={"LAS": stage_stats["LAS"]}, max_keys=["LAS"]
            )
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    #"lr_model": old_lr_model,
                },
                test_stats = stage_stats
            )



    def init_optimizers(self):
        "Initializes the model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:
            self.model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.model_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            # plot_grad_flow(self.hparams.modules["parser"].named_parameters())
            # exit()
            if self.check_gradients(loss):
                self.model_optimizer.step()

            self.model_optimizer.zero_grad()
        return loss.detach()


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = pb.dataio.dataset.DynamicItemDatasetConllu.from_conllu(
        conllu_path=hparams["train_conllu"],
        keys=hparams["conllu_keys"],
    )
    # sorting by len speed up training.
    train_data = train_data.filtered_sorted(sort_key="sent_len")

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
        # we add root in the begining of the sentence.
        words.insert(0, "root")
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
            # [1:-1] to remove the begining and ending <> symbole of bert
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

    @sb.utils.data_pipeline.takes("POS", "HEAD", "DEP")
    @sb.utils.data_pipeline.provides("pos_tokens", "head", "dep_tokens")
    def syntax_pipeline(pos, head, dep):
        pos.insert(0, "ROOT")
        try:
            poss = []
            for p in pos:
                if p == 'N':
                    p = 'NOM'
                if p == 'VINF':
                    p = 'VNF'
                poss.append(p.upper())

            pos_tokens = torch.tensor([pos_dict.get(p) for p in poss])
        except RuntimeError as e:
            print(poss)
            print([pos_dict.get(p) for p in poss])
            print(pos_dict)
            raise e
        yield pos_tokens
        head = [int(h) for h in head]
        head.insert(0, 0)
        yield torch.tensor(head)
        #dummy root for graph based (special tokens)
        dep = [d.lower() for d in dep]
        dep.insert(0, "ROOT")
        try:
            dep_token = torch.tensor([dep_label_dict.get(d) for d in dep])
        except RuntimeError as e:
            print(dep)
            print([dep_label_dict.get(d) for d in dep])
            raise e
        yield dep_token

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


dep_label_dict = {
    "periph": 0,
    "subj": 1,
    "root": 2,
    "dep": 3,
    "dm": 4,
    "spe": 5,
    "mark": 6,
    "para": 7,
    "aux": 8,
    "disflink": 9,
    "morph": 10,
    "parenth": 11,
    "aff": 12,
    "ROOT": 13,
    "__joker__": 14,
    "deletion": 15,
    "insertion": 16
}
reverse_dep_label_dict = {v: k for k, v in dep_label_dict.items()}


pos_dict = {
    "PADDING": 0,
    "ADV": 1,
    "CLS": 2,
    "VRB": 3,
    "PRE": 4,
    "INT": 5,
    "DET": 6,
    "NOM": 7,
    "COO": 8,
    "CLI": 9,
    "ADJ": 10,
    "VNF": 11,
    "CSU": 12,
    "ADN": 13,
    "PRQ": 14,
    "VPP": 15,
    "PRO": 16,
    "NUM": 17,
    "X": 18,
    "CLN": 19,
    "VPR": 20,
    "ROOT": 21,
    "INSERTION": 22
}

reverse_pos_dict = {v: k for k, v in pos_dict.items()}


def main():
    wandb.init()
    # end test debug
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
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
    camembert = CamembertModel.from_pretrained("camembert-base").to(run_opts["device"])
    # tokenizer = hparams["tokenizer"]

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    brain = GraphParser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    brain.hparams.features_extractor.set_model(camembert)
    brain.tokenizer = tokenizer

    brain.data_valid = {}
    brain.result_trees = []
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
    import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
