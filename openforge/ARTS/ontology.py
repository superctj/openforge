from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import pickle

from openforge.ARTS.helpers.mongodb_helper import *
# from ARTS.helpers.openai_helper import *


class OntologyNode(object):
    embeddingByLevelAndIdx = {}
    # textsSetByLevel = defaultdict(set)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    DEVICE = None
    THRESHOLD = None

    def __init__(self, level, idx, text) -> None:
        self.level = level
        self.idx = (
            OntologyNode.embeddingByLevelAndIdx[self.level].shape[0] if level else 0
        )
        # self.idx = idx
        self.texts = [text]
        self.children: list[OntologyNode] = []
        self.tbl_column_matched = []  # {tbl_id: col_name} matched with this node
        self.text_to_tbl_column_matched = defaultdict(list)

        if level:
            self.init_embed()

    def add_matched_table_column(self, tbl_id, col_name, text=""):
        self.tbl_column_matched.append((tbl_id, col_name))
        if text:
            self.text_to_tbl_column_matched[text].append((tbl_id, col_name))
        if text:
            self.text_to_tbl_column_matched[text].append((tbl_id, col_name))

    @staticmethod
    def init__device_and_threshold(device, threshold):
        OntologyNode.DEVICE = device
        OntologyNode.THRESHOLD = threshold

    @staticmethod
    def init_embeddingByLevelAndIdx():
        OntologyNode.embeddingByLevelAndIdx[1] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )
        OntologyNode.embeddingByLevelAndIdx[2] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )
        OntologyNode.embeddingByLevelAndIdx[3] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )
        OntologyNode.embeddingByLevelAndIdx[4] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )
        OntologyNode.embeddingByLevelAndIdx[5] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )
        OntologyNode.embeddingByLevelAndIdx[6] = torch.empty(
            (0, 384), device=OntologyNode.DEVICE
        )

    def init_embed(self):
        OntologyNode.embeddingByLevelAndIdx[self.level] = torch.cat(
            (
                OntologyNode.embeddingByLevelAndIdx[self.level],
                self.compute_embedding().unsqueeze(0),
            ),
            dim=0,
        )
        assert self.idx == OntologyNode.embeddingByLevelAndIdx[self.level].shape[0] - 1

    def __str__(self) -> str:
        return self.texts[0]

    def __repr__(self):
        return "Node: ({})".format("; ".join(self.texts))
        # return f"Node: \"{self.texts[0]}\"\n\tChildren: {", ".join[ child.text for child in self.children]}"

    def is_contains_text(self, text):
        if text in self.texts:
            return True
        return False

    def add_text(self, text: str):
        if self.is_contains_text(text):
            pass
        else:
            self.texts.append(text)
            OntologyNode.embeddingByLevelAndIdx[self.level][
                self.idx
            ] = self.compute_embedding()
        # self.embeddings = encoder.encode(self.texts, device=OntologyNode.DEVICE, convert_to_tensor=True) # type: ignore
        # self.embedding = torch.mean(self.embeddings, dim=0) # type: ignore

    def compute_embedding(self):
        if len(self.texts) >= 1:
            return torch.mean(
                OntologyNode.embedder.encode(self.texts, device=OntologyNode.DEVICE, convert_to_tensor=True),
                dim=0)  # type: ignore
        else:
            raise NotImplementedError

    def get_embedding(self):
        return OntologyNode.embeddingByLevelAndIdx[self.level][self.idx]

    def add_child(self, child):
        assert self.level + 1 == child.level
        self.children.append(child)

    @staticmethod
    def most_similiar_by_level(text: str, level: int):
        query_embedding = OntologyNode.embedder.encode(
            text, convert_to_tensor=True, device=OntologyNode.DEVICE
        )
        corpus_embeddings = OntologyNode.embeddingByLevelAndIdx[level]

        # an empty level
        if corpus_embeddings.shape[0] == 0:
            return 0, -1

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]  # type: ignore
        most_relevant = torch.topk(cos_scores, k=1)

        score, idx = most_relevant[0][0], most_relevant[1][0]
        return score, idx

    def most_similar_with_children(self, text: str):
        if not self.children:
            return -1

        query_embedding = OntologyNode.embedder.encode(
            text, convert_to_tensor=True, device=OntologyNode.DEVICE
        )
        corpus_embeddings = OntologyNode.embeddingByLevelAndIdx[self.level + 1][
            [x.idx for x in self.children]
        ]

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]  # type: ignore
        most_relevant = torch.topk(cos_scores, k=1)
        score, idx = most_relevant[0][0], most_relevant[1][0]
        if score > OntologyNode.THRESHOLD:
            return self.children[idx].idx
        return -1


def builf_ontology_with_decomposed_column_semantics(
        column_semantics: list,
        run_id,
        device="cuda:0",
        threshold=0.9,
        checkpoint_per_cs: int = 0,
        special_on_root_cs=False
):
    """
    column_semantics: [(tbl_id, col_name, decomposed_cs)]
    """
    SAVE_DIRECTORY = "/home/jjxing/ssd/openforge/ARTS/output/"
    # initialize the ontology root
    ontologyRoot = OntologyNode(level=0, idx=0, text="OntologyRoot")
    OntologyNode.init__device_and_threshold(device=device, threshold=threshold)
    OntologyNode.init_embeddingByLevelAndIdx()

    print(
        "Start building ontology:\n\t device: {}\n\t threshold: {}".format(
            OntologyNode.DEVICE, OntologyNode.THRESHOLD
        )
    )
    print("Will save ontology to: {}column_semantics_ontology_threshold_{}_run_{}.pickle".format(
        SAVE_DIRECTORY, threshold, run_id
    ))

    answer = ""
    while answer not in ["y", "n"]:
        answer = input("OK to push to continue [Y/N]? ").lower()
    if answer == "n":
        exit()

    # intialize list of nodes by levels
    ListOfOntologyNode = list[OntologyNode]
    nodeByLevel = defaultdict(ListOfOntologyNode)

    # add root to level 0
    nodeByLevel[0].append(ontologyRoot)

    # do somethiong for level 1
    level_1_cs_text_set = set()
    level_1_cs_text_to_idx = {}

    # index that maps tbl column to ontology nodes
    tbl_to_nodes = defaultdict(dict)

    # start enumerating the column semantics
    cnt = 0
    for tbl_id, col_name, decomposed_column_semantic in (pbar := tqdm(column_semantics)):
        pbar.set_description("Building Ontology: ")
        cnt += 1
        if checkpoint_per_cs and cnt and cnt % checkpoint_per_cs == 0:
            with open(
                    SAVE_DIRECTORY + "column_semantics_ontology_threshold_{}_run_{}_checkpoint_{}.pickle".format(
                        threshold, run_id, cnt
                    ),
                    "wb",
            ) as outfile:
                pickle.dump(
                    {
                        "nodeByLevel": nodeByLevel,
                        "embeddings": OntologyNode.embeddingByLevelAndIdx,
                        "tbl_to_nodes": tbl_to_nodes,
                        "threshold": OntologyNode.THRESHOLD,
                        "device": OntologyNode.DEVICE,
                    },
                    outfile,
                )

        if not decomposed_column_semantic:
            continue

        tbl_to_nodes[tbl_id][col_name] = []
        decomposed_column_semantic = [x.lower() for x in decomposed_column_semantic]

        parentNode = ontologyRoot
        if len(decomposed_column_semantic) > 6:
            decomposed_column_semantic = (
                    decomposed_column_semantic[:5] + decomposed_column_semantic[-1:]
            )
        for idx, cs in enumerate(decomposed_column_semantic):
            level = idx + 1
            if level > 6:
                break
            if special_on_root_cs and level == 1:
                if cs in level_1_cs_text_set:
                    level_1_node = nodeByLevel[1][level_1_cs_text_to_idx[cs]]
                else:
                    # 1. initialized the node
                    level_1_node = OntologyNode(
                        level=level, idx=len(nodeByLevel[level]), text=cs
                    )
                    # 2. add the node to nodeByLevel
                    nodeByLevel[level_1_node.level].append(level_1_node)
                    level_1_cs_text_set.add(cs)
                    level_1_cs_text_to_idx[cs] = level_1_node.idx
                    assert (
                            level_1_node.texts
                            == nodeByLevel[1][level_1_cs_text_to_idx[cs]].texts
                    )
                    # 3. add the node to it's parent node
                    ontologyRoot.add_child(level_1_node)
                    # print("create new node on level {}: {}".format(level, level_1_node))
                level_1_node.add_matched_table_column(tbl_id=tbl_id, col_name=col_name, text=cs)
                level_1_node.add_matched_table_column(tbl_id=tbl_id, col_name=col_name, text=cs)
                tbl_to_nodes[tbl_id][col_name].append((level, level_1_node.idx))
                parentNode = level_1_node
            else:
                matched_child_idx = parentNode.most_similar_with_children(text=cs)
                if matched_child_idx >= 0:
                    matched_child = nodeByLevel[level][matched_child_idx]
                    # if matched_child.is_contains_text(cs):
                    #     print("level {} node found: ".format(level), matched_child)
                    # else:
                    #     print("level {}, {} merged to node: ".format(level, cs), matched_child)
                    matched_child.add_text(text=cs)
                    matched_child.add_matched_table_column(
                        tbl_id=tbl_id, col_name=col_name, text=cs
                    )
                    parentNode = matched_child
                else:
                    # 1. initialized the node
                    new_node = OntologyNode(
                        level=level, idx=len(nodeByLevel[level]), text=cs
                    )
                    # 2. add the node to nodeByLevel
                    nodeByLevel[new_node.level].append(new_node)
                    # 3. add the node to it's parent node
                    parentNode.add_child(new_node)
                    new_node.add_matched_table_column(tbl_id=tbl_id, col_name=col_name, text=cs)
                    new_node.add_matched_table_column(tbl_id=tbl_id, col_name=col_name, text=cs)
                    # print("create new node on level {}: {}".format(level, new_node))

                    parentNode = new_node
                tbl_to_nodes[tbl_id][col_name].append((level, parentNode.idx))

    with open(
            SAVE_DIRECTORY + "column_semantics_ontology_threshold_{}_run_{}.pickle".format(
                threshold, run_id
            ),
            "wb",
    ) as outfile:
        pickle.dump(
            {
                "nodeByLevel": nodeByLevel,
                "embeddings": OntologyNode.embeddingByLevelAndIdx,
                "tbl_to_nodes": tbl_to_nodes,
                "threshold": OntologyNode.THRESHOLD,
                "device": OntologyNode.DEVICE,
            },
            outfile,
        )


def load_ontology(file_path, topk=10):
    with open(file_path, 'rb') as infile:
        data = pickle.load(infile)

    print("Loading ontology from: ", file_path)
    nodeByLevel = data['nodeByLevel']
    OntologyNode.init__device_and_threshold(device=data['device'], threshold=data['device'])
    OntologyNode.embeddingByLevelAndIdx = data['embeddings']
    tbl_to_nodes = data['tbl_to_nodes']

    for level in range(1, 7):
        print("level {}: ".format(level), len(nodeByLevel[level]), "nodes.")
        cs_cnts = []
        for node in nodeByLevel[level]:
            cs_cnts.append((node.texts[0], len(node.tbl_column_matched)))
        cs_cnts.sort(key=lambda t: t[1], reverse=True)
        print(cs_cnts[:topk])


def load_ontology_save_info(file_path, topk=10):
    rst = []
    with open(file_path, 'rb') as infile:
        data = pickle.load(infile)

    print("Loading ontology from: ", file_path)
    nodeByLevel = data['nodeByLevel']
    OntologyNode.init__device_and_threshold(device=data['device'], threshold=data['device'])
    OntologyNode.embeddingByLevelAndIdx = data['embeddings']
    tbl_to_nodes = data['tbl_to_nodes']

    for level in range(1, 7):
        rst[level] = []
        print("level {}: ".format(level), len(nodeByLevel[level]), "nodes.")
        cs_cnts = []
        for node in nodeByLevel[level]:
            cs_cnts.append((node.texts[0], len(node.tbl_column_matched)))
        cs_cnts.sort(key=lambda t: t[1], reverse=True)
        for cs_node in cs_cnts[:topk]:
            print(cs_node)
            print(cs_node.tbl_column_matched)
        # print(cs_cnts[:topk])


def run_nyc_gpt_ontology(special_on_root_cs=False):
    device = "cuda:0"
    threshold = 0.9
    run_id = "nyc_gpt_3.5_merge_root"
    import json
    # from ARTS.helpers.decompose_column_semantics import decompose_sentence_with_stanza, nlp
    #
    # nyc_open_data = list(data_gov_mongo['metadata'].find({"organization.title": "City of New York"}))
    # to_build_ontology = []
    # for ds in (pbar := tqdm(nyc_open_data)):
    #     pbar.set_description(f"Gathering column semantics: ")
    #     csvfiles = list(data_gov_csv_file_col.find({"dataset_name": ds['name']}))
    #     if csvfiles:
    #         gpt = data_gov_gpt_annotation_col.find_one({"table_id": csvfiles[0]['_id']})
    #         gpt_decomposed = data_gov_denpendency_parse_col.find_one({"table_id": csvfiles[0]['_id']})
    #
    #         if gpt_decomposed:
    #             for col in gpt_decomposed['dp'].keys():
    #                 if gpt_decomposed['dp'][col]['text']:
    #                     to_build_ontology.append((csvfiles[0]['_id'], col, decompose_sentence_with_stanza(
    #                         nlp(gpt_decomposed['dp'][col]['text']))))

    # for col in gpt_decomposed['decomposed'].keys():
    # to_build_ontology.append((csvfiles[0]['_id'], col, gpt_decomposed['decomposed'][col]))
    # print(csvfiles[0]['_id'])
    # print(ds['name'])
    # print("Processing {} columns.".format(len(to_build_ontology)))
    with open('/home/jjxing/ssd/column_semantics_ontology_building/explore/nyc_new_decomposed.json', 'r') as infile:
        to_build_ontology = json.load(infile)
        # json.dump(to_build_ontology, outfile)
    builf_ontology_with_decomposed_column_semantics(to_build_ontology, run_id, device, threshold,
                                                    checkpoint_per_cs=0, special_on_root_cs=special_on_root_cs)


def run_nyc_official_ontology(special_on_root_cs=False):
    import json
    from ARTS.helpers.decompose_column_semantics import decompose_sentence_with_stanza, nlp
    device = "cuda:0"
    threshold = 0.9
    run_id = "nyc_official_column_semantics_test_save_merge_info"

    with open(
            '/home/jjxing/ssd/column_semantics_ontology_building/explore/nyc_open_data_column_annotations.json') as infile:
        d = json.load(infile)

    to_build_ontology = []
    for ds in tqdm(d):
        dataset_name = ds['dataset_name']
        csvfiles = list(data_gov_csv_file_col.find({"dataset_name": dataset_name}))
        file_id = csvfiles[0]['_id']
        for (col_name, col_desc) in ds['nyc_annotation']:
            if col_desc:
                col_desc = col_desc.lower().strip()
                doc = nlp(col_desc)
                decomposed = decompose_sentence_with_stanza(doc)
                to_build_ontology.append((file_id, col_name, decomposed))

    print("Processing {} columns.".format(len(to_build_ontology)))

    builf_ontology_with_decomposed_column_semantics(to_build_ontology, run_id, device, threshold, checkpoint_per_cs=0,
                                                    special_on_root_cs=special_on_root_cs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=["load", "run"])
    parser.add_argument("--ontology", "-o", type=str)
    parser.add_argument("--topk", "-k", type=int, default=10)
    parser.add_argument("--special_on_root", "-s", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    if args.function == "load":
        load_ontology_save_info(args.ontology, topk=args.topk)
    elif args.function == "run":
        run_nyc_gpt_ontology(special_on_root_cs=args.special_on_root)
