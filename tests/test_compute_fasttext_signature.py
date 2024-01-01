import os
import pickle

from openforge.ARTS.ontology import OntologyNode
from openforge.feature_extraction.fb_fasttext import FasttextTransformer, compute_fasttext_signature
from openforge.utils.util import get_proj_dir


def test_compute_fasttext_signature():
    proj_dir = get_proj_dir(__file__, file_level=2)

    arts_output_filepath = os.path.join(proj_dir, "data/column_semantics_ontology_threshold_0.9_run_nyc_gpt_3.5_merge_root.pickle")
    assert(os.path.exists(arts_output_filepath))
    arts_level = 2
    num_head_concepts = 3
    num_val_samples = 10000

    with open(arts_output_filepath, "rb") as f:
        data = pickle.load(f)
        nodeByLevel = data["nodeByLevel"]
        OntologyNode.init__device_and_threshold(device=data["device"], threshold=data["device"])
        OntologyNode.embeddingByLevelAndIdx = data["embeddings"]

    nodeByLevel[arts_level].sort(key=lambda x:len(x.tbl_column_matched), reverse=True)
    fasttext_transformer = FasttextTransformer()

    for node in nodeByLevel[arts_level][:num_head_concepts]:
        # node is the head concept
        assert str(node) == node.texts[0]
        # each merged concept has at least one corresponding table column
        assert(len(node.texts) == len(node.text_to_tbl_column_matched))

        # use head concept as a reference point
        fasttext_signature = compute_fasttext_signature(
            node.text_to_tbl_column_matched[str(node)],
            fasttext_transformer,
            num_val_samples
        )
