import sys
from pathlib import Path
from typing import List

import torch
from conllup.conllup import emptyNodeJson, emptySentenceJson, sentenceJson_T
from flask import Flask, jsonify, request

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.modules.BertForDepRel import BertForDeprel
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.types import PredictionConfig

app = Flask(__name__)


@app.route("/ud_analysis/v1", methods=["POST"])
def analyze():
    # {"lang": "en/ru/es/fr/ja", "sentences": {"id": [token...],...}}
    data = request.get_json()

    lang = data["lang"]
    sentences = data["sentences"]

    unlabled_ud_sents: List[sentenceJson_T] = []
    total_tokens = 0
    for sent_id, words in sentences.items():
        sent = emptySentenceJson()
        sent["metaJson"]["sent_id"] = sent_id
        for i, word in enumerate(words):
            sent["treeJson"]["nodesJson"][f"{i}"] = emptyNodeJson(ID=f"{i}", FORM=word)
        total_tokens += len(words)
        unlabled_ud_sents.append(sent)

    # TODO: should be predictor.activate(lang)
    predictor.activate(lang)
    ud_dataset = model.encode_dataset(unlabled_ud_sents)
    predictions, elapsed_seconds = predictor.predict(ud_dataset)
    print(
        f"LOG: labeled {len(ud_dataset)} sentences/{total_tokens} tokens in "
        f"{elapsed_seconds} seconds ({len(ud_dataset)/elapsed_seconds} sents/sec, "
        f"{total_tokens/elapsed_seconds} tokens/sec)"
    )

    output = {
        "lang": lang,
        "parsed": {
            sent["metaJson"]["sent_id"]: sent["treeJson"]["nodesJson"]
            for sent in predictions
        },
    }
    return jsonify(output)


#     audio_filepath = data["audio_filepath"]
#     language = data[language]
#     result = model.transcribe(
#         audio_filepath,
#         language=language,
#         beam_size=5,
#         regroup=True,
#         vad=True,
#         temperature_increment_on_fallback=0.2,
#         best_of=5,
#     )
#     result.to_srt_vtt("audio.srt")
#     serializable_result = result.to_dict()

#     return jsonify(serializable_result)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            """Usage: python3 server.py <models_dir>
models_dir should be a directory containing all model directories.""",
            file=sys.stderr,
        )
        sys.exit(1)

    models_path = Path(sys.argv[1])
    if not models_path.is_dir():
        print(f"{models_path} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # each subdirectory is a model path
    model_paths = {d.name: d for d in models_path.iterdir() if d.is_dir()}
    active_model = next(iter(model_paths))[0]

    device_config = get_devices_configuration("-2")

    model = BertForDeprel.load_pretrained_for_prediction(
        model_paths, active_model, device_config.device
    )

    predictor = Predictor(
        model,
        PredictionConfig(batch_size=16, num_workers=0),
        device_config.multi_gpu,
    )
    torch.manual_seed(42)
    app.run(debug=True)
