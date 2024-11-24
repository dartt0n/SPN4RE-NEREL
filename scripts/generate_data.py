import json
from pathlib import Path
from typing import Annotated

import typer
from attrs import define
from cattr import unstructure
from datasets import load_dataset
from rich.progress import track


@define
class NerelEntity:
    id: int
    type: str
    text: str
    start: int
    end: int


def parse_entities(dataset_entites: list[str]) -> list[NerelEntity]:
    result = []
    for row in dataset_entites:
        id_str, entity_attributes, text = row.split("\t")
        id = int(id_str.removeprefix("T"))

        type = entity_attributes.split()[0]

        for attr_str in entity_attributes.removeprefix(type).split(";"):
            start, end = map(int, attr_str.split())
            result.append(NerelEntity(id, type, text, start, end))
    return result


@define
class NerelRelation:
    id: int
    type: str
    arg1: int
    arg2: int


def parse_relations(dataset_relations: list[str]) -> list[NerelRelation]:
    result = []
    for row in dataset_relations:
        id_str, relation_attributes = row.split("\t")
        id = int(id_str.removeprefix("R"))
        type, arg1_str, arg2_str = relation_attributes.split()
        arg1 = int(arg1_str.removeprefix("Arg1:T"))
        arg2 = int(arg2_str.removeprefix("Arg2:T"))
        result.append(NerelRelation(id, type, arg1, arg2))
    return result


@define
class OutputRelation:
    em1Text: str
    em2Text: str
    label: str


@define
class OutputEntry:
    sentText: str
    relationMentions: list[OutputRelation]


@define
class OutputTriple:
    text: str
    triple_list: list[tuple[str, str, str]]


def main(
    output_dir: Annotated[Path, typer.Argument(default_factory=lambda: Path("data") / "NEREL")],
):
    ds = load_dataset("MalakhovIlya/NEREL", trust_remote_code=True)
    splits = ["train", "test", "dev"]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    relation_types: set[str] = set()
    entity_types: set[str] = set()

    for split_name in splits:
        output_entries: list[OutputEntry] = []
        for entry in track(ds[split_name], description=f"NEREL/{split_name:<10}"):
            entities = parse_entities(entry["entities"])
            entity_dict = {entity.id: entity for entity in entities}
            relations = parse_relations(entry["relations"])

            output_relations = [
                OutputRelation(
                    em1Text=entity_dict[relation.arg1].text,
                    em2Text=entity_dict[relation.arg2].text,
                    label=relation.type,
                )
                for relation in relations
            ]

            relation_types.update([relation.type for relation in relations])
            entity_types.update([entity.type for entity in entities])

            output_entry = OutputEntry(sentText=entry["text"], relationMentions=output_relations)
            output_entries.append(output_entry)

        with open(output_dir / f"{split_name}.jsonl", "w") as f:
            for entry in output_entries:
                f.write(json.dumps(unstructure(entry), ensure_ascii=False) + "\n")

        output_triples: list[OutputTriple] = []
        for entry in output_entries:
            output_triples.append(
                OutputTriple(
                    text=entry.sentText,
                    triple_list=[
                        (
                            relation.em1Text,
                            relation.label,
                            relation.em2Text,
                        )
                        for relation in entry.relationMentions
                    ],
                )
            )

        with open(output_dir / f"{split_name}-triplets.json", "w") as f:
            json.dump([unstructure(triple) for triple in output_triples], f, ensure_ascii=False, indent=2)

    id2rel = dict(enumerate(sorted(relation_types)))
    rel2id = {type: id for id, type in id2rel.items()}

    with open(output_dir / "rel2id.json", "w") as f:
        json.dump([id2rel, rel2id], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    typer.run(main)
