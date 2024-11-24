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
    head_entity: NerelEntity
    tail_entity: NerelEntity


def parse_relations(dataset_relations: list[str], entities: dict[int, NerelEntity]) -> list[NerelRelation]:
    result = []
    for row in dataset_relations:
        id_str, relation_attributes = row.split("\t")
        id = int(id_str.removeprefix("R"))
        type, arg1_str, arg2_str = relation_attributes.split()
        entity_1_id = int(arg1_str.removeprefix("Arg1:T"))
        entity_2_id = int(arg2_str.removeprefix("Arg2:T"))

        result.append(NerelRelation(id, type, entities[entity_1_id], entities[entity_2_id]))
    return result


@define
class OutputEntry:
    text: str
    relations: list[NerelRelation]


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
        for entry in track(ds[split_name], description=f"NEREL/{split_name:<10}"):  # type: ignore
            entities = parse_entities(entry["entities"])  # type: ignore
            entity_dict = {entity.id: entity for entity in entities}
            relations = parse_relations(entry["relations"], entity_dict)  # type: ignore

            relation_types.update([relation.type for relation in relations])
            entity_types.update([entity.type for entity in entities])

            output_entry = OutputEntry(text=entry["text"], relations=relations)  # type: ignore
            output_entries.append(output_entry)

        with open(output_dir / f"{split_name}.json", "w") as f:
            json.dump([unstructure(entry) for entry in output_entries], f, ensure_ascii=False, indent=2)

    id2rel = dict(enumerate(sorted(relation_types)))
    rel2id = {type: id for id, type in id2rel.items()}

    with open(output_dir / "rel2id.json", "w") as f:
        json.dump([id2rel, rel2id], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    typer.run(main)
