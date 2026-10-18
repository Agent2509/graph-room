"""Entity resolution and deduplication module."""

import re
from typing import Dict, List, Tuple
from graph_rag.models import Entity, Relationship


class EntityResolver:
    """Normalizes and resolves entities and relationships across chunks."""

    def __init__(self):
        self.corporate_suffixes = [
            r"Inc\.?$",
            r"LLC\.?$",
            r"Corp\.?$",
            r"Corporation$",
            r"Ltd\.?$",
            r"Co\.?$",
        ]
        self.honorifics = [
            r"\bDr\.?\b",
            r"\bProf\.?\b",
            r"\bMr\.?\b",
            r"\bMrs\.?\b",
            r"\bMs\.?\b",
        ]

    def normalize_name(self, name: str) -> str:
        cleaned = name.strip()
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned).strip()
        cleaned = re.sub(r"^[Tt]he\s+", "", cleaned).strip()

        for honorific in self.honorifics:
            cleaned = re.sub(rf"^{honorific}\s+", "", cleaned, flags=re.IGNORECASE).strip()

        for suffix in self.corporate_suffixes:
            cleaned = re.sub(rf"\s+{suffix}\b$", "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = cleaned.rstrip(",.")
        return cleaned

    @staticmethod
    def merge_descriptions(desc1: str, desc2: str, max_chars: int = 600) -> str:
        d1 = (desc1 or "").strip()
        d2 = (desc2 or "").strip()
        if not d1:
            return d2[:max_chars].strip()
        if not d2:
            return d1[:max_chars].strip()
        if d2.lower() in d1.lower():
            return d1[:max_chars].strip()
        if d1.lower() in d2.lower():
            return d2[:max_chars].strip()

        s1 = [s.strip() for s in re.split(r"(?<=[.?!])\s+", d1) if s.strip()]
        s2 = [s.strip() for s in re.split(r"(?<=[.?!])\s+", d2) if s.strip()]

        retained = list(s1)
        for cand in s2:
            cand_lower = cand.lower()
            cand_words = set(re.findall(r"\w+", cand_lower))
            if not cand_words:
                continue
            is_dup = False
            for existing in retained:
                ex_words = set(re.findall(r"\w+", existing.lower()))
                if ex_words and len(cand_words & ex_words) / len(cand_words) > 0.70:
                    is_dup = True
                    break
            if not is_dup:
                candidate_text = " ".join(retained + [cand])
                if len(candidate_text) <= max_chars:
                    retained.append(cand)

        merged = " ".join(retained)
        return merged[:max_chars].strip()

    def resolve_entities(self, entities: List[Entity]) -> Dict[str, Entity]:
        resolved: Dict[str, Entity] = {}

        for entity in entities:
            canonical = self.normalize_name(entity.name)
            if not canonical:
                continue

            lower_key = canonical.lower()

            if lower_key not in resolved:
                resolved[lower_key] = Entity(
                    name=canonical,
                    type=entity.type,
                    description=entity.description[:600].strip(),
                    source_chunk_ids=list(set(entity.source_chunk_ids)),
                )
            else:
                existing = resolved[lower_key]
                if entity.description:
                    existing.description = self.merge_descriptions(existing.description, entity.description)

                existing.source_chunk_ids = list(
                    set(existing.source_chunk_ids + entity.source_chunk_ids)
                )

                if existing.type == "CONCEPT" and entity.type != "CONCEPT":
                    existing.type = entity.type

        return resolved

    def resolve_relationships(
        self,
        relationships: List[Relationship],
        resolved_entities: Dict[str, Entity],
    ) -> List[Relationship]:
        edge_map: Dict[Tuple[str, str, str], Relationship] = {}

        for rel in relationships:
            src_norm = self.normalize_name(rel.source)
            tgt_norm = self.normalize_name(rel.target)

            src_key = src_norm.lower()
            tgt_key = tgt_norm.lower()

            if src_key not in resolved_entities or tgt_key not in resolved_entities:
                continue

            canonical_src = resolved_entities[src_key].name
            canonical_tgt = resolved_entities[tgt_key].name

            if canonical_src.lower() == canonical_tgt.lower():
                continue

            edge_key = (canonical_src.lower(), canonical_tgt.lower(), rel.relation_type)

            if edge_key not in edge_map:
                edge_map[edge_key] = Relationship(
                    source=canonical_src,
                    target=canonical_tgt,
                    relation_type=rel.relation_type,
                    description=rel.description[:400].strip(),
                    weight=rel.weight,
                    source_chunk_ids=list(set(rel.source_chunk_ids)),
                )
            else:
                existing_rel = edge_map[edge_key]
                if rel.description:
                    existing_rel.description = self.merge_descriptions(
                        existing_rel.description, rel.description, max_chars=400
                    )

                existing_rel.weight += rel.weight
                existing_rel.source_chunk_ids = list(
                    set(existing_rel.source_chunk_ids + rel.source_chunk_ids)
                )

        return list(edge_map.values())
