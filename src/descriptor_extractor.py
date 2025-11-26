from __future__ import annotations

import json
import logging
from typing import List, Optional, Union, Tuple, Any, Dict
from dataclasses import dataclass
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType

import numpy as np

import base64
import io
from openai import OpenAI
import re

from .siglip_itm import SigLipITM

@dataclass
class DescriptorExtractorConfig:
    """Descriptor 추출 설정"""
    use_chain_descriptors: bool = False
    gpt_model: str = "gpt-4o-mini"
    n_descriptors: int = 3


class DescriptorExtractor:
    """
    A class to handle the extraction of text descriptors from image observations
    using different strategies.
    """

    def __init__(self, itm: SigLipITM, cfg: DescriptorExtractorConfig):
        self.itm = itm
        self.cfg = cfg
        self.n_descriptors = cfg.n_descriptors
        self.client = OpenAI()
        self.primary_target_weight = 0.7
        self.secondary_target_weight = 0.3


    def extract_target_objects_from_description(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Uses GPT to extract a main target and context objects from a descriptive goal.
        Returns a tuple: (main_target_list, context_objects_list)
        """
        try:
            # Prepare the prompt for GPT
            prompt = (
                f"Analyze the following indoor navigation instruction: '{description}'\n"
                "Return a JSON object with two keys exactly:\n"
                "- 'main_target_object': a single noun phrase naming the physical object to be found. "
                "It must correspond to the head object typically mentioned before connectors such as 'with', 'and', 'featuring', 'that has'. "
                "Include distinguishing attributes (colour/material) if provided, e.g. 'white dresser'.\n"
                "- 'context_objects': an array containing at most one additional physical object that best helps localize the main target "
                "(e.g. items mentioned after those connectors). "
                "Return an empty array when no such supporting cue exists.\n"
                "Ensure each value is a concise noun phrase with no trailing commas.\n"
                "Example:\n"
                "{\n"
                "  \"main_target_object\": \"white dresser\",\n"
                "  \"context_objects\": [\"wall-mounted mirror\"]\n"
                "}"
            )
            
            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            response_str = response.choices[0].message.content
            
            if not response_str: return [], []

            response_json = json.loads(response_str)
            main_target = response_json.get("main_target_object")
            context_objects = response_json.get("context_objects", [])

            main_target_list: List[str] = []
            if isinstance(main_target, str):
                cleaned_main = main_target.strip(" ,.;")
                if cleaned_main:
                    main_target_list = [cleaned_main]

            if not isinstance(context_objects, list) or not all(isinstance(i, str) for i in context_objects):
                logging.warning(f"Could not parse a valid list for context_objects. Got: {context_objects}")
                context_objects = []

            cleaned_context: List[str] = []
            for ctx in context_objects:
                cleaned_ctx = ctx.strip(" ,.;")
                if cleaned_ctx:
                    cleaned_context.append(cleaned_ctx)
            context_objects = cleaned_context[:1]

            # Heuristic fallback to ensure a sensible primary/secondary even when GPT fails
            if not main_target_list and isinstance(description, str):
                desc_text = description.strip(" .")
                primary_fallback = ""
                secondary_fallback = ""

                lowered = desc_text.lower()
                 
                split_index = -1
                matched_sep_len = 0
                for sep in separators:
                    idx = lowered.find(sep)
                    if idx != -1:
                        split_index = idx
                        matched_sep_len = len(sep)
                        break

                if split_index != -1:
                    primary_fallback = desc_text[:split_index].strip(" ,.;")
                    trailing = desc_text[split_index + matched_sep_len :].strip(" .")
                    if trailing:
                        trailing_replaced = re.sub(r"\band\b", ",", trailing, flags=re.IGNORECASE)
                        trailing_candidates = [
                            part.strip(" ,.;")
                            for part in trailing_replaced.split(",")
                            if part.strip(" ,.;")
                        ]
                        if trailing_candidates:
                            secondary_fallback = trailing_candidates[0]
                else:
                    primary_fallback = desc_text

                if primary_fallback:
                    main_target_list = [primary_fallback]
                if secondary_fallback:
                    context_objects = [secondary_fallback]

            # Final clean-up to ensure unique, non-empty entries
            if main_target_list:
                main_target_list = [main_target_list[0]]
            if context_objects:
                context_objects = [context_objects[0]]

            logging.info(f"From '{description}', extracted Main: {main_target_list}, Context: {context_objects}")
            return main_target_list, context_objects

        except Exception as e:
            logging.error(f"Failed to extract objects from description due to an error: {e}", exc_info=True)
            return [], []

    def compute_itm_scores(
        self,
        primary_targets: List[str],
        secondary_targets: List[str],
        descriptors: List[str],
    ) -> Tuple[float, float, float]:
        """
        primary / secondary 타깃에 대한 ITM 점수를 계산해 (가중합, primary, secondary)를 반환합니다.
        """
        primary_score = 0.0
        secondary_score = 0.0

        if descriptors:
            for target in primary_targets:
                if not target:
                    continue
                scores = self.itm.text_text_scores(target, descriptors)
                if scores is not None and scores.size > 0:
                    primary_score = max(primary_score, float(scores.max()))

            for target in secondary_targets:
                if not target:
                    continue
                scores = self.itm.text_text_scores(target, descriptors)
                if scores is not None and scores.size > 0:
                    secondary_score = max(secondary_score, float(scores.max()))

        if secondary_targets:
            combined = (
                self.primary_target_weight * primary_score
                + self.secondary_target_weight * secondary_score
            )
        else:
            combined = primary_score

        return combined, primary_score, secondary_score

    def extract_keywords_from_image(self, goal_image: PILImageType) -> List[str]:
        """VLM을 사용하여 목표 이미지에서 핵심 키워드를 추출합니다."""
        logging.info("Extracting keywords from goal image...")
        try:
            # VLM에 전달하기 위해 이미지 인코딩
            buffered = io.BytesIO()
            goal_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            prompt = (
                "Analyze the following image and describe the main object "
                "with a few concise, comma-separated keywords. "
                "Focus on color, material, and object type. For example: "
                "'red leather armchair', 'round wooden coffee table', 'tall green ficus tree'."
            )

            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_str}"},
                            },
                        ],
                    }
                ],
                max_tokens=100,
                seed=getattr(self.cfg, 'seed', None),  # config에서 seed 가져오기
            )
            keywords_str = response.choices[0].message.content
            if not keywords_str:
                return []
            
            # 따옴표 제거 및 공백 정리
            keywords_str = keywords_str.replace("'", "").replace('"', '')
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            logging.info(f"Extracted keywords: {keywords}")
            return keywords

        except Exception as e:
            logging.error(f"Error extracting keywords from image: {e}", exc_info=True)
            return []

    def extract_keywords_from_image_modified(self, goal_image: PILImageType) -> List[str]:
        """VLM을 사용하여 목표 이미지에서 핵심 키워드를 추출합니다. (cch added)"""
        logging.info("Extracting keywords from goal image (modified version)...")
        # VLM에 전달하기 위해 이미지 인코딩 (한 번만 수행)
        buffered = io.BytesIO()
        goal_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Analyze the following image and describe the goal object for indoor navigation.\n"
            "If the image is unclear or no object can be identified, respond with {\"keywords\": [\"detection failed\"]}.\n"
            "Otherwise, respond in JSON with a 'keywords' array containing at most two descriptive phrases.\n"
            "The first keyword must be the most distinctive primary description; the second (if any) should be the best supplementary cue.\n"
            "Each keyword should mention color/material and object type, e.g. \"red leather armchair\"."
        )

        max_attempts = 3
        last_keywords: List[str] = ["detection failed"]

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.gpt_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_str}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                    response_format={"type": "json_object"},
                    seed=getattr(self.cfg, 'seed', None),
                )

                response_str = response.choices[0].message.content
                if not response_str:
                    logging.warning(
                        f"Image keyword extraction attempt {attempt} returned empty response."
                    )
                    last_keywords = ["detection failed"]
                    continue

                try:
                    response_json = json.loads(response_str)
                except json.JSONDecodeError as exc:
                    logging.warning(
                        f"Image keyword extraction attempt {attempt} failed to parse JSON: {exc}"
                    )
                    last_keywords = ["detection failed"]
                    continue

                keywords = response_json.get("keywords", [])
                if not isinstance(keywords, list):
                    logging.warning(
                        f"Image keyword extraction attempt {attempt} returned non-list keywords: {keywords}"
                    )
                    last_keywords = ["detection failed"]
                    continue

                cleaned_keywords: List[str] = []
                for kw in keywords:
                    if isinstance(kw, str) and kw.strip():
                        cleaned_kw = kw.strip().replace("'", "").replace('"', "")
                        if cleaned_kw and cleaned_kw.lower() != "detection failed":
                            cleaned_keywords.append(cleaned_kw)
                    elif isinstance(kw, list):
                        for nested_kw in kw:
                            if isinstance(nested_kw, str) and nested_kw.strip():
                                cleaned_nested_kw = nested_kw.strip().replace("'", "").replace('"', "")
                                if cleaned_nested_kw and cleaned_nested_kw.lower() != "detection failed":
                                    cleaned_keywords.append(cleaned_nested_kw)

                if cleaned_keywords:
                    cleaned_keywords = cleaned_keywords[:2]
                    logging.info(
                        f"Extracted keywords (modified, attempt {attempt}): {cleaned_keywords}"
                    )
                    return cleaned_keywords

                logging.warning(
                    f"Image keyword extraction attempt {attempt} produced no usable keywords."
                )
                last_keywords = ["detection failed"]

            except Exception as exc:
                logging.error(
                    f"Error extracting keywords from image (modified) on attempt {attempt}: {exc}",
                    exc_info=True,
                )
                last_keywords = ["detection failed"]

        logging.warning(
            f"Image keyword extraction failed after {max_attempts} attempts; returning 'detection failed'."
        )
        return last_keywords


    def analyze_scene_for_goal(
        self,
        rgb: np.ndarray,
        goal: Any, # Can be string or PIL Image
        key_objects: List[str],
    ) -> Tuple[List[str], float, List[str]]:
        """
        Performs a unified VLM analysis of a scene to get descriptors, a likelihood score, and parsed objects.
        """
        try:
            messages: List[Dict[str, Any]]

            # Prepare scene image encoding first, as it's used in multiple places
            scene_buffered = io.BytesIO()
            PILImage.fromarray(rgb).save(scene_buffered, format="PNG")
            scene_img_str = base64.b64encode(scene_buffered.getvalue()).decode("utf-8")

            # Goal-dependent message preparation (JSON response)
            if isinstance(goal, str) and goal.strip():
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    "Analyze the provided scene image with respect to the textual goal.\n"
                    f"Goal: '{goal}'.\n"
                    f"Return a compact JSON object with exactly these keys: \n"
                    f"- 'descriptors': an array of {self.n_descriptors} short, distinct visual descriptors (strings).\n"
                    f"- 'objects': an array of physical object names (strings) that APPEAR IN the descriptors (verbatim substrings);\n"
                    f"   do not invent new objects; include only nouns/noun phrases present in descriptors; deduplicate.\n"
                    f"- 'likelihood': a float in [0.0, 1.0] indicating how promising it is to proceed in this direction.\n"
                    f"No extra keys or text."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}"},
                            },
                        ],
                    }
                ]
            elif isinstance(goal, PILImageType):
                goal_buffered = io.BytesIO()
                goal.save(goal_buffered, format="PNG")
                goal_img_str = base64.b64encode(goal_buffered.getvalue()).decode("utf-8")
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    "The first image is the current scene; the second is the goal object.\n"
                    f"Return a compact JSON object with exactly these keys: \n"
                    f"- 'descriptors': an array of {self.n_descriptors} short, distinct visual descriptors (strings) of the scene relevant to the goal.\n"
                    f"- 'objects': an array of physical object names (strings) that APPEAR IN the descriptors (verbatim substrings);\n"
                    f"   do not invent new objects; include only nouns/noun phrases present in descriptors; deduplicate.\n"
                    f"- 'likelihood': a float in [0.0, 1.0] reflecting how promising it is to proceed toward the goal.\n"
                    f"No extra keys or text."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}", "detail": "low"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{goal_img_str}", "detail": "low"},
                            },
                        ],
                    }
                ]
            else:
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    f"Return a compact JSON object with: 'descriptors' (array of {self.n_descriptors} short visual descriptors), "
                    "'objects' (array of object names), and 'likelihood' (float 0.5)."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}"},
                            },
                        ],
                    }
                ]

            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=messages,
                max_tokens=300,
                response_format={"type": "json_object"},
                seed=getattr(self.cfg, 'seed', None),  # config에서 seed 가져오기
            )
            response_str = response.choices[0].message.content
            if not response_str:
                return [], 0.0, []

            # --- Parse JSON response ---
            try:
                parsed = json.loads(response_str)
            except Exception:
                logging.warning("Failed to parse JSON; falling back to empty objects.")
                parsed = {}

            descriptors = parsed.get("descriptors", [])
            if not isinstance(descriptors, list):
                descriptors = []
            descriptors = [d.strip() for d in descriptors if isinstance(d, str) and d.strip()]

            likelihood_score = parsed.get("likelihood", 0.0)
            try:
                likelihood_score = float(likelihood_score)
            except Exception:
                likelihood_score = 0.0
            likelihood_score = max(0.0, min(1.0, likelihood_score))

            objects_raw: List[str] = parsed.get("objects", [])
            if not isinstance(objects_raw, list):
                objects_raw = []
            objects_raw = [o.strip() for o in objects_raw if isinstance(o, str) and o.strip()]
            # Keep only objects that actually appear in descriptors (case-insensitive substring)
            objects: List[str] = []
            desc_lower = [d.lower() for d in descriptors]
            for o in objects_raw:
                ol = o.lower()
                if any(ol in d for d in desc_lower):
                    if o not in objects:
                        objects.append(o)
            
            return descriptors, likelihood_score, objects

        except Exception as e:
            logging.error(f"Failed during unified VLM analysis due to an error: {e}", exc_info=True)
            return [], 0.0, []



