"""Acoustic, phone-level pronunciation assessment.

The primary path uses a multilingual Wav2Vec2 CTC phone recognizer and aligns
the expected phone sequence to its frame-level posterior probabilities.  The
alignment provides a GOP-like score, an observed competing phone, and a time
range for every expected phone.

If the acoustic model cannot be loaded, callers still receive the same JSON
shape from a transcript-based fallback.  The fallback is deliberately marked
as such so it is never confused with acoustic assessment.
"""

from __future__ import annotations

import math
import os
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16000


def _is_enabled(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def ipa_units(value: str) -> List[str]:
    """Split IPA into useful display units instead of Unicode code points.

    This is used by the fallback and legacy score.  The acoustic path uses the
    model vocabulary because its CTC labels define the actual scoring units.
    """

    value = unicodedata.normalize("NFC", value or "")
    ignored = set(" /[](),.!?;:\"“”‘’")
    modifiers = {"ː", "ˑ", "ʰ", "ʷ", "ʲ", "ˠ", "ˤ", "ⁿ", "˞"}
    tie_bars = {"͡", "͜"}
    units: List[str] = []

    index = 0
    while index < len(value):
        char = value[index]
        if char in ignored or char.isspace() or char in {"ˈ", "ˌ"}:
            index += 1
            continue

        unit = char
        index += 1

        # A tie bar joins the following base symbol to the same phone.
        if index < len(value) and value[index] in tie_bars:
            unit += value[index]
            index += 1
            if index < len(value):
                unit += value[index]
                index += 1

        while index < len(value):
            following = value[index]
            if unicodedata.combining(following) or following in modifiers:
                unit += following
                index += 1
            else:
                break
        units.append(unit)

    return units


def levenshtein_alignment(
    expected: Sequence[str], observed: Sequence[str]
) -> List[Tuple[str, Optional[str]]]:
    """Align observed phones to expected phones.

    Insertions are skipped because the API is centred on expected phones.  A
    deletion is represented by ``None``.
    """

    rows, columns = len(expected) + 1, len(observed) + 1
    cost = np.zeros((rows, columns), dtype=np.int32)
    move = np.zeros((rows, columns), dtype=np.int8)
    cost[:, 0] = np.arange(rows)
    cost[0, :] = np.arange(columns)

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = cost[row - 1, column - 1] + (
                expected[row - 1] != observed[column - 1]
            )
            deletion = cost[row - 1, column] + 1
            insertion = cost[row, column - 1] + 1
            choices = (substitution, deletion, insertion)
            move[row, column] = int(np.argmin(choices))
            cost[row, column] = choices[move[row, column]]

    aligned_reversed: List[Tuple[str, Optional[str]]] = []
    row, column = len(expected), len(observed)
    while row or column:
        if row and column and move[row, column] == 0:
            aligned_reversed.append((expected[row - 1], observed[column - 1]))
            row -= 1
            column -= 1
        elif row and (not column or move[row, column] == 1):
            aligned_reversed.append((expected[row - 1], None))
            row -= 1
        else:
            column -= 1

    return list(reversed(aligned_reversed))


@dataclass
class CTCAlignment:
    target_frames: List[List[int]]
    state_path: List[int]


def ctc_forced_alignment(
    log_probabilities: torch.Tensor,
    target_ids: Sequence[int],
    blank_id: int,
) -> CTCAlignment:
    """Viterbi-align a known token sequence to CTC frame log probabilities."""

    if log_probabilities.ndim != 2:
        raise ValueError("CTC emissions must have shape [frames, vocabulary]")
    frame_count = int(log_probabilities.shape[0])
    if not target_ids:
        return CTCAlignment([], [0] * frame_count)

    states: List[int] = [blank_id]
    for target_id in target_ids:
        states.extend((int(target_id), blank_id))

    state_count = len(states)
    negative_infinity = torch.tensor(
        float("-inf"), device=log_probabilities.device, dtype=log_probabilities.dtype
    )
    previous = torch.full(
        (state_count,), negative_infinity, device=log_probabilities.device
    )
    previous[0] = log_probabilities[0, blank_id]
    if state_count > 1:
        previous[1] = log_probabilities[0, states[1]]

    backpointers = torch.full(
        (frame_count, state_count), -1, dtype=torch.int16, device="cpu"
    )

    for frame in range(1, frame_count):
        current = torch.full_like(previous, negative_infinity)
        for state_index, token_id in enumerate(states):
            candidates = [(previous[state_index], state_index)]
            if state_index > 0:
                candidates.append((previous[state_index - 1], state_index - 1))
            if (
                state_index > 1
                and token_id != blank_id
                and token_id != states[state_index - 2]
            ):
                candidates.append((previous[state_index - 2], state_index - 2))

            candidate_scores = torch.stack([candidate[0] for candidate in candidates])
            best_candidate = int(torch.argmax(candidate_scores).item())
            previous_state = candidates[best_candidate][1]
            current[state_index] = (
                candidate_scores[best_candidate] + log_probabilities[frame, token_id]
            )
            backpointers[frame, state_index] = previous_state
        previous = current

    final_candidates = [state_count - 1]
    if state_count > 1:
        final_candidates.append(state_count - 2)
    final_state = max(final_candidates, key=lambda state: float(previous[state]))
    if not math.isfinite(float(previous[final_state])):
        raise ValueError("Audio is too short to align the expected phone sequence")

    state_path = [final_state]
    for frame in range(frame_count - 1, 0, -1):
        final_state = int(backpointers[frame, final_state])
        if final_state < 0:
            raise ValueError("Could not backtrack the CTC alignment")
        state_path.append(final_state)
    state_path.reverse()

    target_frames: List[List[int]] = [[] for _ in target_ids]
    for frame, state_index in enumerate(state_path):
        if state_index % 2 == 1:
            target_frames[(state_index - 1) // 2].append(frame)

    return CTCAlignment(target_frames=target_frames, state_path=state_path)


class PhonemeAssessor:
    """Lazy-loaded local acoustic phoneme assessor."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or os.getenv("PHONEME_MODEL_ID", DEFAULT_MODEL_ID)
        self.enabled = _is_enabled(os.getenv("ENABLE_PHONEME_ASSESSMENT", "1"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_extractor = None
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None or not self.enabled:
            return
        with self._load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                from transformers import (
                    AutoFeatureExtractor,
                    AutoModelForCTC,
                    Wav2Vec2PhonemeCTCTokenizer,
                )

                self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.model_id
                )
                # do_phonemize=False avoids an external eSpeak runtime. Expected
                # IPA is tokenized against the model vocabulary below.
                self._tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    self.model_id, do_phonemize=False
                )
                self._model = AutoModelForCTC.from_pretrained(self.model_id)
                self._model.to(self.device)
                self._model.eval()
            except Exception as error:  # fallback is part of the public contract
                self._load_error = f"{type(error).__name__}: {error}"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _vocabulary_tokens(self) -> Tuple[Dict[str, int], set[int]]:
        vocabulary = self._tokenizer.get_vocab()
        special_ids = set(self._tokenizer.all_special_ids)
        tokens = {
            token: token_id
            for token, token_id in vocabulary.items()
            if token_id not in special_ids and token.strip()
        }
        return tokens, special_ids

    @staticmethod
    def _tokenize_with_vocabulary(
        ipa: str, vocabulary: Dict[str, int]
    ) -> Tuple[List[str], List[str]]:
        """Longest-match IPA tokenization using the acoustic model labels."""

        normalized = unicodedata.normalize("NFC", ipa or "").lower()
        normalized = normalized.replace("g", "ɡ").replace("͡", "").replace("͜", "")
        ignored = set(" /[](),.!?;:\"“”‘’ˈˌ")
        candidates = sorted(vocabulary, key=len, reverse=True)
        result: List[str] = []
        unmatched: List[str] = []
        index = 0
        while index < len(normalized):
            if normalized[index].isspace() or normalized[index] in ignored:
                index += 1
                continue
            match = next(
                (
                    token
                    for token in candidates
                    if normalized.startswith(token, index)
                ),
                None,
            )
            if match is None:
                unmatched.append(normalized[index])
                index += 1
            else:
                result.append(match)
                index += len(match)
        return result, unmatched

    @staticmethod
    def _word_times(
        word_index: int, word_count: int, duration: float
    ) -> Tuple[float, float]:
        start = duration * word_index / max(word_count, 1)
        end = duration * (word_index + 1) / max(word_count, 1)
        return start, end

    def assess(
        self,
        audio: torch.Tensor,
        reference_text: str,
        ipa_converter,
        fallback_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        word_start_times: Optional[Sequence[float]] = None,
        word_end_times: Optional[Sequence[float]] = None,
    ) -> dict:
        self._load()
        if self._model is None:
            reason = (
                "Acoustic phoneme assessment is disabled"
                if not self.enabled
                else self._load_error or "Acoustic model is unavailable"
            )
            return self._fallback(
                reference_text,
                ipa_converter,
                fallback_pairs,
                word_start_times,
                word_end_times,
                reason,
            )

        try:
            return self._assess_acoustically(audio, reference_text, ipa_converter)
        except Exception as error:
            return self._fallback(
                reference_text,
                ipa_converter,
                fallback_pairs,
                word_start_times,
                word_end_times,
                f"Acoustic assessment failed: {type(error).__name__}: {error}",
            )

    def _assess_acoustically(self, audio, reference_text, ipa_converter) -> dict:
        words = reference_text.split()
        vocabulary, special_ids = self._vocabulary_tokens()
        expected_tokens: List[str] = []
        word_token_ranges: List[Tuple[int, int, str, str, List[str]]] = []
        all_unmatched: List[str] = []

        for word in words:
            ipa = ipa_converter.convertToPhonem(word)
            tokens, unmatched = self._tokenize_with_vocabulary(ipa, vocabulary)
            start = len(expected_tokens)
            expected_tokens.extend(tokens)
            word_token_ranges.append((start, len(expected_tokens), word, ipa, unmatched))
            all_unmatched.extend(unmatched)

        if not expected_tokens:
            raise ValueError("Reference text produced no phones supported by the model")

        target_ids = [vocabulary[token] for token in expected_tokens]
        audio_array = audio.detach().cpu().float().numpy().reshape(-1)
        duration = len(audio_array) / SAMPLE_RATE
        inputs = self._feature_extractor(
            audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            logits = self._model(
                input_values, attention_mask=attention_mask
            ).logits[0]
            log_probabilities = torch.log_softmax(logits, dim=-1)

        blank_id = int(self._model.config.pad_token_id)
        alignment = ctc_forced_alignment(log_probabilities, target_ids, blank_id)
        frame_count = int(log_probabilities.shape[0])
        blocked_ids = set(special_ids) | {blank_id}

        # Decode the audio without constraining it to the reference.  This is
        # the actual phone hypothesis shown to the learner. Forced alignment is
        # retained only for boundaries and acoustic posterior/GOP evidence.
        greedy_ids = torch.argmax(log_probabilities, dim=-1).tolist()
        recognized_tokens: List[str] = []
        previous_id = None
        for token_id in greedy_ids:
            if token_id == previous_id:
                continue
            previous_id = token_id
            if token_id in blocked_ids:
                continue
            recognized_tokens.append(self._tokenizer.convert_ids_to_tokens(token_id))

        freely_aligned = levenshtein_alignment(expected_tokens, recognized_tokens)
        freely_observed = [observed for _, observed in freely_aligned]

        phone_results: List[dict] = []
        for index, (expected, frames) in enumerate(
            zip(expected_tokens, alignment.target_frames)
        ):
            if not frames:
                phone_results.append(
                    {
                        "expected": expected,
                        "observed": None,
                        "start": None,
                        "end": None,
                        "score": 0,
                        "gop": None,
                        "status": "deletion",
                    }
                )
                continue

            segment = log_probabilities[frames]
            mean_log_probabilities = segment.mean(dim=0)
            competitor_scores = mean_log_probabilities.clone()
            for token_id in blocked_ids:
                if 0 <= token_id < competitor_scores.shape[0]:
                    competitor_scores[token_id] = float("-inf")
            observed_id = int(torch.argmax(competitor_scores).item())
            expected_id = target_ids[index]
            competing = competitor_scores.clone()
            competing[expected_id] = float("-inf")
            best_competing_score = float(torch.max(competing).item())
            gop = float(mean_log_probabilities[expected_id].item()) - best_competing_score
            posterior_score = 100.0 / (
                1.0 + math.exp(-max(-20.0, min(20.0, gop)))
            )
            observed = freely_observed[index]
            acoustic_candidate = self._tokenizer.convert_ids_to_tokens(observed_id)

            # Raw GOP is not a calibrated percentage. Preserve it for future
            # calibration, but make the user-facing score primarily reflect
            # the freely decoded phone. An exact acoustic decode should not be
            # presented as a failing 50-ish score merely because GOP ~= 0.
            if observed is None:
                score = 0
                status = "deletion"
            elif observed == expected:
                score = int(round(85.0 + 15.0 * posterior_score / 100.0))
                status = "correct"
            else:
                score = int(round(min(50.0, posterior_score * 0.5)))
                status = "substitution"

            phone_results.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "acoustic_candidate": acoustic_candidate,
                    "start": round(min(frames) * duration / frame_count, 3),
                    "end": round((max(frames) + 1) * duration / frame_count, 3),
                    "score": score,
                    "posterior_score": round(posterior_score, 2),
                    "gop": round(gop, 4),
                    "status": status,
                }
            )

        word_results: List[dict] = []
        for start, end, word, ipa, unmatched in word_token_ranges:
            phones = phone_results[start:end]
            scores = [phone["score"] for phone in phones]
            starts = [phone["start"] for phone in phones if phone["start"] is not None]
            ends = [phone["end"] for phone in phones if phone["end"] is not None]
            word_results.append(
                {
                    "word": word,
                    "expected_ipa": ipa,
                    "observed_ipa": "".join(
                        phone["observed"] or "" for phone in phones
                    ),
                    "start": min(starts) if starts else None,
                    "end": max(ends) if ends else None,
                    "score": int(round(float(np.mean(scores)))) if scores else 0,
                    "phonemes": phones,
                    "unmatched_ipa_symbols": unmatched,
                }
            )

        overall_scores = [phone["score"] for phone in phone_results]
        recognized_phones_by_word = [
            word["observed_ipa"] or "∅" for word in word_results
        ]
        return {
            "method": "acoustic_ctc_gop",
            "model": self.model_id,
            "calibrated": False,
            "overall_score": int(round(float(np.mean(overall_scores)))),
            "recognized_phones": " ".join(recognized_tokens),
            "recognized_phones_by_word": recognized_phones_by_word,
            "recognized_phones_grouped": " | ".join(recognized_phones_by_word),
            "words": word_results,
            "warnings": (
                ["Some IPA symbols were not present in the acoustic model vocabulary"]
                if all_unmatched
                else []
            ),
        }

    def _fallback(
        self,
        reference_text: str,
        ipa_converter,
        fallback_pairs: Optional[Sequence[Tuple[str, str]]],
        word_start_times: Optional[Sequence[float]],
        word_end_times: Optional[Sequence[float]],
        reason: str,
    ) -> dict:
        words = reference_text.split()
        pairs = list(fallback_pairs or [])
        word_results = []

        for index, word in enumerate(words):
            expected_ipa = ipa_converter.convertToPhonem(word)
            observed_ipa = pairs[index][1] if index < len(pairs) else ""
            aligned = levenshtein_alignment(
                ipa_units(expected_ipa), ipa_units(observed_ipa)
            )
            start = (
                word_start_times[index]
                if word_start_times and index < len(word_start_times)
                else None
            )
            end = (
                word_end_times[index]
                if word_end_times and index < len(word_end_times)
                else None
            )
            phones = []
            phone_count = max(len(aligned), 1)
            for phone_index, (expected, observed) in enumerate(aligned):
                exact = expected == observed
                phone_start = None
                phone_end = None
                if start is not None and end is not None:
                    phone_start = round(start + (end - start) * phone_index / phone_count, 3)
                    phone_end = round(
                        start + (end - start) * (phone_index + 1) / phone_count, 3
                    )
                phones.append(
                    {
                        "expected": expected,
                        "observed": observed,
                        "start": phone_start,
                        "end": phone_end,
                        "score": 100 if exact else (0 if observed is None else 35),
                        "gop": None,
                        "status": (
                            "correct"
                            if exact
                            else "deletion" if observed is None else "substitution"
                        ),
                    }
                )
            scores = [phone["score"] for phone in phones]
            word_results.append(
                {
                    "word": word,
                    "expected_ipa": expected_ipa,
                    "observed_ipa": observed_ipa,
                    "start": start,
                    "end": end,
                    "score": int(round(float(np.mean(scores)))) if scores else 0,
                    "phonemes": phones,
                    "unmatched_ipa_symbols": [],
                }
            )

        scores = [word["score"] for word in word_results]
        recognized_phones_by_word = [
            word["observed_ipa"] or "∅" for word in word_results
        ]
        return {
            "method": "transcript_ipa_fallback",
            "model": None,
            "calibrated": False,
            "overall_score": int(round(float(np.mean(scores)))) if scores else 0,
            "recognized_phones": " ".join(
                phone
                for word in word_results
                for phone in ipa_units(word["observed_ipa"])
            ),
            "recognized_phones_by_word": recognized_phones_by_word,
            "recognized_phones_grouped": " | ".join(recognized_phones_by_word),
            "words": word_results,
            "warnings": [reason, "Fallback scores are not acoustic phoneme scores"],
        }


_shared_assessor: Optional[PhonemeAssessor] = None
_shared_lock = threading.Lock()


def get_phoneme_assessor() -> PhonemeAssessor:
    global _shared_assessor
    if _shared_assessor is None:
        with _shared_lock:
            if _shared_assessor is None:
                _shared_assessor = PhonemeAssessor()
    return _shared_assessor
