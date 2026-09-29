import unittest
from types import SimpleNamespace

import torch

from phoneme_assessment import (
    PhonemeAssessor,
    ctc_forced_alignment,
    ipa_units,
    levenshtein_alignment,
)


class DummyConverter:
    values = {"three": "θriː", "cats": "kæts"}

    def convertToPhonem(self, value):
        return self.values.get(value, value)


class FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_values=torch.tensor(audio).unsqueeze(0))


class FakeTokenizer:
    all_special_ids = [0]
    vocabulary = {"<pad>": 0, "θ": 1, "r": 2, "t": 3}

    def get_vocab(self):
        return self.vocabulary

    def convert_ids_to_tokens(self, token_id):
        return {0: "<pad>", 1: "θ", 2: "r", 3: "t"}[token_id]

    def batch_decode(self, token_ids):
        return ["θ r"]


class FakeAcousticModel:
    config = SimpleNamespace(pad_token_id=0)

    def __call__(self, input_values, attention_mask=None):
        logits = torch.full((1, 5, 3), -10.0)
        logits[0, 0, 0] = 5.0
        logits[0, 1, 1] = 5.0
        logits[0, 2, 0] = 5.0
        logits[0, 3, 2] = 5.0
        logits[0, 4, 0] = 5.0
        return SimpleNamespace(logits=logits)


class FakeSubstitutionModel:
    config = SimpleNamespace(pad_token_id=0)

    def __call__(self, input_values, attention_mask=None):
        logits = torch.full((1, 5, 4), -10.0)
        logits[0, 0, 0] = 5.0
        logits[0, 1, 3] = 5.0  # free decode hears /t/, not target /θ/
        logits[0, 2, 0] = 5.0
        logits[0, 3, 2] = 5.0
        logits[0, 4, 0] = 5.0
        return SimpleNamespace(logits=logits)


class TestPhonemeUtilities(unittest.TestCase):
    def test_ipa_units_keep_modifiers_and_tied_affricates_together(self):
        self.assertEqual(ipa_units("ˈt͡ʃiː"), ["t͡ʃ", "iː"])

    def test_levenshtein_alignment_reports_substitution_and_deletion(self):
        result = levenshtein_alignment(["θ", "r", "iː"], ["t", "r"])
        self.assertEqual(result, [("θ", "t"), ("r", "r"), ("iː", None)])

    def test_ctc_alignment_finds_frames_for_each_target(self):
        emissions = torch.full((5, 3), -10.0)
        emissions[0, 0] = 0.0
        emissions[1, 1] = 0.0
        emissions[2, 0] = 0.0
        emissions[3, 2] = 0.0
        emissions[4, 0] = 0.0

        alignment = ctc_forced_alignment(emissions, [1, 2], blank_id=0)

        self.assertIn(1, alignment.target_frames[0])
        self.assertIn(3, alignment.target_frames[1])

    def test_model_vocabulary_tokenization_prefers_complete_phone(self):
        tokens, unmatched = PhonemeAssessor._tokenize_with_vocabulary(
            "ˈt͡ʃiː", {"t": 1, "ʃ": 2, "tʃ": 3, "i": 4, "iː": 5}
        )
        self.assertEqual(tokens, ["tʃ", "iː"])
        self.assertEqual(unmatched, [])

    def test_disabled_model_returns_labelled_fallback(self):
        assessor = PhonemeAssessor()
        assessor.enabled = False
        result = assessor.assess(
            torch.zeros(1, 16000),
            "three cats",
            DummyConverter(),
            fallback_pairs=[("θriː", "triː"), ("kæts", "kæts")],
            word_start_times=[0.0, 0.5],
            word_end_times=[0.5, 1.0],
        )

        self.assertEqual(result["method"], "transcript_ipa_fallback")
        self.assertEqual(len(result["words"]), 2)
        self.assertEqual(result["words"][0]["phonemes"][0]["status"], "substitution")

    def test_acoustic_result_has_phone_scores_and_timestamps(self):
        assessor = PhonemeAssessor(model_id="fake-model")
        assessor._feature_extractor = FakeFeatureExtractor()
        assessor._tokenizer = FakeTokenizer()
        assessor._model = FakeAcousticModel()

        result = assessor.assess(
            torch.zeros(1, 16000), "θr", DummyConverter()
        )

        self.assertEqual(result["method"], "acoustic_ctc_gop")
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["words"][0]["phonemes"][0]["observed"], "θ")
        self.assertIsNotNone(result["words"][0]["phonemes"][0]["start"])
        self.assertEqual(result["recognized_phones_by_word"], ["θr"])
        self.assertEqual(result["recognized_phones_grouped"], "θr")

    def test_observed_phone_comes_from_free_acoustic_decode(self):
        assessor = PhonemeAssessor(model_id="fake-model")
        assessor._feature_extractor = FakeFeatureExtractor()
        assessor._tokenizer = FakeTokenizer()
        assessor._model = FakeSubstitutionModel()

        result = assessor.assess(
            torch.zeros(1, 16000), "θr", DummyConverter()
        )

        first_phone = result["words"][0]["phonemes"][0]
        self.assertEqual(result["recognized_phones"], "t r")
        self.assertEqual(result["recognized_phones_by_word"], ["tr"])
        self.assertEqual(first_phone["expected"], "θ")
        self.assertEqual(first_phone["observed"], "t")
        self.assertEqual(first_phone["status"], "substitution")
        self.assertLessEqual(first_phone["score"], 50)


if __name__ == "__main__":
    unittest.main()
