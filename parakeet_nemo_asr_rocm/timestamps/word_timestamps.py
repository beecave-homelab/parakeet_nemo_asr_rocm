"""Utilities for extracting word-level timestamps from NeMo ASR hypotheses."""

from typing import List

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

from parakeet_nemo_asr_rocm.timestamps.models import Word


def get_word_timestamps(
    hypotheses: List[Hypothesis],
    model: ASRModel,
    time_stride: float | None = None,
) -> List[Word]:
    """Calculate word-level timestamps from a Transducer model's hypotheses.

    Args:
        hypotheses: A list of Hypothesis objects from NeMo.
        model: The ASR model instance, used to access the tokenizer.
        time_stride: The time duration of a single frame shift.

    Returns:
        A list of Word objects with calculated timestamps.

    """
    all_words: List[Word] = []
    # SentencePiece-based tokenizers (used by NeMo ASR models) encode the beginning
    # of a new word with a leading "▁" character.  QuartzNet-style char tokenizers
    # sometimes expose `tokenizer.space`, but this attribute is not present on
    # `SentencePieceTokenizer`.  Hence we detect word boundaries based on this
    # leading marker instead of relying on a dedicated space token.

    for hypo in hypotheses:
        if not hasattr(hypo, "y_sequence") or not hasattr(hypo, "timestamp"):
            continue

        # Get the token IDs from the hypothesis
        token_ids = hypo.y_sequence.detach().cpu().numpy()
        # Get the timestamps for each token
        timestamps_raw = hypo.timestamp.detach().cpu().numpy().astype(float)
        if time_stride is not None:
            timestamps = timestamps_raw * time_stride
        else:
            timestamps = timestamps_raw

        words_for_hypo = []
        current_word = []
        word_start_time = -1

        for i, token_id_np in enumerate(token_ids):
            token_id = int(token_id_np)  # ensure native int for SentencePiece SWIG
            token_text = model.tokenizer.ids_to_tokens([token_id])[0]
            time = timestamps[i] + getattr(hypo, "start_offset", 0.0)

            # Detect start of a new word. SentencePiece denotes it via leading '▁'.
            is_word_start = token_text.startswith("▁")

            if is_word_start and current_word:
                # Finish previous word
                word_text = model.tokenizer.ids_to_text(current_word)
                words_for_hypo.append(
                    Word(
                        word=word_text.lstrip("▁"),
                        start=word_start_time,
                        end=time,
                        score=None,
                    )
                )
                current_word = []
                word_start_time = time  # new word starts now

            if not current_word:
                word_start_time = time

            current_word.append(token_id)

        # Add the last word if any
        if current_word:
            word_text = model.tokenizer.ids_to_text(current_word)
            end_time = timestamps[-1] + getattr(hypo, "start_offset", 0.0)
            words_for_hypo.append(
                Word(
                    word=word_text.lstrip("▁"),
                    start=word_start_time,
                    end=end_time,
                    score=None,
                )
            )

        all_words.extend(words_for_hypo)

    # Post-process to remove duplicates arising from overlapping chunks.
    if not all_words:
        return []

    all_words.sort(key=lambda w: w.start)
    deduped: List[Word] = []
    last_end = -1.0
    MIN_GAP = 0.03  # 30 ms tolerance for overlap

    for w in all_words:
        if w.start < last_end - MIN_GAP:
            # This word is (almost) entirely contained in the previous window; skip it.
            continue
        deduped.append(w)
        last_end = w.end

    return deduped
