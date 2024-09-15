import os
import logging
import re
import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
    write_srt,
)
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
import csv


class DiarizePipeline:
    def __init__(self, result_file_path, mtypes):
        self.mtypes = mtypes
        self.result_file_path = result_file_path
        self.last_speaker = None
        self.last_segment = None

    def __call__(
        self,
        audio,
        stemming=True,
        suppress_numerals=False,
        model_name="medium.en",
        batch_size=8,
        language=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path="MODEL",
        title="",
        nth_output_file=0,
    ):

        # Create a new output file if it doesn't exist
        if not os.path.exists(f"{self.result_file_path}/dataset_{nth_output_file}.csv"):
            with open(
                f"{self.result_file_path}/dataset_{nth_output_file}.csv",
                mode="w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["speaker", "text", "title", "start_time", "end_time"],
                )
                writer.writeheader()

        # Source Separation (Stemming)
        if stemming:
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "temp_outputs"'
            )
            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
                )
                vocal_target = audio
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio

        # Transcription
        from transcription_helpers import transcribe_batched

        whisper_results, language, audio_waveform = transcribe_batched(
            vocal_target,
            language,
            batch_size,
            model_name,
            self.mtypes[device],
            suppress_numerals,
            device,
            model_path,
        )

        # Forced Alignment
        alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        audio_waveform = (
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device)
        )
        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=batch_size
        )

        del alignment_model
        torch.cuda.empty_cache()

        full_transcript = "".join(segment["text"] for segment in whisper_results)

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[language],
        )

        segments, scores, blank_id = get_alignments(
            emissions,
            tokens_starred,
            alignment_dictionary,
        )

        spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Convert audio to mono for NeMo compatibility
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs")
        os.makedirs(temp_path, exist_ok=True)
        torchaudio.save(
            os.path.join(temp_path, "mono_file.wav"),
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # Speaker Diarization
        msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
        msdd_model.diarize()
        del msdd_model
        torch.cuda.empty_cache()

        # Reading timestamps <> Speaker Labels mapping
        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # Punctuation Restoration
        if language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], wsm))
            labeled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labeled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            logging.warning(
                f"Punctuation restoration is not available for {language} language. Using the original punctuation."
            )

        # Realign and Generate Outputs
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        with open(
            f"{self.result_file_path}/{self.clean_title(title)}.txt",
            "w",
            encoding="utf-8-sig",
        ) as f:
            get_speaker_aware_transcript(ssm, f)

        with open(
            f"{self.result_file_path}/{self.clean_title(title)}.srt",
            "w",
            encoding="utf-8-sig",
        ) as srt:
            write_srt(ssm, srt)

        self.append_data(ssm, title, nth_output_file)
        cleanup(temp_path)

    # Helper method to clean titles for filenames
    def clean_title(self, title):
        replacements = {
            "<": "-",
            ">": "-",
            ":": "-",
            '"': "'",
            "/": "-",
            "\\": "-",
            "|": "-",
            "?": "-",
            "*": "-",
            " ": "_",
        }
        for old, new in replacements.items():
            title = title.replace(old, new)
        return title

    # Method to append data to CSV
    def append_data(self, result_segments, title, nth_output_file):
        with open(
            f"{self.result_file_path}/dataset_{nth_output_file}.csv",
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=["speaker", "text", "title", "start_time", "end_time"]
            )
            print(f"SEGMENTS:\n\n {result_segments}")
            print(f"TITLE:\n\n {title}")

            first_speaker_id = None

            for segment in result_segments:
                speaker = segment.get("speaker")

                if speaker is None:
                    segment["speaker"] = self.last_speaker
                    print(
                        "Warning: 'speaker' key is missing in the segment, inserting last speaker."
                    )

                elif first_speaker_id is None:
                    first_speaker_id = segment["speaker"]

                if first_speaker_id in segment["speaker"]:
                    segment["speaker"] = "user"
                else:
                    segment["speaker"] = "assistant"

                # Ensure the segment has all required fields
                if all(
                    key in segment
                    for key in ["speaker", "text", "start_time", "end_time"]
                ):
                    text_without_quotes = segment["text"].replace('"', "")
                    print(text_without_quotes)
                    print(self.last_segment)
                    print(self.last_speaker)
                    current_speaker = segment["speaker"]

                    # Check if the current speaker is the same as the last one
                    if self.last_speaker == current_speaker:
                        # Append text to the previous entry and update the end time
                        self.last_segment["text"] += text_without_quotes
                        self.last_segment["end_time"] = segment["end_time"]
                    else:
                        # Write the last segment if it exists
                        if self.last_segment:
                            writer.writerow(self.last_segment)

                        # Update the last speaker and last segment
                        self.last_speaker = current_speaker
                        self.last_segment = {
                            "speaker": current_speaker,
                            "text": text_without_quotes,
                            "title": title,
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                        }
                else:
                    print("Missing key in segment, skipping entry...")

            # Write the last segment after the loop
            if self.last_segment:
                writer.writerow(self.last_segment)

            # Reset last speaker and segment for future calls
            self.last_speaker = None
            self.last_segment = None
