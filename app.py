# app.py
import streamlit as st
import pandas as pd
import os
import csv
import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI
import zipfile
import shutil
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import gdown

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path: Path, output_path: Path):
    """音声をWAV（PCM、16-bit、16kHz）に変換"""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(output_path, format="wav")
    return output_path

def download_audio_from_youtube(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "128k", "-o", str(out_path), url]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(out_path, wav_path)

def download_from_google_drive(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / f"{uuid.uuid4().hex}.mp4"
    gdown.download(url, str(out_path), quiet=False)
    audio_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    video = VideoFileClip(str(out_path))
    video.audio.write_audiofile(str(audio_path))
    video.close()
    out_path.unlink(missing_ok=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(audio_path, wav_path)

def extract_audio_from_file(uploaded_file, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    file_ext = uploaded_file.name.split('.')[-1].lower()
    out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
    temp_path = out_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    if file_ext == 'mp3':
        shutil.copy(temp_path, out_path)
    elif file_ext == 'mp4':
        video = VideoFileClip(str(temp_path))
        video.audio.write_audiofile(str(out_path))
        video.close()
    else:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"Unsupported file format: {file_ext}")
    temp_path.unlink(missing_ok=True)
    wav_path = out_dir / f"{uuid.uuid4().hex}.wav"
    return convert_to_wav(out_path, wav_path)

def azure_speech_to_text(audio_path: Path, region: str, key: str) -> str:
    """音声認識で自動書き起こし（目標テキストなし用）"""
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech could be recognized.")
    return result.text

def azure_pronunciation_assess(audio_path: Path, region: str, key: str, target_text: Optional[str] = None) -> Dict[str, Any]:
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    if not target_text:
        target_text = azure_speech_to_text(audio_path, region, key)
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=target_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)
    pronunciation_config.apply_to(recognizer)
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech could be recognized.")
    pron_result = speechsdk.PronunciationAssessmentResult(result)
    return {
        "asr_text": result.text,
        "accuracy": pron_result.accuracy_score,
        "fluency": pron_result.fluency_score,
        "prosody": pron_result.prosody_score,
        "completeness": pron_result.completeness_score,
        "raw": json.loads(result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
    }

def openai_feedback(asr_text: str, target_text: str, azure_summary: Dict[str, Any], config: Dict[str, Any], no_target_text: bool = False) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "（注）OPENAI_API_KEY が未設定のため、AI所見はスキップしました。"
    client = OpenAI(api_key=api_key)
    if no_target_text:
        prompt = config["gpt_prompt_no_target"].format(
            asr_text=asr_text,
            accuracy=azure_summary.get("accuracy"),
            fluency=azure_summary.get("fluency"),
            prosody=azure_summary.get("prosody")
        )
    else:
        prompt = config["gpt_prompt"].format(
            asr_text=asr_text,
            target_text=target_text,
            accuracy=azure_summary.get("accuracy"),
            fluency=azure_summary.get("fluency"),
            prosody=azure_summary.get("prosody")
        )
    response = client.chat.completions.create(
        model=config["openai"]["model"],
        temperature=config["openai"]["temperature"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Japanese university English speaking assessment."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def weighted_score(az: Dict[str, Any], content_org: int, vocab_gram: int, weights: Dict[str, int]) -> float:
    total = 0.0
    total += az.get("accuracy", 0) * (weights.get("pronunciation_accuracy", 0) / 100.0)
    total += az.get("fluency", 0) * (weights.get("fluency", 0) / 100.0)
    total += az.get("prosody", 0) * (weights.get("prosody", 0) / 100.0)
    total += content_org * (weights.get("content_organization", 0) / 100.0)
    total += vocab_gram * (weights.get("vocabulary_grammar", 0) / 100.0)
    return round(total, 1)

def band_from_score(score: float, bands: Dict[str, Dict[str, Any]]) -> str:
    best = "D"
    best_min = -1
    for b, cfg in bands.items():
        if score >= cfg["min"] and cfg["min"] > best_min:
            best, best_min = b, cfg["min"]
    return best

def process_single_input(input_type, input_value, target_text: Optional[str] = None, config_path="config.yaml"):
    cfg = load_config(config_path)
    weights = cfg["weights"]
    region = os.getenv("AZURE_SPEECH_REGION", "")
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not region or not key:
        st.error("エラー: AZURE_SPEECH_REGION または AZURE_SPEECH_KEY が設定されていません。")
        return None

    downloads_dir = Path(cfg.get("downloads_dir", "./downloads"))
    ensure_dir(downloads_dir)

    bands = {
        "A": {"min": 85, "label_ja": "A（到達目標を十分達成）"},
        "B": {"min": 70, "label_ja": "B（概ね達成）"},
        "C": {"min": 55, "label_ja": "C（一部達成）"},
        "D": {"min": 0,  "label_ja": "D（要改善）"}
    }

    try:
        if input_type == "youtube":
            audio_path = download_audio_from_youtube(input_value, downloads_dir)
        elif input_type == "google_drive":
            audio_path = download_from_google_drive(input_value, downloads_dir)
        elif input_type == "file":
            audio_path = extract_audio_from_file(input_value, downloads_dir)
        az = azure_pronunciation_assess(audio_path, region, key, target_text)
        content_org = 70
        vocab_gram = 70
        feedback = openai_feedback(az.get("asr_text", ""), target_text if target_text else az.get("asr_text", ""), az, cfg, no_target_text=(target_text is None))
        total = weighted_score(az, content_org, vocab_gram, weights)
        band = band_from_score(total, bands)
        return {
            "score_total": total,
            "band": band,
            "accuracy": az["accuracy"],
            "fluency": az["fluency"],
            "prosody": az["prosody"],
            "comment": feedback
        }
    except Exception as e:
        st.error(f"処理失敗: {str(e)}")
        return None

def process_submissions(input_type, csv_file=None, uploaded_files=None, target_texts=None, config_path="config.yaml"):
    cfg = load_config(config_path)
    weights = cfg["weights"]
    region = os.getenv("AZURE_SPEECH_REGION", "")
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not region or not key:
        st.error("エラー: AZURE_SPEECH_REGION または AZURE_SPEECH_KEY が設定されていません。")
        return None, None, None

    downloads_dir = Path(cfg.get("downloads_dir", "./downloads"))
    feedbacks_dir = Path(cfg.get("feedbacks_dir", "./feedbacks"))
    ensure_dir(downloads_dir)
    ensure_dir(feedbacks_dir)

    bands = {
        "A": {"min": 85, "label_ja": "A（到達目標を十分達成）"},
        "B": {"min": 70, "label_ja": "B（概ね達成）"},
        "C": {"min": 55, "label_ja": "C（一部達成）"},
        "D": {"min": 0,  "label_ja": "D（要改善）"}
    }

    results = []
    if input_type == "youtube":
        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            sid = str(row.get("student_id", "")).strip()
            sname = str(row.get("student_name", "")).strip()
            url = str(row.get("youtube_url", "")).strip()
            target_text = str(row.get("target_text", "")).strip() if "target_text" in row else None
            try:
                audio_path = download_audio_from_youtube(url, downloads_dir)
                az = azure_pronunciation_assess(audio_path, region, key, target_text)
                content_org = 70
                vocab_gram = 70
                feedback = openai_feedback(az.get("asr_text", ""), target_text if target_text else az.get("asr_text", ""), az, cfg, no_target_text=(target_text is None))
                total = weighted_score(az, content_org, vocab_gram, weights)
                band = band_from_score(total, bands)
                fb_path = feedbacks_dir / f"{sid}.txt"
                with open(fb_path, "w", encoding="utf-8") as fb:
                    fb.write(f"# {sname} ({sid})\n")
                    fb.write(f"- Accuracy: {az['accuracy']}  Fluency: {az['fluency']}  Prosody: {az['prosody']}\n")
                    fb.write(f"- Total: {total}  Band: {band}\n\n")
                    fb.write("=== Feedback ===\n")
                    fb.write(feedback + "\n")
                results.append({
                    "student_id": sid,
                    "student_name": sname,
                    "score_total": total,
                    "band": band,
                    "accuracy": az["accuracy"],
                    "fluency": az["fluency"],
                    "prosody": az["prosody"],
                    "comment": feedback.replace("\n", " ")
                })
            except Exception as e:
                results.append({
                    "student_id": sid,
                    "student_name": sname,
                    "score_total": "",
                    "band": "ERR",
                    "accuracy": "",
                    "fluency": "",
                    "prosody": "",
                    "comment": f"(処理失敗) {str(e)}"
                })

    elif input_type == "files":
        for idx, uploaded_file in enumerate(uploaded_files):
            sid = f"student_{idx+1}" if not uploaded_file.name.split('.')[0].isdigit() else uploaded_file.name.split('.')[0]
            sname = uploaded_file.name.split('.')[0]
            target_text = target_texts[idx] if target_texts and idx < len(target_texts) and target_texts[idx] else None
            try:
                audio_path = extract_audio_from_file(uploaded_file, downloads_dir)
                az = azure_pronunciation_assess(audio_path, region, key, target_text)
                content_org = 70
                vocab_gram = 70
                feedback = openai_feedback(az.get("asr_text", ""), target_text if target_text else az.get("asr_text", ""), az, cfg, no_target_text=(target_text is None))
                total = weighted_score(az, content_org, vocab_gram, weights)
                band = band_from_score(total, bands)
                fb_path = feedbacks_dir / f"{sid}.txt"
                with open(fb_path, "w", encoding="utf-8") as fb:
                    fb.write(f"# {sname} ({sid})\n")
                    fb.write(f"- Accuracy: {az['accuracy']}  Fluency: {az['fluency']}  Prosody: {az['prosody']}\n")
                    fb.write(f"- Total: {total}  Band: {band}\n\n")
                    fb.write("=== Feedback ===\n")
                    fb.write(feedback + "\n")
                results.append({
                    "student_id": sid,
                    "student_name": sname,
                    "score_total": total,
                    "band": band,
                    "accuracy": az["accuracy"],
                    "fluency": az["fluency"],
                    "prosody": az["prosody"],
                    "comment": feedback.replace("\n", " ")
                })
            except Exception as e:
                results.append({
                    "student_id": sid,
                    "student_name": sname,
                    "score_total": "",
                    "band": "ERR",
                    "accuracy": "",
                    "fluency": "",
                    "prosody": "",
                    "comment": f"(処理失敗) {str(e)}"
                })

    out_csv = Path("results.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["student_id", "student_name", "score_total", "band", "accuracy", "fluency", "prosody", "comment"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    moodle_csv = Path("grading_worksheet.csv")
    moodle_rows = [
        {
            "Identifier": r["student_id"],
            "Full name": r["student_name"],
            "Grade": r["score_total"],
            "Feedback comments": r["comment"]
        } for r in results
    ]
    with open(moodle_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Identifier", "Full name", "Grade", "Feedback comments"])
        writer.writeheader()
        writer.writerows(moodle_rows)

    zip_path = Path("feedbacks.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fb_file in feedbacks_dir.glob("*.txt"):
            zipf.write(fb_file, fb_file.relative_to(fb_file.parent))

    return out_csv, moodle_csv, zip_path

# Streamlitインターフェース
st.title("英語音読・スピーキング評価ツール（拡張版）")
st.write("CSV/リンク/ファイルを入力して評価します。Moodle連携可能。目標テキストは任意。")
st.write("**同意テンプレートをMoodle課題説明に記載してください**: 本課題の音声は、クラウドの音声認識/発音評価API（Azure）に送信して自動評価します。生成AI（OpenAI）により日本語講評が作成されます。最終成績は教員が確認後確定します。")

input_type = st.radio("入力方法を選択", ("YouTubeリンク", "Google Driveリンク", "MP3/MP4ファイル", "CSV (複数)"))

target_text = st.text_area("目標テキストを入力（任意）", placeholder="目標テキストがなくても評価可能です")

if input_type == "YouTubeリンク":
    url = st.text_input("YouTube限定公開リンクを貼り付け")
    if url and st.button("評価を実行"):
        with st.spinner("処理中..."):
            result = process_single_input("youtube", url, target_text if target_text else None)
            if result:
                st.success("評価完了！")
                st.write(result)

elif input_type == "Google Driveリンク":
    url = st.text_input("Google Drive共有リンクを貼り付け")
    if url and st.button("評価を実行"):
        with st.spinner("処理中..."):
            result = process_single_input("google_drive", url, target_text if target_text else None)
            if result:
                st.success("評価完了！")
                st.write(result)

elif input_type == "MP3/MP4ファイル":
    uploaded_file = st.file_uploader("ファイルをアップロード", type=["mp3", "mp4"])
    if uploaded_file and st.button("評価を実行"):
        with st.spinner("処理中..."):
            result = process_single_input("file", uploaded_file, target_text if target_text else None)
            if result:
                st.success("評価完了！")
                st.write(result)

elif input_type == "CSV (複数)":
    st.write("Moodleの課題提出からエクスポートしたCSV（学生ID、名前、YouTubeリンク、目標テキスト（任意））をアップロード。")
    uploaded_csv = st.file_uploader("CSVをアップロード", type=["csv"])
    if uploaded_csv:
        st.write("CSVを読み込みました。以下のボタンを押して評価を開始します。")
        if st.button("評価を実行（CSV）"):
            with st.spinner("処理中...（数分かかる場合があります）"):
                temp_csv = Path("temp_submissions.csv")
                with open(temp_csv, "wb") as f:
                    f.write(uploaded_csv.getvalue())
                results_csv, moodle_csv, feedback_zip = process_submissions("youtube", csv_file=temp_csv)
                if results_csv and moodle_csv:
                    st.success("評価が完了しました！以下から結果をダウンロードしてください。")
                    with open(results_csv, "rb") as f:
                        st.download_button(
                            label="評価結果CSVをダウンロード (results.csv)",
                            data=f,
                            file_name="results.csv",
                            mime="text/csv"
                        )
                    with open(moodle_csv, "rb") as f:
                        st.download_button(
                            label="Moodle用CSVをダウンロード (grading_worksheet.csv)",
                            data=f,
                            file_name="grading_worksheet.csv",
                            mime="text/csv"
                        )
                    with open(feedback_zip, "rb") as f:
                        st.download_button(
                            label="フィードバックテキストをダウンロード (feedbacks.zip)",
                            data=f,
                            file_name="feedbacks.zip",
                            mime="application/zip"
                        )
                    st.write("**次のステップ**: grading_worksheet.csvをMoodleの課題 > 成績 > オフライン採点ワークシートインポートでアップロードしてください。")
                temp_csv.unlink(missing_ok=True)
                shutil.rmtree("downloads", ignore_errors=True)
                shutil.rmtree("feedbacks", ignore_errors=True)
                if results_csv: results_csv.unlink(missing_ok=True)
                if moodle_csv: moodle_csv.unlink(missing_ok=True)
                if feedback_zip: feedback_zip.unlink(missing_ok=True)