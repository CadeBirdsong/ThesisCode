import os
import json
import time
import re
import statistics
import collections 
from datetime import datetime, timezone
from langdetect import detect, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Settings
FOLDER = r"C:\Users\Cade\RealFusionProject"
OUTPUT_REPORT = os.path.join(FOLDER, "vader_sentiment_report.txt")
OUTPUT_JSON = os.path.join(FOLDER, "vader_all_sentiment_outputs.json")
IRRELEVANT_TOKEN = "IRRELEVANT"
BATCH_SIZE = 100 
DATE_FORMAT = '%Y-%m-%d' 

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def parse_date_or_none(d_str):
    if not d_str:
        return None
    try:
        if d_str.endswith("Z"):
            d_str = d_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(d_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None

# --- Pass 1: Article Analysis ---
start_analysis = time.time()
print("--- Starting Pass 1: Article Sentiment Analysis (VADER) ---")

all_sentiment_outputs = []
iterate = 0
total_processed = 0
total_relevant = 0
total_irrelevant = 0
total_failed_lang_detect = 0

print("Counting total articles...")
content_count = 0
log_files = [f for f in os.listdir(FOLDER) if f.endswith(".log")]
log_files = [f for f in log_files if f not in (os.path.basename(OUTPUT_REPORT), os.path.basename(OUTPUT_JSON))]

for file_name in log_files:
    file_path = os.path.join(FOLDER, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            content_count += sum(1 for article in data if "content" in article and article["content"])
    except Exception as e:
        print(f"Warning reading {file_name}: {e}")

print(f"Found approximately {content_count} articles with 'content'.")

for file_name in log_files:
    file_path = os.path.join(FOLDER, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

    articles_in_file = [article for article in data if article.get("content")]
    if not articles_in_file:
        continue

    for article_index, article in enumerate(articles_in_file):
        iterate += 1
        text = article.get("content")
        pub_date_obj = parse_date_or_none(article.get("estimatedPublishedDate"))
        article_id = article.get("id", f"file_{file_name}_index_{article_index}")
        print(f"Processing: Article {iterate}/{content_count} [File: {file_name}, Item: {article_index+1}/{len(articles_in_file)}]")

        if len(text) < 20:
            print(f"  Skipping: Content too short ({len(text)} chars).")
            total_failed_lang_detect += 1
            continue
        try:
            if detect(text[:500]) != 'en':
                print("  Skipping: Non-English detected.")
                total_failed_lang_detect += 1
                continue
        except LangDetectException:
            print("  Skipping: Language detection failed.")
            total_failed_lang_detect += 1
            continue

        # --- US Context Check ---
        if not re.search(r'\b(United States|U\.S\.|US)\b', text, re.IGNORECASE):
            print("  Result: Irrelevant (Not primarily about US)")
            total_irrelevant += 1
            all_sentiment_outputs.append({
                "id": article_id,
                "file": file_name,
                "pub_date_obj": pub_date_obj,
                "is_relevant": False,
                "score": IRRELEVANT_TOKEN,
                "justification": "Article is not primarily about the US."
            })
            continue

        # --- VADER Sentiment Analysis ---
        vader_scores = analyzer.polarity_scores(text)
        compound = vader_scores["compound"]
        # Map compound score (-1 to 1) to Likert scale 1 to 7
        score = int(round((compound + 1) / 2 * 4 + 1))
        # Determine sentiment category for justification
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        justification = f"VADER compound score is {compound:.2f}, indicating {sentiment} sentiment."

        total_processed += 1
        total_relevant += 1
        print(f"  Result: Relevant (US Context), Fusion Sentiment Score={score}")
        all_sentiment_outputs.append({
            "id": article_id,
            "file": file_name,
            "pub_date_obj": pub_date_obj,
            "is_relevant": True,
            "score": score,
            "justification": justification
        })

end_analysis = time.time()
print("\n--- Pass 1 Summary ---")
print(f"Total .log files scanned: {len(log_files)}")
print(f"Approximate articles found: {content_count}")
print(f"Articles processed: {total_processed}")
print(f"  Relevant (US Context): {total_relevant}")
print(f"  Irrelevant (Not US Context): {total_irrelevant}")
print(f"Articles skipped (Lang Detect/Short): {total_failed_lang_detect}")
print(f"Pass 1 duration: {end_analysis - start_analysis:.2f} seconds.")

# --- Save Intermediate Results ---
print(f"\nSaving intermediate results to {OUTPUT_JSON}...")
try:
    serializable_outputs = []
    for item in all_sentiment_outputs:
        new_item = item.copy()
        pub_date_obj = new_item.pop("pub_date_obj", None)
        new_item["pub_date"] = pub_date_obj.isoformat() if pub_date_obj else None
        serializable_outputs.append(new_item)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable_outputs, f, indent=2)
    print("Intermediate results saved.")
except Exception as e:
    print(f"Error saving intermediate results: {e}")

# --- Score Summary in OUTPUT_REPORT ---
# Filter out relevant articles with a valid numeric score
relevant_scores = [item["score"] for item in all_sentiment_outputs 
                   if item.get("is_relevant") and isinstance(item.get("score"), int)]

if relevant_scores:
    avg_score = statistics.mean(relevant_scores)
    median_score = statistics.median(relevant_scores)
    try:
        modes = statistics.multimode(relevant_scores)
        mode_score = ", ".join(map(str, sorted(modes))) if modes else "N/A"
    except Exception:
        mode_score = "N/A"
    freq_dist = collections.Counter(relevant_scores)
    freq_dist_str = ", ".join([f"Score {s}: {c}" for s, c in sorted(freq_dist.items())])
else:
    avg_score = median_score = None
    mode_score = freq_dist_str = "N/A"

try:
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as report_file:
        report_file.write("--- VADER SENTIMENT ANALYSIS SCORE SUMMARY ---\n")
        report_file.write(f"Total Articles Processed: {total_processed}\n")
        report_file.write(f"Relevant (US Context) Articles: {total_relevant}\n")
        report_file.write(f"Irrelevant Articles: {total_irrelevant}\n\n")
        report_file.write("--- Score Summary for Relevant Articles ---\n")
        if relevant_scores:
            report_file.write(f"Average Score: {avg_score:.2f}\n")
            report_file.write(f"Median Score: {median_score}\n")
            report_file.write(f"Mode Score(s): {mode_score}\n")
            report_file.write(f"Score Frequency: {freq_dist_str}\n")
        else:
            report_file.write("No relevant scores available.\n")
    print(f"\nReport saved to: {OUTPUT_REPORT}")
except Exception as e:
    print(f"Error saving report: {e}")

print(f"Intermediate data saved to: {OUTPUT_JSON}")
